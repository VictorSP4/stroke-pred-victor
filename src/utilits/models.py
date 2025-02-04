import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim

# mover device a gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir la red neuronal en PyTorch
class NeuralNet(nn.Module):
    def __init__(self, input_dim, neurons=64, num_layers=1, activation=nn.ReLU, dropout_rate=0.2):
        super(NeuralNet, self).__init__()
        layers = []
        
        # Capa de entrada
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(activation())

        # Capas ocultas
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(neurons, neurons // 2))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            neurons = neurons // 2  # Reducir las neuronas en cada capa

        # Capa de salida
        layers.append(nn.Linear(neurons, 1))  
        layers.append(nn.Sigmoid())  # <-- Para hacerla igual a Keras

        self.network = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.network(x)

# 🔹 Clase Adaptadora para usar la Red en Sklearn Pipeline
class TorchClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = 'classifier'
    
    def __init__(self, input_dim, epochs=50, batch_size=32, lr=0.001, neurons=64, num_layers=1):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.neurons = neurons
        self.num_layers = num_layers
        self.model = NeuralNet(input_dim, neurons=neurons, num_layers=num_layers)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.is_fitted = False

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).view(-1, 1)  # Asegurar la dimensión correcta

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        
        self.n_features_in_ = 10
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("❌ Error: El modelo `TorchClassifier` no ha sido entrenado. Debes llamar a `fit()` antes de `predict()`.")  

        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)

            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)
            outputs = self.model(X_tensor)
            predictions = (outputs >= 0.5).float().cpu()
        return predictions.numpy().astype(int)





# Definir los pipelines explícitamente
pipelines = {
    "LogisticRegression": ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=0)),
        ('logreg', LogisticRegression(max_iter=5000, random_state=0))
    ]),
    "SVC": ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=0)),
        ('svc', SVC(random_state=0))
    ]),
    "RandomForest": ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=0)),
        ('rf', RandomForestClassifier(random_state=0))
    ]),
    "NeuralNet": ImbPipeline([
        ('scaler', StandardScaler()),  
        ('smote', SMOTE(random_state=0)),  
        ('torch_nn', TorchClassifier(input_dim=10, epochs=50, batch_size=32, lr=0.001)) 
    ])
}



def calculate_metrics(estimator, X, y, cv):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f3_scores = []
    
    for train_index, test_index in cv.split(X, y):
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        y_series = pd.Series(y) if isinstance(y, np.ndarray) else y
        X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train, y_test = y_series.iloc[train_index], y_series.iloc[test_index]
        
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=1))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=1))
        f3_scores.append(fbeta_score(y_test, y_pred, beta=3, zero_division=1))
    
    return {
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f3_score': np.mean(f3_scores)
    }
