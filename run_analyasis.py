import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, fbeta_score

models_dir = "models"
model_files = {
    "LogisticRegression": os.path.join(models_dir, "log_reg.joblib"),
    "SVC": os.path.join(models_dir, "svc.joblib"),
    "RandomForest": os.path.join(models_dir, "random_forest.joblib"),
    "NeuralNet": os.path.join(models_dir, "neural_net_pipeline.joblib")
}

data_path = "data/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(data_path)
df = df.drop(columns=["id"])

df = df.replace({
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'ever_married': {'Yes': 0, 'No': 1},
    'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4},
    'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3},
    'Residence_type': {'Urban': 0, 'Rural': 1}
})

knn_imputer = KNNImputer(n_neighbors=3)
df[df.columns] = knn_imputer.fit_transform(df)

target_col = 'stroke'
X = df.drop([target_col], axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

loaded_models = {}
for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        loaded_models[model_name] = joblib.load(file_path)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f3 = fbeta_score(y_test, y_pred, beta=3, zero_division=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap="Blues")
    print(f"\nMétricas para {model_name}:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F3 Score : {f3:.4f}")
    return accuracy, precision, recall, f3

evaluations = {}
for model_name, model in loaded_models.items():
    evaluations[model_name] = evaluate_model(model, X_test, y_test, model_name)

print("\n✅ Evaluación completada con éxito.")
