from sklearn.model_selection import GridSearchCV
from torch import nn

from src.utilits.models import pipelines, NeuralNet, TorchClassifier, calculate_metrics

# Espacios de búsqueda de hiperparámetros para cada modelo

# para la regresión logística
param_grid_logreg = {
    'logreg__solver': ['liblinear', 'saga'],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__max_iter': [1000, 5000, 10000, 20000],
    'logreg__class_weight': [None],
    'logreg__C': [0.0001, 0.001, 1, 100, 1000, 10000]
}

# para SVC
param_grid_svc = {
    'svc__C': [0.01, 0.1, 1, 10],
    'svc__gamma': [1, 0.1, 0.01, 0.001],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# para Random Forest
param_grid_rf = {
    'rf__n_estimators': [100, 500, 1000],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__max_depth': [10, 50, 100],
    'rf__min_samples_split': [2, 10, 20],
    'rf__min_samples_leaf': [1, 10, 20],
    'rf__bootstrap': [True],
    'rf__criterion': ['gini', "entropy"]
}

# para la red neuronal con Torch (adaptación del grid para Keras)
param_grid_torch = {
    'torch_nn__lr': [0.001, 0.0001],
    'torch_nn__neurons': [32, 64, 128],
    'torch_nn__num_layers': [2, 3, 4],
    'torch_nn__epochs': [50],
    'torch_nn__batch_size': [16, 32, 64]
}

# Diccionario que asocia cada modelo con su espacio de búsqueda
param_grids = {
    "LogisticRegression": param_grid_logreg,
    "SVC": param_grid_svc,
    "RandomForest": param_grid_rf,
    "NeuralNet": param_grid_torch
}