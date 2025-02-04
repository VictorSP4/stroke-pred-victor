from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Crear una instancia de Flask
app = Flask(__name__)

# Cargar el modelo entrenado de regresión logística (mejor resultado)
model_path = os.path.join("models", "log_reg.joblib")
model = joblib.load(model_path)

@app.route("/")
def home():
    return jsonify({"message": "Bienvenido a la API de prediccion de stroke. Usa /predict para hacer una prediccion."})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para recibir datos en formato JSON y devolver la predicción.
    Se espera un objeto JSON con la clave 'features' que contenga una lista de listas,
    donde cada sublista es un conjunto de datos de un paciente.
    """
    try:
        # Obtener datos JSON enviados por el usuario
        data = request.get_json(force=True)
        
        # Validar que 'features' está presente en el JSON
        features = data.get("features", None)
        if features is None:
            return jsonify({"error": "No se proporcionaron las características."}), 400

        # Convertir a un array numpy
        X = np.array(features)
        
        # Hacer la predicción
        y_pred = model.predict(X)

        # Devolver la predicción en formato JSON
        return jsonify({"prediction": y_pred.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ejecutar la API si este script es el principal
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
