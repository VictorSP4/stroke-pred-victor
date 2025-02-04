# Proyecto de Clasificación - Data Science

Este proyecto implementa una solución de clasificación utilizando diferentes modelos de Machine Learning. Incluye una API con Flask y un análisis detallado de los modelos.

## 📌 Instrucciones para ejecutar el proyecto

### 1️⃣ Requisitos previos
Antes de ejecutar el proyecto, asegúrate de tener instalado:
- **Python 3.10.4 o superior**
- **pip**

### 2️⃣ Instalación de dependencias
Ejecuta el siguiente comando para instalar los paquetes necesarios:
```sh
pip install -r requirements.txt
```

### 3️⃣ Ejecutar la API Flask
Para iniciar la API, usa el siguiente comando:
```sh
python app.py
```
La API correrá en `http://127.0.0.1:5000/`

### 4️⃣ Probar la API
Para probar la API, usa `curl` o Postman:
```sh
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"features": [[1,2,3,4,5,6,7,8,9,10]]}'
```

### 5️⃣ Ejecutar el análisis de modelos
Para ejecutar la evaluación de modelos, corre:
```sh
python run_analysis.py
```
Esto cargará los modelos, realizará predicciones y mostrará métricas de rendimiento.

## 📂 Estructura del Proyecto
```
/mi_proyecto
│── models/               # Modelos entrenados
│── data/                 # Dataset usado
│── src/                  # Código fuente
│── app.py                # API Flask
│── run_analysis.py       # Evaluación de modelos
│── requirements.txt      # Dependencias
│── README.md             # Instrucciones
```

