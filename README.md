# Proyecto Integrador - Data Mining (Boston Housing)

Este proyecto realiza un análisis completo del dataset de Boston Housing, desde la limpieza de datos hasta el despliegue de un dashboard predictivo.

## Estructura del Proyecto

* `data/raw/`: Dataset original.
* `data/processed/`: Dataset limpio y procesado (`clean_data.csv`) y evidencias de análisis.
* `src/`: Código fuente.
    * `clean_data.py`: Script de limpieza, normalización y clustering (Responsabilidad Integrante A).
    * `models/train.py`: Script de entrenamiento de modelos (Responsabilidad Integrante B y C).
* `requirements.txt`: Dependencias del proyecto.

## Responsabilidades - Integrante A

1. **Limpieza de Datos**: Tratamiento de nulos (mediana) y outliers (IQR Capping).
2. **Procesamiento**: Normalización mediante `StandardScaler` con justificación técnica.
3. **Selección de Variables**: Análisis de correlación de Pearson (ver `data/processed/correlation_matrix.png`).
4. **Algoritmo 1**: K-Means para segmentación de vecindarios (Aprendizaje No Supervisado).

## Ejecución

Para preparar los datos, ejecute:
```bash
python src/clean_data.py
```
