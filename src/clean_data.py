import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Rutas de archivos
INPUT_PATH = "data/raw/housing.csv"
OUTPUT_PATH = "data/processed/clean_data.csv"

# Columnas estándar del dataset Boston Housing
COLUMN_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

def handle_missing_values(df):
    """
    Evidencia de tratamiento de nulos.
    JUSTIFICACIÓN: Se utiliza la mediana para la imputación ya que es más robusta 
    a los outliers presentes en el dataset de Boston que la media.
    """
    print("Revisando valores nulos...")
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median())
        print("Valores nulos imputados con la mediana.")
    else:
        print("No se encontraron valores nulos.")
    return df

def cap_outliers_iqr(df, column):
    """
    Limita los outliers usando el método IQR (Capping).
    JUSTIFICACIÓN: El dataset de Boston contiene variables con distribuciones sesgadas 
    (como CRIM o B). El capping (limitar a 1.5 * IQR) permite mantener la información 
    sin que los valores extremos distorsionen los modelos de regresión y clustering.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

def visualize_correlations(df):
    """
    Genera y guarda una matriz de correlación.
    EVIDENCIA: Se analiza la correlación de Pearson para identificar colinealidad.
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación - Dataset Boston")
    plt.tight_layout()
    plot_path = "data/processed/correlation_matrix.png"
    plt.savefig(plot_path)
    print(f"Evidencia de análisis de variables guardada en: {plot_path}")

def feature_selection(df):
    """
    Selección de variables eliminando colinealidad evidente.
    JUSTIFICACIÓN: RAD y TAX suelen tener correlación > 0.9 en Boston, lo que causa 
    multicolinealidad. Se elimina TAX para mejorar la estabilidad de los coeficientes.
    """
    print("Realizando selección de variables...")
    visualize_correlations(df)
    if 'TAX' in df.columns:
        df = df.drop(columns=['TAX'])
        print("Columna 'TAX' eliminada por alta colinealidad con 'RAD'.")
    return df

def main():
    # 1. Cargar datos
    print("Cargando datos originales...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # El dataset de Boston raw no tiene encabezados y usa espacios como delimitador
    try:
        df = pd.read_csv(INPUT_PATH, sep=r'\s+', names=COLUMN_NAMES, header=None)
    except Exception as e:
        print(f"Error al cargar con sep=r'\\s+': {e}")
        df = pd.read_csv(INPUT_PATH, delim_whitespace=True, names=COLUMN_NAMES, header=None)
    
    # 2. Tratamiento de nulos
    df = handle_missing_values(df)
    
    # 3. Tratamiento de Outliers
    outlier_cols = ['CRIM', 'ZN', 'B', 'LSTAT']
    print(f"Aplicando Capping de Outliers (IQR) a las columnas: {outlier_cols}")
    for col in outlier_cols:
        if col in df.columns:
            df = cap_outliers_iqr(df, col)
            
    # 4. Selección de Variables
    df = feature_selection(df)
    
    # Separamos la variable objetivo (MEDV)
    y = df['MEDV']
    X = df.drop(columns=['MEDV'])
    
    # 5. Normalización (StandardScaler)
    # JUSTIFICACIÓN: Los algoritmos basados en distancia (K-Means, SVR) son sensibles 
    # a la escala. StandardScaler (z-score) asegura que todas las variables 
    # contribuyan equitativamente al modelo.
    print("Normalizando datos...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 6. Algoritmo 1: K-Means (Aprendizaje No Supervisado)
    # JUSTIFICACIÓN: Segmentar los vecindarios ayuda a entender patrones de precios 
    # antes de aplicar modelos predictivos.
    print("Ejecutando K-Means para segmentación de vecindarios...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    X_scaled['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Unificamos todo de vuelta
    df_final = X_scaled.copy()
    df_final['MEDV'] = y.values
    
    # 7. Guardar resultados
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"¡Éxito! Datos limpios y procesados guardados en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
