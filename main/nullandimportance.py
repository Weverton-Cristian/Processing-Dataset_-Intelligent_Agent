import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Carregar dataset
df = pd.read_csv('dataset_limpo.csv')

# Lista de colunas de interesse
colunas = [
    'idade','genero','etnia','pcd','vive_no_brasil','estado_moradia',
    'nivel_ensino','formacao','tempo_experiencia_dados','linguagens_preferidas',
    'bancos_de_dados','cloud_preferida','cargo'
]

# ============================
# 2. Contar valores nulos
# ============================
print("\n🔍 Valores nulos por coluna")
print(df[colunas].isnull().sum().sort_values(ascending=False))

# ============================
# 3. Importância das variáveis
# ============================
# Vamos supor que a coluna alvo é "target"
# Se não tiver, substitua pelo nome da sua coluna resposta
if 'target' in df.columns:
    df_encoded = pd.get_dummies(df[colunas], drop_first=True)
    X = df_encoded
    y = df['target']

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X, y)

    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    print("\n📊 Importância das variáveis (RandomForest):")
    print(importancias.sort_values(ascending=False).head(15))
else:
    print("\n⚠ Nenhuma coluna 'target' encontrada — importância das variáveis não calculada.")

# ============================
# 4. Identificação de outliers
# ============================
print("\n🚨 Outliers detectados por coluna numérica (IQR)")
for col in colunas:
    if pd.api.types.is_numeric_dtype(df[col]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)][col]
        print(f"{col}: {len(outliers)} outliers (Limites: {limite_inferior:.2f} a {limite_superior:.2f})")
