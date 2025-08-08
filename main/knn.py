import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.read_csv('dados_processados.csv')

colunas_features = df.columns.drop('cargo').tolist()
colunas_features = df.columns.drop('idade').tolist()


df[colunas_features] = df[colunas_features].replace(-1, np.nan)

imputer = KNNImputer(n_neighbors=5)
df[colunas_features] = imputer.fit_transform(df[colunas_features])

for col in colunas_features:
    # Se a coluna original era int, arredondar e converter para int
    if pd.api.types.is_integer_dtype(df[col].dropna()):
        df[col] = df[col].round().astype(int)

df.to_csv('processados_knn.csv', index=False)
print("Arquivo 'processados_knn.csv' salvo com dados imputados via KNN.")
