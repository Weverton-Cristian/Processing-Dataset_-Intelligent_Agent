import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 1. Ler o CSV
df = pd.read_csv('dataset_mapeado.csv')
df = df[df['cargo'] != -1] 

# 2. Definir colunas para imputação (removendo cargo e idade)
colunas_features = df.columns.drop(['cargo', 'idade']).tolist()

# 3. Guardar tipos originais
tipos_originais = df[colunas_features].dtypes

# 4. Substituir -1 por NaN
df[colunas_features] = df[colunas_features].replace(-1, np.nan)

# 5. Aplicar KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[colunas_features] = imputer.fit_transform(df[colunas_features])

# 6. Restaurar colunas inteiras para int
for col in colunas_features:
    if np.issubdtype(tipos_originais[col], np.integer):
        df[col] = df[col].round().astype(int)

# 7. Salvar CSV
df.to_csv('dataset_knn.csv', index=False, encoding='utf-8-sig')
print("✅ Arquivo 'dataset_knn.csv' salvo com dados imputados via KNN.")
