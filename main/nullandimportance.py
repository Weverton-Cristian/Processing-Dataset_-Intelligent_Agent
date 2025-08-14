import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset_mapeado.csv')

colunas = [
    'idade','genero','etnia','pcd','vive_no_brasil','estado_moradia',
    'nivel_ensino','formacao','tempo_experiencia_dados','linguagens_preferidas',
    'cloud_preferida', 'sql', 'nosql'
]

print("🔍 Valores nulos por coluna (incluindo 0)")
print(((df == 0) | df.isna()).sum())


df_encoded = pd.get_dummies(df[colunas], drop_first=True)
X = df_encoded
y = df['cargo']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importancias = pd.Series(model.feature_importances_, index=X.columns)
print("\n📊 Importância das variáveis (RandomForest):")
print(importancias.sort_values(ascending=False).head(15))

# Colunas numéricas
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n🚨 Outliers detectados por coluna numérica (IQR)")
outliers_dict = {}

for col in colunas_numericas:
    serie = df[col].dropna()
    
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]
    outliers_dict[col] = len(outliers)
    print(f"{col}: {len(outliers)} outliers (Limites: {limite_inferior:.2f} a {limite_superior:.2f})")

    # Criar boxplot individual
    plt.figure(figsize=(5, 4))
    sns.boxplot(x=serie)
    plt.title(f"{col} — Outliers: {len(outliers)}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.show()
