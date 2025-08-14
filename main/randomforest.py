import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset_knn.csv") 


df = df.dropna() 
df = df[df['cargo'] != -1]  


colunas_base = [
    'idade', 'estado_moradia', 'tempo_experiencia_dados',
    'formacao', 'sql',
    'etnia', 'nivel_ensino', 'genero', 'linguagens_preferidas'
]

X = df[colunas_base]
y = df['cargo']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print("Acurácia:", acc)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Matriz de Confusão")
# plt.xlabel("Predito")
# plt.ylabel("Real")
# plt.show()

# importances = model.feature_importances_
# feat_names = X.columns
# plt.figure(figsize=(10, 8))
# sns.barplot(x=importances, y=feat_names)
# plt.title('Importância das Variáveis')
# plt.show()

# PCA para visualização
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X_test)

# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='tab10', alpha=0.7)
# plt.title("PCA - Visualização das Previsões")
# plt.xlabel("Componente Principal 1")
# plt.ylabel("Componente Principal 2")
# plt.colorbar(scatter, label='Classe prevista')
# plt.show()
