import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dados_processados.csv") 

df = df.dropna() 
df = df.drop_duplicates() 
df = df[df['cargo'] != '-1']


X = df[['idade','genero','etnia','pcd','estado_moradia','nivel_ensino','formacao','tempo_experiencia_dados','linguagens_preferidas','bancos_de_dados','cloud_preferida']]

y = df['cargo']

# 3. Dividindo treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Treinando o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Prevendo
y_pred = model.predict(X_test)

# 6. Métricas
acc = accuracy_score(y_test, y_pred)
print("Acurácia:", acc)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# 7. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Importância das features
importances = model.feature_importances_
feat_names = X.columns
sns.barplot(x=importances, y=feat_names)
plt.title('Importância das Variáveis')
plt.show()

# 8. Redução de Dimensionalidade com PCA (para visualização)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_test)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='tab10', alpha=0.7)
plt.title("PCA - Visualização das Previsões")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(scatter, label='Classe prevista')
plt.show()

# # 9. Erros: exemplos em que o modelo errou
# df_test = X_test.copy()
# df_test['y_real'] = y_test.values
# df_test['y_pred'] = y_pred
# erros = df_test[df_test['y_real'] != df_test['y_pred']]
# print("Exemplos onde o modelo errou:")
# print(erros.head())
