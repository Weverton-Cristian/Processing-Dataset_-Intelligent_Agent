import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dados_knn_imputed.csv')

X = df[['idade','genero','etnia','pcd','vive_no_brasil','estado_moradia','nivel_ensino','formacao','tempo_experiencia_dados','linguagens_preferidas','bancos_de_dados','cloud_preferida']]

y = df['cargo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)

report = classification_report(y_test, y_pred)
print("Relatório de Classificação:\n", report)

prever = pd.DataFrame({
  'idade': 25,
  'genero': 0,
  'etnia': 0,
  'pcd': 0,
  'vive_no_brasil': 1,
  'estado_moradia': 15,
  'nivel_ensino': 5,
  'formacao': 2,
  'tempo_experiencia_dados': 0,
  'linguagens_preferidas': 0,
  'bancos_de_dados': 4,
  'cloud_preferida': 0
}, index=[0])


entrada = prever[['idade','genero','etnia','pcd','vive_no_brasil','estado_moradia','nivel_ensino','formacao','tempo_experiencia_dados','linguagens_preferidas','bancos_de_dados','cloud_preferida']]

prever['cargo'] = model.predict(entrada)


print("Predição do cargo:", prever['cargo'][0])

matrix = confusion_matrix(y_test, y_pred)

print("Matriz de Confusão:\n", matrix)


plt.figure(figsize=(10, 7))
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = range(len(model.classes_))
plt.xticks(tick_marks, model.classes_, rotation=45)
plt.yticks(tick_marks, model.classes_)
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.tight_layout()
# plt.show()

sns.boxplot(x=df['idade'])
# plt.show()

X_num = df.select_dtypes(include=['int64', 'float64'])  # só variáveis numéricas
X_scaled = StandardScaler().fit_transform(X_num)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.5)
plt.title("PCA dos dados")
# plt.show()



print("VERIFICAR ERROS!!!")
# Converte X_test para DataFrame se ainda for numpy array
X_test_df = X_test.copy()

# Junta X_test, y_test e y_pred em um só DataFrame
resultado_test = X_test_df.copy()
resultado_test['y_real'] = y_test.values
resultado_test['y_pred'] = y_pred

# Filtra onde houve erro
erros = resultado_test[resultado_test['y_real'] != resultado_test['y_pred']]

print("Exemplos onde o modelo errou:")
print(erros.head())

