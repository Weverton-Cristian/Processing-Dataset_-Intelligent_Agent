import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Carregar o novo dataset
df = pd.read_csv("dados_processados.csv")  # aqui seu arquivo já deve ter as colunas extras

df = df.dropna()
df = df.drop_duplicates()
df = df[df['cargo'] != -1]  # cargo numérico (int), não string '-1'

# 2. Definir as colunas de features incluindo bancos de dados
colunas_bancos = [
    'amazon','redshift','excel','azure','bigquery','cassandra','databricks',
    'db2','dynamodb','elaticsearch','firebase','firebird','google','hana',
    'hive','mariadb','microsoft','mongodb','mysql','oracle','postgresql',
    'presto','s3','snowflake','sqlserver'
]

# Colunas base + colunas dos bancos
colunas_features = [
    'idade','genero','etnia','pcd','estado_moradia','nivel_ensino',
    'formacao','tempo_experiencia_dados','linguagens_preferidas','cloud_preferida'
] + colunas_bancos

X = df[colunas_features]
y = df['cargo']

# 3. Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinar modelo Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. Fazer predição no teste
y_pred = model.predict(X_test)

# 6. Métricas e relatório
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo:", accuracy)

report = classification_report(y_test, y_pred)
print("Relatório de Classificação:\n", report)

# 7. Predição manual (exemplo) - ajuste as colunas com as novas colunas bancos = 0/1
prever = pd.DataFrame({
    'idade': [25],
    'genero': [0],
    'etnia': [0],
    'pcd': [0],
    'estado_moradia': [15],
    'nivel_ensino': [5],
    'formacao': [2],
    'tempo_experiencia_dados': [0],
    'linguagens_preferidas': [0],
    'cloud_preferida': [0],
    # Bancos (exemplo: usa mysql e sqlserver)
    'amazon': [0],
    'redshift': [0],
    'excel': [0],
    'azure': [0],
    'bigquery': [0],
    'cassandra': [0],
    'databricks': [0],
    'db2': [0],
    'dynamodb': [0],
    'elaticsearch': [0],
    'firebase': [0],
    'firebird': [0],
    'google': [0],
    'hana': [0],
    'hive': [0],
    'mariadb': [0],
    'microsoft': [0],
    'mongodb': [0],
    'mysql': [1],
    'oracle': [0],
    'postgresql': [0],
    'presto': [0],
    's3': [0],
    'snowflake': [0],
    'sqlserver': [1]
})

entrada = prever[colunas_features]

prever['cargo'] = model.predict(entrada)
print("Predição do cargo:", prever['cargo'][0])

# 8. Matriz de confusão
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
plt.show()

# 9. Boxplot da idade
sns.boxplot(x=df['idade'])
plt.show()

# 10. PCA para visualização
X_num = df[colunas_features]  # todas numéricas
X_scaled = StandardScaler().fit_transform(X_num)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.5, c=y, cmap='tab10')
plt.title("PCA dos dados")
plt.colorbar(label='Classe (cargo)')
plt.show()

# 11. Exemplo de erros
X_test_df = X_test.copy()
resultado_test = X_test_df.copy()
resultado_test['y_real'] = y_test.values
resultado_test['y_pred'] = y_pred
erros = resultado_test[resultado_test['y_real'] != resultado_test['y_pred']]

print("Exemplos onde o modelo errou:")
print(erros.head())
