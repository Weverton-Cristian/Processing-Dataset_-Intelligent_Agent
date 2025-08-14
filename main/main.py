import pandas as pd
import numpy as np
import unidecode
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# Função para limpar strings
def limpar_texto(s):
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = unidecode.unidecode(s)  # remove acentos
    return s

# Carrega dados
df = pd.read_csv('dataset.csv')

# Limpa colunas de interesse
colunas_texto = ['cargo', 'formacao', 'tempo_experiencia_dados', 
                 'linguagens_preferidas', 'bancos_de_dados', 'cloud_preferida']

for col in colunas_texto:
    df[col] = df[col].apply(limpar_texto)

# Mapeia experiencia para ordinal
map_exp = {
    'nao tenho experiencia na area de dados': 0,
    'menos de 1 ano': 1,
    'de 1 a 2 anos': 2,
    'de 3 a 4 anos': 3,
    'de 4 a 6 anos': 4,
    'de 7 a 10 anos': 5,
    'mais de 10 anos': 6
}
df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].map(map_exp)

# Agrupamento de linguagens
top_langs = ['python', 'r', 'sql', 'scala']
df['linguagens_preferidas'] = df['linguagens_preferidas'].apply(
    lambda x: x if x in top_langs else 'outra'
)

# Agrupamento de bancos (exemplo simples: SQL, NoSQL, Cloud, Outro)
def agrupar_banco(b):
    if pd.isna(b): return b
    sql_terms = ['mysql', 'postgresql', 'sql server', 'sqlite', 'oracle', 'mariadb']
    nosql_terms = ['mongodb', 'redis', 'cassandra', 'neo4j', 'dynamodb']
    cloud_terms = ['google bigquery', 's3', 'snowflake', 'amazon redshift', 'amazon athena']
    if b in sql_terms: return 'sql'
    if b in nosql_terms: return 'nosql'
    if b in cloud_terms: return 'cloud'
    return 'outro'
df['bancos_categoria'] = df['bancos_de_dados'].apply(agrupar_banco)

# Feature: quantidade de bancos
df['qtd_bancos_conhece'] = df['bancos_de_dados'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) else 0
)

# One-Hot Encoding para colunas categóricas
categoricas = ['cargo', 'formacao', 'linguagens_preferidas', 'cloud_preferida', 'bancos_categoria']
df = pd.get_dummies(df, columns=categoricas, drop_first=False)

# Substitui -1 por NaN e aplica KNN Imputer
df = df.replace(-1, np.nan)
imputer = KNNImputer(n_neighbors=5)
df[df.columns] = imputer.fit_transform(df)

# Separa features e target
X = df.drop(columns=['cargo'])  # ajuste conforme sua variável target
y = df['cargo']

# Balanceamento
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Salva
df_bal = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name='cargo')], axis=1)
df_bal.to_csv('processados_knn_smote.csv', index=False)
print("Arquivo processados_knn_smote.csv salvo com dados imputados e balanceados.")
