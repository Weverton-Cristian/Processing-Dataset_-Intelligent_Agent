import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dados_tratados.csv')

# --- Função para categorizar bancos de dados ---
def categorizar_banco(db_name):
    sql_dbs = ['sqlserver', 'mysql', 'postgresql', 'oracle', 'sqlite', 'mariadb', 'db2', 'sql', 'azure sql', 'mssql', 'amazon aurora ou rds', 'amazon redshift', 'amazon athena', 'databricks sql', 'snowflake', 'synapse']
    nosql_dbs = ['mongodb', 'cassandra', 'dynamodb', 'hbase', 'firebase', 'firestore', 'redis', 'couchbase', 'neo4j', 'datomic', 'hive', 'hadoop', 'presto', 'elasticsearch', 'solr', 'clickhouse', 'bigtable']
    cloud_dbs = ['google bigquery', 'aws', 'amazon', 'google cloud storage', 'gcp', 'azure', 'aws s3', 'amazon s3', 'azure blob storage', 'azure data lake', 'google firestore', 'google sheets', 'google analytics', 'google', 'bigquery', 'google big query']
    
    db_name = db_name.strip().lower()
    
    if db_name in sql_dbs:
        return 'sql'
    elif db_name in nosql_dbs:
        return 'nosql'
    elif db_name in cloud_dbs:
        return 'cloud'
    else:
        return 'outros'

# --- Carregue seu dataframe (exemplo) ---
# df = pd.read_csv('seu_arquivo.csv')

# --- 1. Remove linhas sem cargo ---
df = df.dropna(subset=['cargo'])
df = df[df['cargo'].astype(str).str.strip() != '']

# --- 2. Mapeia tempo de experiência para número ---
tempo_map = {
    'menos de 1 ano': 0.5,
    '1 a 2 anos': 1.5,
    '3 a 4 anos': 3.5,
    '4 a 6 anos': 5,
    '7 a 10 anos': 8.5,
    'mais de 10 anos': 12,
    'nao_informado': 0,
    'não_informado': 0
}
df['tempo_experiencia_num'] = df['tempo_experiencia_dados'].map(tempo_map).fillna(0)

# --- 3. Trata bancos de dados ---
df['bancos_list'] = df['bancos_de_dados'].astype(str).str.lower().str.replace(' ', '').str.split(',')

def categorize_bancos_list(bancos):
    categorias = {'sql':0, 'nosql':0, 'cloud':0, 'outros':0}
    for banco in bancos:
        cat = categorizar_banco(banco)
        categorias[cat] = 1
    return pd.Series(categorias)

bancos_cat = df['bancos_list'].apply(categorize_bancos_list)
df = pd.concat([df, bancos_cat], axis=1)

# --- 4. Label Encoding para colunas com poucas categorias ---
label_encode_cols = ['genero', 'etnia', 'pcd', 'vive_no_brasil', 'formacao', 'cloud_preferida', 'linguagens_preferidas']

for col in label_encode_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col+'_num'] = le.fit_transform(df[col])

# --- 5. One-hot encoding para cargo ---
df = pd.get_dummies(df, columns=['cargo'], prefix='cargo')

# --- 6. Converte colunas numéricas ---
num_cols = ['idade', 'estado_moradia', 'nivel_ensino', 'linguagem_principal', 'cloud_principal', 'linguagens_preferidas']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- 7. Remove colunas originais textuais ---
cols_to_drop = ['bancos_de_dados', 'bancos_list', 'tempo_experiencia_dados'] + label_encode_cols
df.drop(columns=cols_to_drop, inplace=True)

# --- Pronto! ---
print("Dataframe tratado com shape:", df.shape)
print("Colunas finais:", df.columns)

categorizar_banco("dados_tratados.csv")
# Salva o DataFrame tratado em disco
# df = df.astype({col: 'int' for col in df.select_dtypes(include='bool').columns})

df.to_csv('dados_tratados_processado.csv', index=False)

print("Arquivo salvo: dados_tratados_processado.csv")

