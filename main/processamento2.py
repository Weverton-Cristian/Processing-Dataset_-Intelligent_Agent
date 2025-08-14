import pandas as pd
import numpy as np
import re
# from sklearn.impute import KNNImputer

print("🔍 Mapeamento + KNN Imputer")
print("=" * 60)

# =============================
# PARTE 1: PROCESSAMENTO PRINCIPAL
# =============================

print("\n📋 ETAPA 1: CARREGAMENTO E PROCESSAMENTO DOS DADOS")
print("-" * 50)

# Carregamento de dados (o CSV já tem cabeçalho)
df = pd.read_csv('dataset.csv')

# Tratamento inicial de valores ausentes
print("📋 Tratando valores ausentes iniciais...")
df = df.replace(r'^\s*$', pd.NA, regex=True)
print(f"Registros com campo genero vazio: {df['genero'].isna().sum()}")

print("Primeiras Linhas:")
print(df.head())

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

print("\nRegistros duplicados:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape após remoção de duplicados:", df.shape)

df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
df['idade'] = df['idade'].fillna(df['idade'].median())
df['idade'] = df['idade'].astype(int)

def extrair_sigla_estado(valor):
    if isinstance(valor, str):
        match = re.search(r'\((\w{2})\)', valor)
        if match:
            return match.group(1)
    return 'MISSING'

df['estado_moradia'] = df['estado_moradia'].apply(extrair_sigla_estado)

df = df.fillna('MISSING')

print("\n📋 APLICANDO MAPEAMENTOS MANUAIS...")

mapa_genero = {
    'Masculino': 0,
    'Feminino': 1,
    'MISSING': -1
}
df['genero'] = df['genero'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['genero'] = df['genero'].map(mapa_genero).fillna(-1).astype(int)

mapa_etnia = {
    'Branca': 0,
    'Parda': 1,
    'Preta': 2,
    'Amarela': 3,
    'Indígena': 4,
    'Prefiro não informar': 5,
    'MISSING': -1
}
df['etnia'] = df['etnia'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['etnia'] = df['etnia'].map(mapa_etnia).fillna(-1).astype(int)

mapa_pcd = {
    'Não': 0, 
    'False': 0,
    'Sim': 1,
    'True': 1,
    'MISSING': -1
}
df['pcd'] = df['pcd'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['pcd'] = df['pcd'].map(mapa_pcd).fillna(-1).astype(int)

mapa_cloud = {
    'Amazon Web Services (AWS)': 0,
    'Google Cloud (GCP)': 1,
    'Azure (Microsoft)': 2,
    'Não sei opinar': 3,
    'Não utilizo': 4,
    'Oracle Cloud': 5,
    'IBM Cloud': 6,
    'Outra opção': 7,
    'MISSING': -1
}
df['cloud_preferida'] = df['cloud_preferida'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['cloud_preferida'] = df['cloud_preferida'].map(mapa_cloud).fillna(-1).astype(int)

mapa_cargo = {
    'Cientista de Dados/Data Scientist': 0,
    'Analista de Dados/Data Analyst': 1,
    'Engenheiro de Dados/Data Engineer': 2,
    'Desenvolvedor/ Engenheiro de Software/ Analista de Sistemas': 3,
    'Analista de BI/BI Analyst': 4,
    'DBA/Administrador de Banco de Dados': 5,
    'Professor': 6,
    'Analista de Negócios/Business Analyst': 7,
    'Analista de Suporte/Analista Técnico': 8,
    'Engenheiro de Dados/Arquiteto de Dados/Data Engineer/Data Architect': 10,
    'Analytics Engineer ': 11,
    'Engenheiro de Machine Learning/ML Engineer': 12,
    'Product Manager/ Product Owner (PM/APM/DPM/GPM/PO)': 13,
    'Analista de Inteligência de Mercado/Market Intelligence': 14,
    'MISSING': -1
}
df['cargo'] = df['cargo'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['cargo'] = df['cargo'].map(mapa_cargo).fillna(-1).astype(int)

mapa_formacao = {
    'Computação / Engenharia de Software / Sistemas de Informação/ TI': 0,
    'Estatística/ Matemática / Matemática Computacional/ Ciências Atuariais': 1,
    'Outras Engenharias': 2,
    'Economia/ Administração / Contabilidade / Finanças/ Negócios': 3,
    'Ciências Biológicas/ Farmácia/ Medicina/ Área da Saúde': 4,
    'Ciências Sociais': 5,
    'Química / Física': 6,
    'Marketing / Publicidade / Comunicação / Jornalismo': 7,
    'Outra opção': 8,
    'MISSING': -1
}
df['formacao'] = df['formacao'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['formacao'] = df['formacao'].map(mapa_formacao).fillna(-1).astype(int)

mapa_estados = {
    'AC': 0, 'AL': 1, 'AP': 2, 'AM': 3, 'BA': 4, 'CE': 5, 'DF': 6, 'ES': 7, 'GO': 8,
    'MA': 9, 'MT': 10, 'MS': 11, 'MG': 12, 'PA': 13, 'PB': 14, 'PR': 15, 'PE': 16,
    'PI': 17, 'RJ': 18, 'RN': 19, 'RS': 20, 'RO': 21, 'RR': 22, 'SC': 23, 'SP': 24,
    'SE': 25, 'TO': 26, 'MISSING': -1
}
df['estado_moradia'] = df['estado_moradia'].apply(lambda x: mapa_estados.get(x, -1))

mapa_nivel = {
    'medio': 1,
    'tecnico': 2,
    'estudante de graduação': 3,
    'superior completo': 4,
    'pos-graduação': 5,
    'pós-graduação': 5,
    'mestrado': 6,
    'doutorado ou phd': 7,
    'MISSING': -1,
    'graduação/bacharelado': 4
}
df['nivel_ensino'] = df['nivel_ensino'].str.lower().map(mapa_nivel).fillna(-1).astype(int)

mapa_experiencia = {
    'menos de 1 ano': 0,
    '1 a 2 anos': 1,
    '1-2 anos': 1,
    'de 1 a 2 anos': 1,
    '2 a 3 anos': 2,
    '3 a 4 anos': 3,
    '3-5 anos': 4,
    '5-10 anos': 5,
    'mais de 10 anos': 6,
    'de 7 a 10 anos': 7, 
    'MISSING': -1
}
df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].str.lower().map(mapa_experiencia).fillna(-1).astype(int)

# =============================
# BANCOS DE DADOS
# =============================

lista_bancos = [
    'amazon', 'redshift', 'excel', 'azure', 'bigquery', 'cassandra', 'databricks',
    'db2', 'dynamodb', 'elaticsearch', 'firebase', 'firebird', 'google',
    'hana', 'hive', 'mariadb', 'microsoft', 'mongodb', 'mysql', 'oracle',
    'postgresql', 'presto', 's3', 'snowflake', 'sqlserver'
]

# Garantir texto limpo
df['bancos_de_dados'] = df['bancos_de_dados'].fillna('').astype(str).str.lower().str.strip()

mapa_bancos_regex = {
    'amazon': [r'\bamazon\b'],
    'redshift': [r'\bredshift\b'],
    'excel': [r'\bexcel\b'],
    'azure': [r'\bazure\b'],
    'bigquery': [r'\bbigquery\b', r'\bbig\s*query\b'],
    'cassandra': [r'\bcassandra\b'],
    'databricks': [r'\bdatabricks\b'],
    'db2': [r'\bdb2\b'],
    'dynamodb': [r'\bdynamodb\b'],
    'elaticsearch': [r'\belaticsearch\b'],
    'firebase': [r'\bfirebase\b'],
    'firebird': [r'\bfirebird\b'],
    'google': [r'\bgoogle\b'],
    'hana': [r'\bhana\b'],
    'hive': [r'\bhive\b'],
    'mariadb': [r'\bmariadb\b'],
    'microsoft': [r'\bmicrosoft\b'],
    'mongodb': [r'\bmongodb\b'],
    'mysql': [r'\bmysql\b', r'\bmy\s*sql\b'],       
    'oracle': [r'\boracle\b'],
    'postgresql': [r'\bpostgresql\b', r'\bpostgre\s*sql\b'],
    'presto': [r'\bpresto\b'],
    's3': [r'\bs3\b'],
    'snowflake': [r'\bsnowflake\b'],
    'sqlserver': [r'\bsqlserver\b', r'\bsql\s*server\b'] 
}

def encontra_banco(texto, padroes):
    texto = texto.lower() if isinstance(texto, str) else ''
    for padrao in padroes:
        if re.search(padrao, texto):
            return 1
    return 0

# Criar as colunas binárias
for banco, padroes in mapa_bancos_regex.items():
    df[banco] = df['bancos_de_dados'].apply(lambda x: encontra_banco(x, padroes))

# Remover coluna original
df = df.drop(columns=['bancos_de_dados'])

# =============================
# LINGUAGENS PREFERIDAS
# =============================
def extrair_principal(x):
    if isinstance(x, str) and x.strip() != '':
        items = [i.strip() for i in x.split(',')]
        return items[0] if items else 'MISSING'
    return 'MISSING'

df['linguagens_preferidas'] = df['linguagens_preferidas'].apply(extrair_principal)
mapa_linguagens = {
    'Python': 0,
    'R': 1,
    'SQL': 2,
    'javascript': 3,
    'Java': 4,
    'C/C++/C#': 5,
    'Scala': 6,
    'Go': 7,
    'Julia': 8,
    'SAS': 9,
    'Excel': 10,
    'Dax': 11,
    'M': 12,
    'Não uso': 13,
    'Não utilizo': 14,
    'MISSING': -1
}
df['linguagens_preferidas'] = df['linguagens_preferidas'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['linguagens_preferidas'] = df['linguagens_preferidas'].map(mapa_linguagens).fillna(-1).astype(int)

# =============================
# AJUSTE FINAL
# =============================
df = df.reset_index(drop=True)
df = df.fillna(-1)
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
df[colunas_numericas] = df[colunas_numericas].astype(int)
df = df.drop_duplicates()

print(f"\n📈 Total de colunas finais: {df.shape[1]}")
print(f"Colunas: {list(df.columns)}")

# print(f"\n✅ Verificação final:")
# for col in df.columns:
#     tipo = df[col].dtype
#     min_val = df[col].min()
#     max_val = df[col].max()
#     print(f"{col}: {tipo} (min: {min_val}, max: {max_val})")

# Remover coluna 'vive_no_brasil' antes de exportar
if 'vive_no_brasil' in df.columns:
    df = df.drop(columns=['vive_no_brasil'])

df.to_csv('processados.csv', index=False)
print("✅ Arquivo 'dados_processados.csv' salvo com sucesso.")

# for banco in lista_bancos:
#     count_usuarios = df[banco].sum()  # soma dos valores 1 em cada coluna
#     print(f"{banco}: {count_usuarios} usuário(s) usam")

