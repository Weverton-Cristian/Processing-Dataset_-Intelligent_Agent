import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Definição dos nomes das colunas (caso o CSV não tenha cabeçalho)
colunas = [
    'idade', 'genero', 'etnia', 'pcd', 'vive_no_brasil', 'estado_moradia',
    'nivel_ensino', 'formacao', 'tempo_experiencia_dados', 'idioma',
    'bancos_de_dados', 'linguagens_preferidas', 'cloud_preferida', 'cargo'
]

# Carregamento de dados
df = pd.read_csv('dataset.csv', names=colunas)
print("Primeiras Linhas:")
print(df.head())

# =============================
# PROCESSO DE LIMPEZA
# =============================

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

print("\nRegistros duplicados:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape após remoção de duplicados:", df.shape)

# Corrigir idade
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
df['idade'] = df['idade'].fillna(df['idade'].median())
df['idade'] = df['idade'].astype(int)

# Normalizar valores de 'vive_no_brasil' ANTES do label encoding
df['vive_no_brasil'] = df['vive_no_brasil'].astype(str).str.strip().str.lower()
df['vive_no_brasil'] = df['vive_no_brasil'].replace({
    'true': 'sim',
    'false': 'não',
    'nan': 'MISSING',
    'missing': 'MISSING'
})

# Extrair sigla do estado
def extrair_sigla_estado(valor):
    if isinstance(valor, str):
        match = re.search(r'\((\w{2})\)', valor)
        if match:
            return match.group(1)
    return 'MISSING'

df['estado_moradia'] = df['estado_moradia'].apply(extrair_sigla_estado)

# Preencher valores ausentes
df = df.fillna('MISSING')

# Label encoding para colunas categóricas
colunas_label = ['genero', 'etnia', 'pcd', 'vive_no_brasil', 'cloud_preferida', 'cargo']
label_encoders = {}

for col in colunas_label:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Obter valor codificado de 'sim' para 'vive_no_brasil'
valor_sim_original = 'sim'
le_vive = label_encoders['vive_no_brasil']
sim_value = le_vive.transform([valor_sim_original])[0]

# Mapeamento de estados brasileiros (siglas)
mapa_estados = {
    'AC': 0, 'AL': 1, 'AP': 2, 'AM': 3, 'BA': 4, 'CE': 5, 'DF': 6, 'ES': 7, 'GO': 8,
    'MA': 9, 'MT': 10, 'MS': 11, 'MG': 12, 'PA': 13, 'PB': 14, 'PR': 15, 'PE': 16,
    'PI': 17, 'RJ': 18, 'RN': 19, 'RS': 20, 'RO': 21, 'RR': 22, 'SC': 23, 'SP': 24,
    'SE': 25, 'TO': 26, 'MISSING': -1
}

# Aplicar mapeamento condicional de estado_moradia
df['estado_moradia'] = df.apply(
    lambda row: mapa_estados.get(row['estado_moradia'], -1)
    if row['vive_no_brasil'] == sim_value else -1,
    axis=1
)

# Label encoder para 'formacao' (se existir)
if 'formacao' in df.columns:
    le_formacao = LabelEncoder()
    df['formacao'] = le_formacao.fit_transform(df['formacao'].astype(str))

# Mapeamento do nível de ensino
mapa_nivel = {
    'fundamental': 0,
    'medio': 1,
    'tecnico': 2,
    'superior incompleto': 3,
    'superior completo': 4,
    'pos-graduação': 5,
    'pós-graduação': 5,
    'mestrado': 6,
    'doutorado': 7,
    'MISSING': -1,
    'graduação/bacharelado': 4
}
df['nivel_ensino'] = df['nivel_ensino'].str.lower().map(mapa_nivel).fillna(-1).astype(int)

# Mapeamento do tempo de experiência
mapa_experiencia = {
    'menos de 1 ano': 0,
    '1 a 2 anos': 1,
    '1-2 anos': 1,
    'de 1 a 2 anos': 1,
    '2 a 3 anos': 2,
    '3 a 4 anos': 2,
    '3-5 anos': 2,
    '5-10 anos': 3,
    'mais de 10 anos': 4,
    'MISSING': -1
}
df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].str.lower().map(mapa_experiencia).fillna(-1).astype(int)

# Função para dividir listas com segurança
def safe_split(x):
    if isinstance(x, str) and x != 'MISSING' and x.strip() != '':
        return [i.strip() for i in x.split(',')]
    else:
        return []

# Aplicar MultiLabelBinarizer para bancos de dados
df['bancos_de_dados'] = df['bancos_de_dados'].apply(safe_split)
mlb_bd = MultiLabelBinarizer()
bd_bin = pd.DataFrame(mlb_bd.fit_transform(df['bancos_de_dados']),
                      columns=[f"bd_{b}" for b in mlb_bd.classes_])
df = pd.concat([df, bd_bin], axis=1)
df = df.drop(columns=['bancos_de_dados'])

# Aplicar MultiLabelBinarizer para linguagens preferidas
df['linguagens_preferidas'] = df['linguagens_preferidas'].apply(safe_split)
mlb_ling = MultiLabelBinarizer()
ling_bin = pd.DataFrame(mlb_ling.fit_transform(df['linguagens_preferidas']),
                        columns=[f"ling_{l}" for l in mlb_ling.classes_])
df = pd.concat([df, ling_bin], axis=1)
df = df.drop(columns=['linguagens_preferidas'])

# Resetar índice para evitar problemas com índices não numéricos
df = df.reset_index(drop=True)

# Diagnóstico: verificar se restam colunas texto antes da conversão final
print("\nColunas do tipo 'object' antes da conversão para inteiro:")
print(df.dtypes[df.dtypes == 'object'])

# Preencher valores ausentes e converter colunas numéricas para int
df = df.fillna(-1)
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
df[colunas_numericas] = df[colunas_numericas].astype(int)

# Salvar resultado
df.to_csv('dados_processados.csv', index=False)
print("✅ Arquivo 'dados_processados.csv' salvo com sucesso.")
