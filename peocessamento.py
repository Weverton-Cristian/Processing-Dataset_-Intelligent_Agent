import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Carregamento de dados do Dataset
df = pd.read_csv('dataset.csv')
print(df.head())

# Tratar a coluna 'idade' separadamente antes de substituir strings por 'MISSING'
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')  # transforma textos inválidos em NaN
df['idade'] = df['idade'].fillna(df['idade'].median())     # substitui NaN pela mediana
df['idade'] = df['idade'].astype(int)

# Tratar valores ausentes nas demais colunas (categóricas)
df = df.fillna('MISSING')  # substitui valores ausentes por 'MISSING'

# Label Encoding para colunas com 1 valor por célula
colunas_label = ['genero', 'etnia', 'pcd', 'vive_no_brasil', 'estado_moradia',
                 'cloud_preferida', 'cargo']

label_encoders = {}
for col in colunas_label:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Tratamento específico para 'formacao' com LabelEncoder
if 'formacao' in df.columns:
    le_formacao = LabelEncoder()
    df['formacao'] = le_formacao.fit_transform(df['formacao'].astype(str))

# Mapeando nível de Escolaridade manualmente
mapa_nivel = {
    'fundamental': 0,
    'medio': 1,
    'tecnico': 2,
    'superior incompleto': 3,
    'superior completo': 4,
    'pos-graduacao': 5,
    'mestrado': 6,
    'doutorado': 7,
    'MISSING': -1
}
df['nivel_ensino'] = df['nivel_ensino'].map(mapa_nivel).fillna(-1).astype(int)

# Mapeando tempo de experiência
mapa_experiencia = {
    'menos de 1 ano': 0,
    '1-3 anos': 1,
    '3-5 anos': 2,
    '5-10 anos': 3,
    'mais de 10 anos': 4,
    'MISSING': -1
}
df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].map(mapa_experiencia).fillna(-1).astype(int)

# Função para garantir que split funcione mesmo se valor for 'MISSING' ou vazio
def safe_split(x):
    if isinstance(x, str) and x != 'MISSING' and x.strip() != '':
        return [i.strip() for i in x.split(',')]
    else:
        return []

# Linguagens preferidas
df['linguagens_preferidas'] = df['linguagens_preferidas'].apply(safe_split)
mlb1 = MultiLabelBinarizer()
linguagens_bin = pd.DataFrame(mlb1.fit_transform(df['linguagens_preferidas']),
                              columns=[f"ling_{l}" for l in mlb1.classes_])

# Bancos de dados
df['bancos_de_dados'] = df['bancos_de_dados'].apply(safe_split)
mlb2 = MultiLabelBinarizer()
bd_bin = pd.DataFrame(mlb2.fit_transform(df['bancos_de_dados']),
                      columns=[f"bd_{b}" for b in mlb2.classes_])

# Concatenar no DataFrame
df = pd.concat([df, linguagens_bin, bd_bin], axis=1)
df = df.drop(columns=['linguagens_preferidas', 'bancos_de_dados'])

# Garantir que tudo é inteiro
df = df.fillna(-1)  # caso ainda tenha algum valor faltando
df = df.astype(int)  # tudo vira inteiro

# Salvar arquivo preprocessado
df.to_csv('dados_processados.csv', index=False)
