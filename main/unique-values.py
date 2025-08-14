import pandas as pd

# Carregar e limpar
df = pd.read_csv("dataset_limpo.csv")
df = df.drop_duplicates()

# Remove linhas com 'cargo' vazio/nulo
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]


# Colunas para analisar
colunas = [
    "cargo",
    "formacao",
    "tempo_experiencia_dados",
    "linguagens_preferidas",
    "bancos_de_dados",
    "cloud_preferida",
    "nivel_ensino", "genero", "etnia", "pcd"
]

# Função para processar cada coluna
def contar_valores(coluna):
    contagem = {}
    for linha in df[coluna].dropna():
        # Divide por vírgula e remove espaços extras
        valores = [v.strip().lower() for v in str(linha).split(",") if v.strip()]
        for v in valores:
            contagem[v] = contagem.get(v, 0) + 1
    return contagem

# Mostrar resultados
for coluna in colunas:
    print(f"\n=== {coluna.upper()} ===")
    contagem = contar_valores(coluna)
    for valor, qtd in sorted(contagem.items(), key=lambda x: (-x[1], x[0])):
        print(f"{valor}: {qtd}")
