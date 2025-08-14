import pandas as pd
import numpy as np
import unidecode
import re

# apagar linhas duplicadas
# apagar linhas que a coluna cargo esteja vazia
# colocar tudo em minusculo, tirar caracteres especiais e coloca eles como comuns
# transformar sim e nao em 0 e 1

# Carregar e limpar
df = pd.read_csv("dataset.csv")
df = df.drop_duplicates()

# Remove linhas com 'cargo' vazio/nulo
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

# deixa TUDO em minusculo
df = df.applymap(lambda s: s.lower() if type(s) == str else s)

# Remove caracteres especiais e coloca eles como comuns
df = df.applymap(lambda s: unidecode.unidecode(s) if type(s) == str else s)

# Transforma sim e nao em 0 e 1
df = df.replace({'sim': 1, 'nao': 0})

novo_csv = df.to_csv("dataset_limpo.csv", index=False)