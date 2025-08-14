import pandas as pd
import numpy as np
import unidecode
import re

# Carregar e limpar
df = pd.read_csv("dataset.csv")
df = df.drop_duplicates()

# Remove linhas com 'cargo' vazio/nulo
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

# deixa TUDO em minusculo
df = df.map(lambda s: s.lower() if type(s) == str else s)

# Remove caracteres especiais e coloca eles como comuns
df = df.map(lambda s: unidecode.unidecode(s) if type(s) == str else s)

# Remove todos os espaços
df = df.map(lambda s: s.replace(" ", "") if type(s) == str else s)

# Transforma sim e nao em 0 e 1
df = df.replace({'sim': 1, 'nao': 0})

# Salvar CSV limpo
df.to_csv("dataset_limpo.csv", index=False)
