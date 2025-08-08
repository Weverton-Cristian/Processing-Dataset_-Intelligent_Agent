import pandas as pd
import numpy as np
import re

df = pd.read_csv("dataset.csv")
df = df.drop_duplicates()

df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

valores_unicos = set()

for linha in df["nivel_ensino"].dropna():
    bancos = [b.strip().lower() for b in linha.split(",")]
    valores_unicos.update(bancos)

for valor in sorted(valores_unicos):
    print(valor)