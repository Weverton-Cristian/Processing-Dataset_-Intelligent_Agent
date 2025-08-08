import pandas as pd
import numpy as np
import re

#preciso mudar esses dados todos para numeros pra treinar depois meu modelo.
# apagar coluna com "cargo" vazio

df = pd.read_csv("dataset.csv")
df = df.drop_duplicates()  

#funcao que retorna todos os valores unicos de todas as colunas

def get_unique_values(df):
    unique_values = {}
    for column in df.columns:
        unique_values[column] = df[column].unique().tolist()
    return unique_values  

get_unique_values(df)

mapa_genero = {
    'Masculino': 0,
    'Feminino': 1,
    'MISSING': -1
}

mapa_etnia = {
    'Branca': 0,
    'Parda': 1,
    'Preta': 2,
    'Amarela': 3,
    'Indígena': 4,
    'Prefiro não informar': 5,
    'MISSING': -1
}