import pandas as pd
import numpy as np
import unidecode
import re

df = pd.read_csv('dataset_limpo.csv')

def mapear(csv): 
  def extrair_sigla_estado(valor):
    if isinstance(valor, str):
      match = re.search(r'\b[A-Z]{2}\b', valor)
      if match:
        return match.group(1).lower()
      elif len(valor.strip()) == 2:
        return valor.strip().lower()
    return -1

  def agrupar_valores_raros(valor, validos, label="outros"):
    if valor in validos:
      return valor
    if valor == "" or valor is None:
        return ""
    return label

  if 'estado_moradia' in df.columns:
    df['estado_moradia'] = df['estado_moradia'].apply(extrair_sigla_estado)

    mapa_estados = {
        'ac': 0, 'al': 1, 'ap': 2, 'am': 3, 'ba': 4, 'ce': 5, 
        'df': 6, 'es': 7, 'go': 8, 'ma': 9, 'mt': 10, 'ms': 11, 
        'mg': 12, 'pa': 13, 'pb': 14, 'pr': 15, 'pe': 16,
        'pi': 17, 'rj': 18, 'rn': 19, 'rs': 20, 'ro': 21, 
        'rr': 22, 'sc': 23, 'sp': 24, 'se': 25, 'to': 26, '': -1
    }
    df['estado_moradia'] = df['estado_moradia'].map(mapa_estados).fillna(-1).astype(int)

  
