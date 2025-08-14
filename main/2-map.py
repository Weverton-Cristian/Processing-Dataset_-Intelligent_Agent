import pandas as pd
import numpy as np
import unidecode
import re

df = pd.read_csv('dataset_limpo.csv')

def agrupar_valores_raros(df, coluna, mapa, preprocess=None):
    def mapear(valor):
        if pd.isna(valor) or str(valor).strip() == "":
            return -1  # ausência de valor
        if preprocess:
            valor = preprocess(valor)
        valor_norm = str(valor).strip().replace('"', '').replace("'", "")
        return mapa.get(valor_norm, -1)

    df[coluna] = df[coluna].apply(mapear).astype(int)



def extrair_estado(valor):
    if isinstance(valor, str):
        match = re.search(r'\((\w{2})\)', valor)
        if match:
            return match.group(1)
    return valor

# listas de bancos
bancos_sql = {
    'sqlserver','mysql','postgresql','oracle','googlebigquery',
    'sqlite','saphana','snowflake','amazonauroraourds','mariadb',
    'db2','firebird','amazonredshift','microsoftaccess'
}

bancos_nosql = {
    's3','databricks','amazonathena','mongodb','hive','dynamodb',
    'presto','elaticsearch','redis','firebase','splunk','nenhum',
    'cassandra','hbase','googlefirestore','neo4j','excel'
}

def contar_bancos(row):
    if pd.isna(row) or str(row).strip() == "":
        return pd.Series([0, 0])  # ausência → 0 para ambos
    bancos = [b.strip().lower() for b in str(row).split(",")]
    qtd_sql = sum(1 for b in bancos if b in bancos_sql)
    qtd_nosql = sum(1 for b in bancos if b in bancos_nosql)
    return pd.Series([qtd_sql, qtd_nosql])

def mapear(df):
    if 'estado_moradia' in df.columns:
        mapa_estado_moradia = {
            'ac': 27,'al': 1,'ap': 2,'am': 3,'ba': 4,'ce': 5,
            'df': 6,'es': 7,'go': 8,'ma': 9,'mt': 10,'ms': 11,
            'mg': 12,'pa': 13,'pb': 14,'pr': 15,'pe': 16,
            'pi': 17,'rj': 18,'rn': 19,'rs': 20,'ro': 21,
            'rr': 22,'sc': 23,'sp': 24,'se': 25,'to': 26
        }
        agrupar_valores_raros(df, 'estado_moradia', mapa_estado_moradia, preprocess=extrair_estado)

    if 'genero' in df.columns:
        mapa_genero = {'masculino': 2,'feminino': 1}
        agrupar_valores_raros(df, 'genero', mapa_genero)

    if 'etnia' in df.columns:
        mapa_etnia = {'branca': 4,'parda': 1,'preta': 2,'amarela': 3}
        agrupar_valores_raros(df, 'etnia', mapa_etnia)

    if 'nivel_ensino' in df.columns:
        mapa_nivel_ensino = {
            'graduacao/bacharelado': 3,'pos-graduacao': 1,
            'estudantedegraduacao': 2,'mestrado': 1,
            'doutoradoouphd': 1,'naotenhograduacaoformal': 5
        }
        agrupar_valores_raros(df, 'nivel_ensino', mapa_nivel_ensino)

    if 'cloud_preferida' in df.columns:
        mapa_cloud_preferida = {
            'amazonwebservices(aws)': 4,'googlecloud(gcp)': 1,
            'naoseiopinar': 2,'azure(microsoft)': 3
        }
        agrupar_valores_raros(df, 'cloud_preferida', mapa_cloud_preferida)

    df[['sql','nosql']] = df['bancos_de_dados'].apply(contar_bancos)

    if 'vive_no_brasil' in df.columns:
        mapa_vive_no_brasil = {'True': 1, 'False': 2}
        agrupar_valores_raros(df, 'vive_no_brasil', mapa_vive_no_brasil)

    if 'linguagens_preferidas' in df.columns:
        mapa_linguagens_preferidas = {
            'python': 5,'r': 1,'sql': 2,'scala': 3,'c/c++/c#': 4
        }
        agrupar_valores_raros(df, 'linguagens_preferidas', mapa_linguagens_preferidas)

    if 'pcd' in df.columns:
        mapa_pcd = {'0': 1, '1': 2}
        agrupar_valores_raros(df, 'pcd', mapa_pcd)


    if 'tempo_experiencia_dados' in df.columns:
        mapa_tempo_experiencia_dados = {
            'de1a2anos': 7,'de3a4anos': 1,'menosde1ano': 2,
            'de4a6anos': 3,'maisde10anos': 4,'de7a10anos': 5,
            'naotenhoexperiencianaareadedados': 6
        }
        agrupar_valores_raros(df, 'tempo_experiencia_dados', mapa_tempo_experiencia_dados)



    if 'formacao' in df.columns:
        mapa_formacao = {
            'computacao/engenhariadesoftware/sistemasdeinformacao/ti': 9,
            'outrasengenharias': 1,
            'economia/administracao/contabilidade/financas/negocios': 2,
            'estatistica/matematica/matematicacomputacional/cienciasatuariais': 3,
            'outraopcao': 4,
            'marketing/publicidade/comunicacao/jornalismo': 5,
            'quimica/fisica': 6,
            'cienciasbiologicas/farmacia/medicina/areadasaude': 7,
            'cienciassociais': 8
        }
        agrupar_valores_raros(df, 'formacao', mapa_formacao)

    if 'cargo' in df.columns:
       
        mapa_cargo = {
            'cientistadedados/datascientist': 1,
            'engenheirodedados/arquitetodedados/dataengineer/dataarchitect': 2,
            'analistadebi/bianalyst': 3,
            'outraopcao': 4,
            'analistadenegocios/businessanalyst': 5,
            'desenvolvedor/engenheirodesoftware/analistadesistemas': 6,
            # 'analistadesuporte/analistatecnico': 7,
            # 'analyticsengineer': 8,
            # 'engenheirodemachinelearning/mlengineer': 9,
            # 'productmanager/productowner(pm/apm/dpm/gpm/po)': 10,
            'analistadedados/dataanalyst': 11,
        }
        agrupar_valores_raros(df, 'cargo', mapa_cargo)

# Executa o mapeamento
mapear(df)

# Remove coluna original de bancos
df = df.drop(columns=['bancos_de_dados'])

# Salva CSV final
df.to_csv('dataset_mapeado.csv', index=False)
print("✅ Dataset mapeado e salvo como 'dataset_mapeado.csv'.")
