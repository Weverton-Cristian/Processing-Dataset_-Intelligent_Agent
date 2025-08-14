import pandas as pd
import numpy as np
import unidecode
import re

# --- Funções auxiliares ---

def normalizar_texto(valor):
    """Transforma texto para minúsculo, sem acento e tira espaços extras"""
    if pd.isna(valor) or str(valor).strip() == "":
        return ""
    texto = unidecode.unidecode(str(valor)).strip().lower()
    return texto

def extrair_sigla_estado(valor):
    """Extrai sigla do estado de texto tipo 'Distrito Federal (DF)' ou só 'DF'"""
    if isinstance(valor, str):
        match = re.search(r'\((\w{2})\)', valor)
        if match:
            return match.group(1).upper()
        elif len(valor.strip()) == 2:
            return valor.strip().upper()
    return "MISSING"

def agrupar_valores_raros(valor, validos, label_outros="outros"):
    """
    Mantém o valor se estiver na lista de válidos, senão agrupa em 'outros'.
    Não joga valores vazios para 'nao_informado', apenas para 'outros'.
    """
    if valor in validos:
        return valor
    if valor == "" or valor is None:
        return ""
    return label_outros

def limpar_lista(valor):
    """Limpa listas separadas por vírgula: normaliza e remove duplicados mantendo ordem"""
    if pd.isna(valor) or valor == "":
        return ""
    itens = [normalizar_texto(i) for i in str(valor).split(",") if i.strip() != ""]
    seen = set()
    lista = []
    for i in itens:
        if i not in seen:
            seen.add(i)
            lista.append(i)
    return ", ".join(lista)

def mapear_lista(valor, mapa, label_outros="outros"):
    """Mapeia cada item da lista separada por vírgula no mapa, com fallback 'outros'"""
    if not valor:
        return ""
    itens = [i.strip() for i in valor.split(",")]
    itens_mapeados = []
    for item in itens:
        m = mapa.get(item, label_outros)
        if m not in itens_mapeados:
            itens_mapeados.append(m)
    return ", ".join(itens_mapeados)

def extrair_primeiro_item(valor):
    """Pega o primeiro item da lista ou vazio"""
    if pd.isna(valor) or valor == "":
        return ""
    itens = [i.strip() for i in valor.split(",") if i.strip() != ""]
    return itens[0] if itens else ""

def normalizar_bool_sim_nao(valor):
    """
    Normaliza texto sim/não para 'sim' ou 'nao'.
    Aceita várias formas comuns, não joga valor válido para 'nao_informado'.
    """
    if pd.isna(valor) or str(valor).strip() == "":
        return ""
    valor_norm = unidecode.unidecode(str(valor).strip().lower())
    if valor_norm in ['true', 'sim', 'yes', '1', 'Sim']:
        return 'sim'
    elif valor_norm in ['false', 'nao', 'não', 'no', '0', 'Não']:
        return 'nao'
    else:
        return valor_norm  # Retorna o valor normalizado caso não esteja no padrão esperado

# --- Mapas (exemplos com variantes comuns para manter o máximo de valores) ---

mapa_genero = {
    'masculino': 'masculino',
    'feminino': 'feminino',
    'outros': 'outros',
    '': '',  # mantém vazio se não informado
}

mapa_etnia = {
    'branca': 'branca',
    'parda': 'parda',
    'preta': 'preta',
    'amarela': 'amarela',
    'indigena': 'indigena',
    'outros': 'outros',
    '': '',
}

mapa_pcd = {
    'sim': 'sim',
    'nao': 'nao',
    '': '',
}

mapa_vive_no_brasil = {
    'sim': 'sim',
    'nao': 'nao',
    '': '',
}

mapa_nivel = {
    'medio': 1,
    'tecnico': 2,
    'estudante de graduacao': 3,
    'superior completo': 4,
    'graduacao/bacharelado': 4,
    'pos-graduacao': 5,
    'mestrado': 6,
    'doutorado ou phd': 7,
    '': -1
}

mapa_formacao = {
    'computacao / engenharia de software / sistemas de informacao/ ti': 'computacao',
    'outras engenharias': 'outras engenharias',
    'economia/ administracao / contabilidade / financas/ negocios': 'economia/negocios',
    'estatistica/ matematica / matematica computacional/ ciencias atuariais': 'estatistica/matematica',
    'marketing / publicidade / comunicacao / jornalismo': 'marketing/comunicacao',
    'quimica / fisica': 'quimica/fisica',
    'ciencias biologicas/ farmacia/ medicina/ area da saude': 'ciencias da saude',
    'ciencias sociais': 'ciencias sociais',
    '': '',
}

mapa_tempo_exp = {
    'menos de 1 ano': 'menos 1 ano',
    '1 a 2 anos': '1 a 2 anos',
    '2 a 3 anos': '2 a 3 anos',
    '3 a 4 anos': '3 a 4 anos',
    '4 a 6 anos': '4 a 6 anos',
    '5 a 10 anos': '5 a 10 anos',
    'mais de 10 anos': 'mais de 10 anos',
    'nao tenho experiencia na area de dados': 'nao tenho experiencia',
    '': '',
}

mapa_cargo = {
    'analista de dados/data analyst': 'analista de dados',
    'cientista de dados/data scientist': 'cientista de dados',
    'engenheiro de dados/arquiteto de dados/data engineer/data architect': 'engenheiro de dados',
    'analista de bi/bi analyst': 'analista de bi',
    'analista de negocios/business analyst': 'analista de negocios',
    'desenvolvedor/ engenheiro de software/ analista de sistemas': 'desenvolvedor/engenheiro de software',
    'analista de suporte/analista tecnico': 'analista de suporte',
    'analytics engineer': 'analytics engineer',
    'engenheiro de machine learning/ml engineer': 'engenheiro de ml',
    'product manager/ product owner (pm/apm/dpm/gpm/po)': 'product manager',
    'analista de inteligencia de mercado/market intelligence': 'analista inteligencia mercado',
    'outras engenharias (nao inclui dev)': 'outras engenharias',
    'professor': 'professor',
    'analista de marketing': 'analista marketing',
    'estatistico': 'estatistico',
    'economista': 'economista',
    'dba/administrador de banco de dados': 'dba',
    '': '',
}

mapa_linguagens = {
    'python': 'python',
    'r': 'r',
    'sql': 'sql',
    'scala': 'scala',
    'c/c++/c#': 'c/c++/c#',
    'julia': 'julia',
    'excel': 'excel',
    'go': 'go',
    'java': 'java',
    'javascript': 'javascript',
    'dax': 'dax',
    'sas': 'sas',
    'm': 'm',
    'outros': 'outros',
    '': '',
}

mapa_bancos = {
    'sql server': 'sqlserver',
    'mysql': 'mysql',
    'postgresql': 'postgresql',
    'google bigquery': 'bigquery',
    'bigquery': 'bigquery',
    's3': 's3',
    'databricks': 'databricks',
    'oracle': 'oracle',
    'redshift': 'redshift',
    'mongodb': 'mongodb',
    'hive': 'hive',
    'sqlite': 'sqlite',
    'snowflake': 'snowflake',
    'dynamodb': 'dynamodb',
    'presto': 'presto',
    'elasticsearch': 'elasticsearch',
    'mariadb': 'mariadb',
    'db2': 'db2',
    'firebase': 'firebase',
    'cassandra': 'cassandra',
    'firebird': 'firebird',
    'azure': 'azure',
    'outros': 'outros',
    '': '',
}

mapa_cloud = {
    'aws': 'aws',
    'gcp': 'gcp',
    'azure': 'azure',
    'nao sei opinar': 'nao sei opinar',
    'outra': 'outra',
    '': '',
}

# --- Função principal ---

def processar_csv_completo(arquivo_entrada, arquivo_saida):
    print("Carregando CSV...")
    df = pd.read_csv(arquivo_entrada)

    # Remover duplicados
    df = df.drop_duplicates()

    # Normalizar texto das colunas categóricas
    col_str = ['genero', 'etnia', 'pcd', 'vive_no_brasil', 'nivel_ensino', 'formacao', 'cargo', 'tempo_experiencia_dados']
    for c in col_str:
        if c in df.columns:
            df[c] = df[c].apply(normalizar_texto)

    # Normalizar booleanos sim/nao
    if 'pcd' in df.columns:
        df['pcd'] = df['pcd'].apply(normalizar_bool_sim_nao)
    if 'vive_no_brasil' in df.columns:
        df['vive_no_brasil'] = df['vive_no_brasil'].apply(normalizar_bool_sim_nao)

    # Idade
    if 'idade' in df.columns:
        df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
        mediana_idade = int(df['idade'].median())
        df['idade'] = df['idade'].fillna(mediana_idade).astype(int)

    # Extrair sigla do estado e mapear
    if 'estado_moradia' in df.columns:
        df['estado_moradia'] = df['estado_moradia'].apply(extrair_sigla_estado)
        mapa_estados = {
            'AC': 0, 'AL': 1, 'AP': 2, 'AM': 3, 'BA': 4, 'CE': 5, 'DF': 6, 'ES': 7, 'GO': 8,
            'MA': 9, 'MT': 10, 'MS': 11, 'MG': 12, 'PA': 13, 'PB': 14, 'PR': 15, 'PE': 16,
            'PI': 17, 'RJ': 18, 'RN': 19, 'RS': 20, 'RO': 21, 'RR': 22, 'SC': 23, 'SP': 24,
            'SE': 25, 'TO': 26, 'MISSING': -1, '': -1
        }
        df['estado_moradia'] = df['estado_moradia'].map(mapa_estados).fillna(-1).astype(int)

    # Agrupar valores raros mantendo o máximo de valores possíveis
    df['genero'] = df['genero'].apply(lambda x: agrupar_valores_raros(x, mapa_genero.keys(), "outros"))
    df['genero'] = df['genero'].map(mapa_genero).fillna("outros")

    df['etnia'] = df['etnia'].apply(lambda x: agrupar_valores_raros(x, mapa_etnia.keys(), "outros"))
    df['etnia'] = df['etnia'].map(mapa_etnia).fillna("outros")

    df['pcd'] = df['pcd'].apply(lambda x: agrupar_valores_raros(x, mapa_pcd.keys(), "outros"))
    df['pcd'] = df['pcd'].map(mapa_pcd).fillna("outros")

    df['vive_no_brasil'] = df['vive_no_brasil'].apply(lambda x: agrupar_valores_raros(x, mapa_vive_no_brasil.keys(), "outros"))
    df['vive_no_brasil'] = df['vive_no_brasil'].map(mapa_vive_no_brasil).fillna("outros")

    df['nivel_ensino'] = df['nivel_ensino'].apply(lambda x: agrupar_valores_raros(x, mapa_nivel.keys(), ""))
    df['nivel_ensino'] = df['nivel_ensino'].map(mapa_nivel).fillna(-1).astype(int)

    df['formacao'] = df['formacao'].apply(lambda x: agrupar_valores_raros(x, mapa_formacao.keys(), "outros"))
    df['formacao'] = df['formacao'].map(mapa_formacao).fillna("outros")

    df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].apply(lambda x: agrupar_valores_raros(x, mapa_tempo_exp.keys(), ""))
    df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].map(mapa_tempo_exp).fillna("")

    df['cargo'] = df['cargo'].apply(lambda x: agrupar_valores_raros(x, mapa_cargo.keys(), "outros"))
    df['cargo'] = df['cargo'].map(mapa_cargo).fillna("outros")

    # Listas: limpar e mapear
    for coluna, mapa in [('linguagens_preferidas', mapa_linguagens), ('bancos_de_dados', mapa_bancos), ('cloud_preferida', mapa_cloud)]:
        if coluna in df.columns:
            df[coluna] = df[coluna].apply(limpar_lista)
            df[coluna] = df[coluna].apply(lambda x: mapear_lista(x, mapa))

    # Extrair principal linguagem e cloud (exemplo numérico)
    mapa_linguagens_num = {k: i for i, k in enumerate(sorted(set(mapa_linguagens.values()) - {''}))}
    mapa_linguagens_num[''] = -1
    df['linguagem_principal'] = df['linguagens_preferidas'].apply(extrair_primeiro_item)
    df['linguagem_principal'] = df['linguagem_principal'].map(mapa_linguagens_num).fillna(-1).astype(int)

    mapa_cloud_num = {k: i for i, k in enumerate(sorted(set(mapa_cloud.values()) - {''}))}
    mapa_cloud_num[''] = -1
    df['cloud_principal'] = df['cloud_preferida'].apply(extrair_primeiro_item)
    df['cloud_principal'] = df['cloud_principal'].map(mapa_cloud_num).fillna(-1).astype(int)

    # Salvar resultado
    df.to_csv(arquivo_saida, index=False)
    print(f"Arquivo tratado salvo em: {arquivo_saida}")
    print("\nResumo pós-processamento:")
    print(df.describe(include='all').transpose())

    return df

# Exemplo de uso:
df_tratado = processar_csv_completo("dataset.csv", "tratados.csv")
