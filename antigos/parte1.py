import pandas as pd
import numpy as np
import unidecode
import re

# --- Funções auxiliares ---
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
        return valor_norm

def normalizar_texto(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return "nao_informado"
    texto = unidecode.unidecode(str(valor)).strip().lower()
    return texto

def extrair_sigla_estado(valor):
    if isinstance(valor, str):
        match = re.search(r'\((\w{2})\)', valor)
        if match:
            return match.group(1).upper()
        # Caso venha só a sigla ou o nome do estado
        elif len(valor.strip()) == 2:
            return valor.strip().upper()
    return "MISSING"

def agrupar_valores_raros(valor, validos, label_outros="outros", label_nao_inf="nao_informado"):
    # Se valor na lista válida retorna ele, senão agrupa em 'outros' ou 'nao_informado'
    if valor in validos:
        return valor
    if valor in [label_outros, label_nao_inf]:
        return valor
    return label_outros

def limpar_lista(valor):
    # Divide por vírgula, limpa, normaliza e remove duplicados mantendo ordem
    if pd.isna(valor) or valor == "nao_informado":
        return ""
    itens = [normalizar_texto(i) for i in str(valor).split(",") if i.strip() != ""]
    # Remove duplicados mantendo ordem
    seen = set()
    lista = []
    for i in itens:
        if i not in seen:
            seen.add(i)
            lista.append(i)
    return ", ".join(lista)

def mapear_lista(valor, mapa, label_outros="outros"):
    # Para cada item da lista separada por vírgula, aplica o mapa
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
    # Extrai o primeiro item da lista, ou "nao_informado"
    if pd.isna(valor) or valor == "" or valor == "nao_informado":
        return "nao_informado"
    itens = [i.strip() for i in valor.split(",") if i.strip() != ""]
    return itens[0] if itens else "nao_informado"

# --- Mapas com os valores válidos (extraídos da sua frequência) ---

# Para cargos: mantemos os principais, juntamos o resto em "outros"
mapa_cargo = {
    'analista de dados/data analyst': 'analista de dados',
    'cientista de dados/data scientist': 'cientista de dados',
    'engenheiro de dados/arquiteto de dados/data engineer/data architect': 'engenheiro de dados',
    'analista de bi/bi analyst': 'analista de bi',
    'outra opção': 'outros',
    'analista de negócios/business analyst': 'analista de negocios',
    'desenvolvedor/ engenheiro de software/ analista de sistemas': 'desenvolvedor/engenheiro de software',
    'analista de suporte/analista técnico': 'analista de suporte',
    'analytics engineer': 'analytics engineer',
    'engenheiro de machine learning/ml engineer': 'engenheiro de ml',
    'product manager/ product owner (pm/apm/dpm/gpm/po)': 'product manager',
    'analista de inteligência de mercado/market intelligence': 'analista inteligencia mercado',
    'outras engenharias (não inclui dev)': 'outras engenharias',
    'professor': 'professor',
    'analista de marketing': 'analista marketing',
    'estatístico': 'estatistico',
    'economista': 'economista',
    'dba/administrador de banco de dados': 'dba',
    'nao_informado': 'nao_informado'
}

# Para formação: mesma regra
mapa_formacao = {
    'computação / engenharia de software / sistemas de informação/ ti': 'computacao',
    'outras engenharias': 'outras engenharias',
    'economia/ administração / contabilidade / finanças/ negócios': 'economia/negocios',
    'estatística/ matemática / matemática computacional/ ciências atuariais': 'estatistica/matematica',
    'outra opção': 'outros',
    'marketing / publicidade / comunicação / jornalismo': 'marketing/comunicacao',
    'química / física': 'quimica/fisica',
    'ciências biológicas/ farmácia/ medicina/ área da saúde': 'ciencias da saude',
    'ciências sociais': 'ciencias sociais',
    'nao_informado': 'nao_informado'
}

# Para tempo experiência dados: manter como está, agrupar valores que não existem ou com pouco peso
mapa_tempo_exp = {
    'menos de 1 ano': 'menos 1 ano',
    '1 a 2 anos': '1 a 2 anos',
    '1-2 anos': '1 a 2 anos',
    'de 1 a 2 anos': '1 a 2 anos',
    '2 a 3 anos': '2 a 3 anos',
    '3 a 4 anos': '3 a 4 anos',
    '3-5 anos': '3 a 4 anos',
    '4 a 6 anos': '4 a 6 anos',
    '5-10 anos': '5 a 10 anos',
    'mais de 10 anos': 'mais de 10 anos',
    'de 7 a 10 anos': '7 a 10 anos',
    'não tenho experiência na área de dados': 'nao tenho experiencia',
    'nao_informado': 'nao_informado'
}

# Para cloud preferida
mapa_cloud = {
    'amazon web services (aws)': 'aws',
    'amazon': 'aws',
    'aws': 'aws',
    'google cloud (gcp)': 'gcp',
    'google cloud': 'gcp',
    'gcp': 'gcp',
    'azure (microsoft)': 'azure',
    'azure': 'azure',
    'nao sei opinar': 'nao sei opinar',
    'outra cloud': 'outra',
    'nao_informado': 'nao_informado'
}

# Para genero
mapa_genero = {
    'masculino': 'masculino',
    'feminino': 'feminino',
    'prefiro não informar': 'nao_informado',
    'nao_informado': 'nao_informado'
}

# Para etnia
mapa_etnia = {
    'branca': 'branca',
    'parda': 'parda',
    'preta': 'preta',
    'amarela': 'amarela',
    'prefiro não informar': 'nao_informado',
    'indígena': 'indigena',
    'outra': 'outros',
    'nao_informado': 'nao_informado'
}

# Para PCD
mapa_pcd = {
    'nao': 'nao',
    'não': 'nao',
    'sim': 'sim',
    'prefiro não informar': 'nao_informado',
    'nao_informado': 'nao_informado'
}

mapa_vive_no_brasil = {
    'true': 'sim',
    'false': 'nao',
    'sim': 'sim',
    'não': 'nao',
    'nao': 'nao',
    'nao_informado': 'nao_informado'
}


# Mapas para linguagens, bancos de dados, etc. usando o que você já tem e ampliando com "outros"
mapa_linguagens = {
    'python': 'python',
    'r': 'r',
    'sql': 'sql',
    'scala': 'scala',
    'c/c++/c#': 'c/c++/c#',
    'julia': 'julia',
    'elixir': 'outros',
    'rust': 'outros',
    'excel': 'excel',
    'go': 'go',
    'java': 'java',
    'javascript': 'javascript',
    'dax': 'dax',
    'sas': 'sas',
    'm': 'm',
    'm language': 'm',
    'não sei': 'outros',
    'não uso': 'outros',
    'não utilizo': 'outros',
    'pyspark': 'outros',
    'spark': 'outros',
    'outros': 'outros',
    'nao_informado': 'nao_informado'
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
    'amazon athena': 'outros',
    'amazon redshift': 'redshift',
    'mongodb': 'mongodb',
    'hive': 'hive',
    'sqlite': 'sqlite',
    'sap hana': 'outros',
    'snowflake': 'snowflake',
    'amazon aurora ou rds': 'outros',
    'dynamodb': 'dynamodb',
    'microsoft access': 'outros',
    'presto': 'presto',
    'elaticsearch': 'elasticsearch',
    'elasticsearch': 'elasticsearch',
    'mariadb': 'mariadb',
    'db2': 'db2',
    'redis': 'outros',
    'firebase': 'firebase',
    'splunk': 'outros',
    'nenhum': 'nenhum',
    'cassandra': 'cassandra',
    'firebird': 'firebird',
    'hbase': 'outros',
    'google firestore': 'outros',
    'neo4j': 'outros',
    'excel': 'excel',
    'sybase': 'outros',
    'datomic': 'outros',
    'sas': 'sas',
    'teradata': 'outros',
    'não utilizo': 'nenhum',
    'azure': 'azure',
    'outros': 'outros',
    'nao_informado': 'nao_informado'
}

# --- Função principal que processa o csv ---

def processar_csv_completo(arquivo_entrada, arquivo_saida):
    print("Carregando CSV...")
    df = pd.read_csv(arquivo_entrada)

    df = df.dropna(subset=['cargo'])  # remove linhas onde cargo é NaN
    df = df[df['cargo'].astype(str).str.strip() != '']

    # Substituir valores vazios/espacos em branco por NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Remover duplicados
    df = df.drop_duplicates()

    # Idade
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    mediana_idade = int(df['idade'].median())
    df['idade'] = df['idade'].fillna(mediana_idade).astype(int)

    # Estado moradia: extrair sigla
    df['estado_moradia'] = df['estado_moradia'].apply(extrair_sigla_estado)
    # Mapear siglas para índice
    mapa_estados = {
        'AC': 0, 'AL': 1, 'AP': 2, 'AM': 3, 'BA': 4, 'CE': 5, 'DF': 6, 'ES': 7, 'GO': 8,
        'MA': 9, 'MT': 10, 'MS': 11, 'MG': 12, 'PA': 13, 'PB': 14, 'PR': 15, 'PE': 16,
        'PI': 17, 'RJ': 18, 'RN': 19, 'RS': 20, 'RO': 21, 'RR': 22, 'SC': 23, 'SP': 24,
        'SE': 25, 'TO': 26, 'MISSING': -1, 'nao_informado': -1
    }
    df['estado_moradia'] = df['estado_moradia'].str.upper().map(mapa_estados).fillna(-1).astype(int)

    # Normalizar strings simples
    col_str = ['genero', 'etnia', 'pcd', 'vive_no_brasil', 'nivel_ensino', 'formacao', 'cargo']
    for c in col_str:
        if c in df.columns:
            df[c] = df[c].apply(normalizar_texto)

    # Mapear colunas com agrupamento de valores raros
    df['genero'] = df['genero'].map(lambda x: agrupar_valores_raros(x, mapa_genero.keys(), "nao_informado"))
    df['genero'] = df['genero'].map(mapa_genero).fillna(mapa_genero['nao_informado'])

    df['etnia'] = df['etnia'].map(lambda x: agrupar_valores_raros(x, mapa_etnia.keys(), "nao_informado"))
    df['etnia'] = df['etnia'].map(mapa_etnia).fillna(mapa_etnia['nao_informado'])

    df['pcd'] = df['pcd'].map(lambda x: agrupar_valores_raros(x, mapa_pcd.keys(), "nao_informado"))
    df['pcd'] = df['pcd'].map(mapa_pcd).fillna(mapa_pcd['nao_informado'])

    df['nivel_ensino'] = df['nivel_ensino'].map(lambda x: agrupar_valores_raros(x, {
        'medio', 'tecnico', 'estudante de graduação', 'superior completo', 'pos-graduação', 
        'pós-graduação', 'mestrado', 'doutorado ou phd', 'graduação/bacharelado', 'nao_informado'
    }, 'nao_informado'))
    # mapear níveis para números
    mapa_nivel = {
        'medio': 1,
        'tecnico': 2,
        'estudante de graduação': 3,
        'superior completo': 4,
        'graduação/bacharelado': 4,
        'pos-graduação': 5,
        'pós-graduação': 5,
        'mestrado': 6,
        'doutorado ou phd': 7,
        'nao_informado': -1
    }
    df['nivel_ensino'] = df['nivel_ensino'].map(mapa_nivel).fillna(-1).astype(int)

    df['formacao'] = df['formacao'].map(lambda x: agrupar_valores_raros(x, mapa_formacao.keys(), "nao_informado"))
    df['formacao'] = df['formacao'].map(mapa_formacao).fillna('outros')

    # Tempo experiência dados
    if 'tempo_experiencia_dados' in df.columns:
        df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].apply(normalizar_texto)
        df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].map(lambda x: agrupar_valores_raros(x, mapa_tempo_exp.keys(), "nao_informado"))
        df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].map(mapa_tempo_exp).fillna('nao_informado')

    # Cargo
    df['cargo'] = df['cargo'].map(lambda x: agrupar_valores_raros(x, mapa_cargo.keys(), "nao_informado"))
    df['cargo'] = df['cargo'].map(mapa_cargo).fillna('outros')

    # --- Processar listas ---
    # Limpar e normalizar listas
    for coluna in ['linguagens_preferidas', 'bancos_de_dados', 'cloud_preferida']:
        if coluna in df.columns:
            df[coluna] = df[coluna].apply(limpar_lista)

    # Mapear valores nas listas
    if 'linguagens_preferidas' in df.columns:
        df['linguagens_preferidas'] = df['linguagens_preferidas'].apply(lambda x: mapear_lista(x, mapa_linguagens))

    if 'bancos_de_dados' in df.columns:
        df['bancos_de_dados'] = df['bancos_de_dados'].apply(lambda x: mapear_lista(x, mapa_bancos))

    if 'cloud_preferida' in df.columns:
        df['cloud_preferida'] = df['cloud_preferida'].apply(lambda x: mapear_lista(x, mapa_cloud))

    # Extrair o principal da linguagem preferida para mapeamento numérico (exemplo)
    df['linguagem_principal'] = df['linguagens_preferidas'].apply(extrair_primeiro_item)
    mapa_linguagens_num = {
        'python': 0,
        'r': 1,
        'sql': 2,
        'javascript': 3,
        'java': 4,
        'c/c++/c#': 5,
        'scala': 6,
        'go': 7,
        'julia': 8,
        'dax': 9,
        'sas': 10,
        'excel': 11,
        'm': 12,
        'nao_informado': -1,
        'outros': 13
    }
    df['linguagem_principal'] = df['linguagem_principal'].map(mapa_linguagens_num).fillna(-1).astype(int)

    # Para cloud preferida principal (exemplo numérico)
    df['cloud_principal'] = df['cloud_preferida'].apply(extrair_primeiro_item)
    mapa_cloud_num = {
        'aws': 0,
        'gcp': 1,
        'azure': 2,
        'nao sei opinar': 3,
        'outra': 4,
        'nao_informado': -1
    }
    df['cloud_principal'] = df['cloud_principal'].map(mapa_cloud_num).fillna(-1).astype(int)

    # Para colunas booleanas e sim/não como vive_no_brasil e pcd:
    def normalizar_bool_sim_nao(valor):
        if pd.isna(valor) or valor == '' or valor == 'nao_informado':
            return 'nao_informado'
        valor = str(valor).strip().lower()
        if valor in ['true', 'sim', 'yes', '1']:
            return 'sim'
        elif valor in ['false', 'não', 'nao', 'no', '0']:
            return 'nao'
        else:
            return 'nao_informado'

    df['pcd'] = df['pcd'].apply(normalizar_bool_sim_nao)
    df['vive_no_brasil'] = df['vive_no_brasil'].apply(normalizar_bool_sim_nao)


    # mapa_pcd = {'sim':1, 'nao':0, 'nao_informado':-1}
    mapa_vive_no_brasil = {'sim':1, 'nao':0, 'nao_informado':-1}

    df['pcd'] = df['pcd'].map(mapa_pcd).fillna(-1)
    df['vive_no_brasil'] = df['vive_no_brasil'].map(mapa_vive_no_brasil).fillna(-1)

    # Salvar csv tratado
    df.to_csv(arquivo_saida, index=False)
    print(f"Arquivo tratado salvo em: {arquivo_saida}")

    # Mostrar resumo para conferir
    print("\n--- Resumo após processamento ---")
    print(df.describe(include='all').transpose())

    return df

# Exemplo de uso:
df_tratado = processar_csv_completo("dataset.csv", "dados_tratados.csv")
