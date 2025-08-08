import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer

print("🔍 Mapeamento + KNN Imputer")
print("=" * 60)

# =============================
# PARTE 1: PROCESSAMENTO PRINCIPAL
# =============================

print("\n📋 ETAPA 1: CARREGAMENTO E PROCESSAMENTO DOS DADOS")
print("-" * 50)

# Carregamento de dados (o CSV já tem cabeçalho)
df = pd.read_csv('dataset.csv')

# Tratamento inicial de valores ausentes
print("📋 Tratando valores ausentes iniciais...")
df = df.replace(r'^\s*$', pd.NA, regex=True)
print(f"Registros com campo genero vazio: {df['genero'].isna().sum()}")

print("Primeiras Linhas:")
print(df.head())

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

print("\nRegistros duplicados:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape após remoção de duplicados:", df.shape)

df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
df['idade'] = df['idade'].fillna(df['idade'].median())
df['idade'] = df['idade'].astype(int)

def extrair_sigla_estado(valor):
    if isinstance(valor, str):
        match = re.search(r'\((\w{2})\)', valor)
        if match:
            return match.group(1)
    return 'MISSING'

df['estado_moradia'] = df['estado_moradia'].apply(extrair_sigla_estado)

df = df.fillna('MISSING')

print("\n📋 APLICANDO MAPEAMENTOS MANUAIS...")

mapa_genero = {
    'Masculino': 0,
    'Feminino': 1,
    'MISSING': -1
}
df['genero'] = df['genero'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['genero'] = df['genero'].map(mapa_genero).fillna(-1).astype(int)

mapa_etnia = {
    'Branca': 0,
    'Parda': 1,
    'Preta': 2,
    'Amarela': 3,
    'Indígena': 4,
    'Prefiro não informar': 5,
    'MISSING': -1
}
df['etnia'] = df['etnia'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['etnia'] = df['etnia'].map(mapa_etnia).fillna(-1).astype(int)

mapa_pcd = {
    'Não': 0, 
    'False': 0,
    'Sim': 1,
    'True': 1,
    'MISSING': -1
}
df['pcd'] = df['pcd'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['pcd'] = df['pcd'].map(mapa_pcd).fillna(-1).astype(int)

mapa_cloud = {
    'Amazon Web Services (AWS)': 0,
    'Google Cloud (GCP)': 1,
    'Azure (Microsoft)': 2,
    'Não sei opinar': 3,
    'Não utilizo': 4,
    'Oracle Cloud': 5,
    'IBM Cloud': 6,
    'Outra opção': 7,
    'MISSING': -1
}
df['cloud_preferida'] = df['cloud_preferida'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['cloud_preferida'] = df['cloud_preferida'].map(mapa_cloud).fillna(-1).astype(int)

mapa_cargo = {
    'Cientista de Dados/Data Scientist': 0,
    'Analista de Dados/Data Analyst': 1,
    'Engenheiro de Dados/Data Engineer': 2,
    'Desenvolvedor/ Engenheiro de Software/ Analista de Sistemas': 3,
    'Analista de BI/BI Analyst': 4,
    'DBA/Administrador de Banco de Dados': 5,
    'Professor': 6,
    'Analista de Negócios/Business Analyst': 7,
    'Analista de Suporte/Analista Técnico': 8,
    'Gerente/Coordenador de TI': 9,
    'Outra Opção': 10,
    'MISSING': -1
}
df['cargo'] = df['cargo'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['cargo'] = df['cargo'].map(mapa_cargo).fillna(-1).astype(int)

mapa_formacao = {
    'Computação / Engenharia de Software / Sistemas de Informação/ TI': 0,
    'Estatística/ Matemática / Matemática Computacional/ Ciências Atuariais': 1,
    'Outras Engenharias': 2,
    'Economia/ Administração / Contabilidade / Finanças/ Negócios': 3,
    'Ciências Biológicas/ Farmácia/ Medicina/ Área da Saúde': 4,
    'Ciências Sociais': 5,
    'Química / Física': 6,
    'Marketing / Publicidade / Comunicação / Jornalismo': 7,
    'Outra opção': 8,
    'MISSING': -1
}
df['formacao'] = df['formacao'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['formacao'] = df['formacao'].map(mapa_formacao).fillna(-1).astype(int)

mapa_estados = {
    'AC': 0, 'AL': 1, 'AP': 2, 'AM': 3, 'BA': 4, 'CE': 5, 'DF': 6, 'ES': 7, 'GO': 8,
    'MA': 9, 'MT': 10, 'MS': 11, 'MG': 12, 'PA': 13, 'PB': 14, 'PR': 15, 'PE': 16,
    'PI': 17, 'RJ': 18, 'RN': 19, 'RS': 20, 'RO': 21, 'RR': 22, 'SC': 23, 'SP': 24,
    'SE': 25, 'TO': 26, 'MISSING': -1
}
df['estado_moradia'] = df['estado_moradia'].apply(lambda x: mapa_estados.get(x, -1))

mapa_nivel = {
    'fundamental': 0,
    'medio': 1,
    'tecnico': 2,
    'superior incompleto': 3,
    'superior completo': 4,
    'pos-graduação': 5,
    'pós-graduação': 5,
    'mestrado': 6,
    'doutorado': 7,
    'MISSING': -1,
    'graduação/bacharelado': 4
}
df['nivel_ensino'] = df['nivel_ensino'].str.lower().map(mapa_nivel).fillna(-1).astype(int)

mapa_experiencia = {
    'menos de 1 ano': 0,
    '1 a 2 anos': 1,
    '1-2 anos': 1,
    'de 1 a 2 anos': 1,
    '2 a 3 anos': 2,
    '3 a 4 anos': 3,
    '3-5 anos': 4,
    '5-10 anos': 5,
    'mais de 10 anos': 6,
    'MISSING': -1
}
df['tempo_experiencia_dados'] = df['tempo_experiencia_dados'].str.lower().map(mapa_experiencia).fillna(-1).astype(int)

def extrair_principal(x):
    if isinstance(x, str) and x != 'MISSING' and x.strip() != '':
        items = [i.strip() for i in x.split(',')]
        return items[0] if items else 'MISSING'
    return 'MISSING'

df['bancos_de_dados'] = df['bancos_de_dados'].apply(extrair_principal)
mapa_bancos = {
    'PostgreSQL': 0,
    'MySQL': 1,
    'SQL SERVER': 2,
    'Oracle': 3,
    'Google BigQuery': 4,
    'Microsoft Access': 5,
    'SQLite': 6,
    'Amazon Athena': 7,
    'MongoDB': 8,
    'Hive': 9,
    'Amazon Redshift': 10,
    'Databricks': 11,
    'Snowflake': 12,
    'DB2': 13,
    'Cassandra': 14,
    'Não utilizo': 15,
    'MISSING': -1
}
df['bancos_de_dados'] = df['bancos_de_dados'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['bancos_de_dados'] = df['bancos_de_dados'].map(mapa_bancos).fillna(-1).astype(int)

df['linguagens_preferidas'] = df['linguagens_preferidas'].apply(extrair_principal)
mapa_linguagens = {
    'Python': 0,
    'R': 1,
    'SQL': 2,
    'javascript': 3,
    'Java': 4,
    'C/C++/C#': 5,
    'Scala': 6,
    'Go': 7,
    'Julia': 8,
    'SAS': 9,
    'Excel': 10,
    'Dax': 11,
    'M': 12,
    'Não uso': 13,
    'Não utilizo': 14,
    'MISSING': -1
}
df['linguagens_preferidas'] = df['linguagens_preferidas'].fillna('MISSING').astype(str).replace('', 'MISSING')
df['linguagens_preferidas'] = df['linguagens_preferidas'].map(mapa_linguagens).fillna(-1).astype(int)

df = df.reset_index(drop=True)

print("\nColunas do tipo 'object' antes da conversão para inteiro:")
print(df.dtypes[df.dtypes == 'object'])

df = df.fillna(-1)
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
df[colunas_numericas] = df[colunas_numericas].astype(int)

df = df.drop_duplicates()

print(f"\n📈 Total de colunas finais: {df.shape[1]} (objetivo: 13)")
print(f"Colunas: {list(df.columns)}")

print(f"\n✅ Verificação final:")
for col in df.columns:
    tipo = df[col].dtype
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"{col}: {tipo} (min: {min_val}, max: {max_val})")

df.to_csv('dados_processados.csv', index=False)
print("✅ Arquivo 'dados_processados.csv' salvo com sucesso.")

print("\n📋 ETAPA 2: ANÁLISE E IMPUTAÇÃO KNN")
print("-" * 50)

missing_count = (df == -1).sum()
missing_percent = (missing_count / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    'Coluna': missing_count.index,
    'Valores_Ausentes': missing_count.values,
    'Percentual': missing_percent.values
})

print(missing_summary)

colunas_com_missing = missing_count[missing_count > 0].index.tolist()
print(f"\nColunas com valores ausentes: {colunas_com_missing}")
print(f"Total de valores ausentes: {missing_count.sum()}")

if missing_count.sum() == 0:
    print("🎉 Não há valores ausentes para imputar!")
    print("✅ Processo finalizado - dados já estão completos!")
else:

    mapeamentos_reversos = {
        'genero': {0: 'Masculino', 1: 'Feminino', -1: 'MISSING'},
        'etnia': {0: 'Branca', 1: 'Parda', 2: 'Preta', 3: 'Amarela', 4: 'Indígena', 5: 'Prefiro não informar', -1: 'MISSING'},
        'pcd': {0: 'Não', 1: 'Sim', -1: 'MISSING'},
        'formacao': {0: 'Computação/TI', 1: 'Estatística/Matemática', 2: 'Outras Engenharias',
                    3: 'Economia/Administração', 4: 'Ciências Biológicas', 5: 'Ciências Sociais',
                    6: 'Química/Física', 7: 'Marketing/Comunicação', 8: 'Outra opção', -1: 'MISSING'},
        'cargo': {0: 'Cientista de Dados', 1: 'Analista de Dados', 2: 'Engenheiro de Dados',
                 3: 'Desenvolvedor/Engenheiro Software', 4: 'Analista de BI', 5: 'DBA',
                 6: 'Professor', 7: 'Analista de Negócios', 8: 'Analista de Suporte',
                 9: 'Gerente/Coordenador TI', 10: 'Outra Opção', -1: 'MISSING'}
    }

    # Remover 'vive_no_brasil' da lista de colunas para imputar, se presente
    if 'vive_no_brasil' in colunas_com_missing:
        colunas_com_missing.remove('vive_no_brasil')

    # Também remover do dataframe para que não interfira no knn
    df_knn = df.drop(columns=['vive_no_brasil'], errors='ignore').copy()

    def preparar_dados_knn(df, colunas_imputar):
        df_knn = df.copy()
        for col in colunas_imputar:
            df_knn.loc[df_knn[col] == -1, col] = np.nan

        print(f"📋 Preparação para KNN:")
        print(f"Colunas para imputação: {colunas_imputar}")
        for col in colunas_imputar:
            nan_count = df_knn[col].isna().sum()
            print(f"  {col}: {nan_count} valores NaN")

        return df_knn

    df_knn = preparar_dados_knn(df_knn, colunas_com_missing)

    def aplicar_knn_inteligente(df, colunas_imputar, k_neighbors=5):
        print(f"\n🔄 Aplicando KNN Imputer (k={k_neighbors})")
        print("-" * 40)
        df_imputed = df.copy()
        imputer = KNNImputer(
            n_neighbors=k_neighbors,
            weights='distance',
            metric='nan_euclidean'
        )
        if colunas_imputar:
            print("Aplicando imputação...")
            dados_imputados = imputer.fit_transform(df[colunas_imputar])
            dados_imputados = np.round(dados_imputados).astype(int)
            for i, col in enumerate(colunas_imputar):
                df_imputed[col] = dados_imputados[:, i]
            print("✅ Imputação concluída!")
            missing_after = (df_imputed == -1).sum().sum()
            nan_after = df_imputed.isna().sum().sum()
            print(f"Valores -1 restantes: {missing_after}")
            print(f"Valores NaN restantes: {nan_after}")
        return df_imputed, imputer

    resultados_k = {}
    k_values = [3, 5, 7, 10]

    print("\n🎯 Testando Diferentes Valores de K:")
    print("=" * 40)

    for k in k_values:
        print(f"\n--- Testando k={k} ---")
        df_result, imputer = aplicar_knn_inteligente(df_knn, colunas_com_missing, k)
        resultados_k[k] = df_result
        for col in colunas_com_missing:
            original_missing = (df_knn[col] == -1).sum()
            valores_imputados = df_result[col][df_knn[col] == -1]
            if len(valores_imputados) > 0:
                print(f"  {col}: {original_missing} valores imputados")
                print(f"    Distribuição: {valores_imputados.value_counts().to_dict()}")

    def validar_qualidade_imputacao(df_original, df_imputado, coluna):
        if coluna not in df_original.columns or coluna not in df_imputado.columns:
            return
        print(f"\n📊 Validação para {coluna}:")
        print("-" * 30)
        valores_originais = df_original[df_original[coluna] != -1][coluna]
        mask_era_missing = df_original[coluna] == -1
        valores_imputados = df_imputado[mask_era_missing][coluna]
        if len(valores_imputados) > 0:
            print(f"Distribuição original:")
            dist_orig = valores_originais.value_counts().sort_index()
            for val, count in dist_orig.items():
                nome_val = mapeamentos_reversos.get(coluna, {}).get(val, val)
                pct = (count / len(valores_originais) * 100)
                print(f"  {nome_val}: {count} ({pct:.1f}%)")
            print(f"\nDistribuição imputada:")
            dist_imp = valores_imputados.value_counts().sort_index()
            for val, count in dist_imp.items():
                nome_val = mapeamentos_reversos.get(coluna, {}).get(val, val)
                pct = (count / len(valores_imputados) * 100)
                print(f"  {nome_val}: {count} ({pct:.1f}%)")

    if 5 in resultados_k:
        print("\n🔍 VALIDAÇÃO DE QUALIDADE (k=5):")
        print("=" * 50)
        for col in colunas_com_missing[:3]:
            validar_qualidade_imputacao(df, resultados_k[5], col)

    # Salvar resultados
    melhor_k = 5
    df_final = resultados_k[melhor_k]

    print(f"\n💾 Salvando Resultado Final (k={melhor_k}):")
    print("-" * 40)

    # Verificação final
    print(f"Shape final: {df_final.shape}")
    print(f"Valores -1 restantes: {(df_final == -1).sum().sum()}")
    print(f"Valores NaN restantes: {df_final.isna().sum().sum()}")

    # Salvar dataset com valores imputados
    df_final.to_csv('dados_knn_imputed.csv', index=False)
    print("✅ Dataset salvo como 'dados_knn_imputed.csv'")

    # Relatório final
    print("\n📋 RELATÓRIO FINAL - KNN IMPUTATION")
    print("=" * 60)
    print(f"Dataset original: {df.shape}")
    print(f"Valores ausentes originais: {(df == -1).sum().sum()}")
    print(f"Colunas com missing: {len(colunas_com_missing)}")
    print(f"Método: KNN Imputer (k={melhor_k})")
    print(f"Dataset final: {df_final.shape}")
    print(f"Valores ausentes finais: {(df_final == -1).sum().sum()}")

    if (df == -1).sum().sum() > 0:
        taxa_sucesso = 100 - ((df_final == -1).sum().sum() / (df == -1).sum().sum() * 100)
        print(f"Taxa de sucesso: {taxa_sucesso:.1f}%")


print("\n✅ Pipeline Unificado Concluído com Sucesso!")