import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset_mapeado.csv')

colunas = [
    'idade','etnia','estado_moradia', 'genero',
    'nivel_ensino','formacao','tempo_experiencia_dados',
    'cloud_preferida', 'sql', 'nosql', 
    'linguagens_preferidas'
]

X = df[colunas].copy()
y = df['cargo']

# --- Criar colunas combinadas ---
X['formacao_x_tempo'] = X['formacao'].astype(str) + "_" + X['tempo_experiencia_dados'].astype(str)
X['idade_x_tempo'] = X['idade'].astype(str) + "_" + X['tempo_experiencia_dados'].astype(str)
X['idade_x_escolaridade'] = X['idade'].astype(str) + "_" + X['nivel_ensino'].astype(str)
X['tempo_x_linguagem'] = X['tempo_experiencia_dados'].astype(str) + "_" + X['linguagens_preferidas'].astype(str)

# --- One-hot encoding ---
df_encoded = pd.get_dummies(X, drop_first=True)
X = df_encoded

# --- Imputação ---
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)
X = pd.DataFrame(X_filled, columns=X.columns)

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- SMOTE ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- RandomForest ---
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_res, y_train_res)

y_pred = rf.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print("Balanced Accuracy:", balanced_acc)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

importancias = pd.Series(rf.feature_importances_, index=X.columns)
print("\n📊 Importância das variáveis (RandomForest):")
print(importancias.sort_values(ascending=False).head(15))


# import pandas as pd
# import numpy as np
# from sklearn.metrics import classification_report, balanced_accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import seaborn as sns


# df = pd.read_csv('dataset_knn.csv')

# colunas = [
#     'idade','etnia','estado_moradia', 'genero',
#     'nivel_ensino','formacao','tempo_experiencia_dados',
#     'cloud_preferida', 'sql', 'nosql', 
#     'linguagens_preferidas'
# ]

# df_encoded = pd.get_dummies(df[colunas], drop_first=True)
# X = df_encoded
# y = df['cargo']


# imputer = SimpleImputer(strategy='mean')
# X_filled = imputer.fit_transform(X)
# X = pd.DataFrame(X_filled, columns=X.columns)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=42, stratify=y
# )


# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# rf = RandomForestClassifier(
#     n_estimators=200,
#     class_weight='balanced',
#     random_state=42
# )
# rf.fit(X_train_res, y_train_res)

# y_pred = rf.predict(X_test)

# balanced_acc = balanced_accuracy_score(y_test, y_pred)
# print("Balanced Accuracy:", balanced_acc)
# print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# importancias = pd.Series(rf.feature_importances_, index=X.columns)
# print("\n📊 Importância das variáveis (RandomForest):")
# print(importancias.sort_values(ascending=False).head(15))
