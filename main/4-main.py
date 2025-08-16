import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv('dataset_knn.csv')

class FeatureInteractions(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Interações desejadas
        X['formacao_x_tempo'] = X['formacao'].astype(str) + "_" + X['tempo_experiencia_dados'].astype(str)
        X['idade_x_tempo'] = X['idade'] * X['tempo_experiencia_dados']
        X['tempo_x_linguagem'] = X['tempo_experiencia_dados'].astype(str) + "_" + X['linguagens_preferidas'].astype(str)
        X['idade_x_escolaridade'] = X['idade'].astype(str) + "_" + X['nivel_ensino'].astype(str)
        X['ensino_x_formacao'] = X['nivel_ensino'].astype(str) + "_" + X['formacao'].astype(str)

        return X

numerical_cols = ['idade', 'tempo_experiencia_dados', 'sql', 'nosql'] 
categorical_cols = ['genero', 'etnia', 'estado_moradia', 'nivel_ensino', 'formacao', 'cloud_preferida', 'linguagens_preferidas']
target_col = 'cargo'

X = df[numerical_cols + categorical_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

pipeline = ImbPipeline([
    ('interactions', FeatureInteractions()),  # adiciona interações
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])


param_grid = {
    'classifier__n_estimators': [200, 500],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("✅ Melhor modelo encontrado:")
print(grid.best_params_)

y_pred = best_model.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print("Balanced Accuracy:", balanced_acc)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# Precisamos transformar o X_train para pegar feature names do one-hot encoding
X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    importancias = pd.Series(best_model.named_steps['classifier'].feature_importances_,
                              index=np.array(best_model.named_steps['preprocessor'].get_feature_names_out()))
    print("\n📊 Top 15 Importância das variáveis:")
    print(importancias.sort_values(ascending=False).head(15))
