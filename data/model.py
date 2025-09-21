# importar bibliotecas necessárias
import pandas as pd
import numpy as np

# para pré-processamento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# para o modelo de Machine Learning
import lightgbm as lgb

# importing for charts generation
import matplotlib.pyplot as plt
import seaborn as sns

print("Bibliotecas importadas com sucesso!")

# carregar os dados
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("Arquivos 'train.csv' e 'test.csv' carregados.")
except FileNotFoundError:
    print("ERRO: Certifique-se de que os arquivos 'train.csv' e 'test.csv' estão na mesma pasta que este script.")
    exit() # Encerra o script caso os arquivos não forem encontrados

# guarda os ids do conjunto de teste para a submissão final
test_ids = test_df['id']

# definir a variável alvo (y) e as features (X)
X = train_df.drop(columns=['id', 'labels'])
y = train_df['labels']

# o conjunto de teste não tem 'labels', então só remove o 'id'
X_test = test_df.drop(columns=['id'])

print(f"Dados de treino preparados: {X.shape[0]} amostras e {X.shape[1]} features.")
print(f"Dados de teste preparados: {X_test.shape[0]} amostras e {X_test.shape[1]} features.")

# identificar os tipos de colunas
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['category_code']

# criar o transformador para features numéricas
# SimpleImputer: preenche os NaNs com a mediana
# StandardScaler: ajusta a escala dos dados
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# criar o transformador para features categóricas
# SimpleImputer: preenche NaNs com a categoria mais frequente
# OneHotEncoder: transforma categorias em colunas de 0s e 1s
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# juntar os transformadores em um único pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

print("Pipeline de pré-processamento criado.")

# treinar o modelo
# definir o modelo LightGBM
# usando class_weight='balanced' para ajudar com o desbalanceamento
lgbm_model = lgb.LGBMClassifier(objective='binary', class_weight='balanced', random_state=42)

# criar o pipeline final que junta o pré-processador e o modelo
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm_model)
])

# treinar o modelo com todos os dados de treino
print("Iniciando o treinamento do modelo...")
model_pipeline.fit(X, y)
print("Modelo treinado com sucesso!")

# fazer previsões no conjunto de teste
print("Fazendo previsões nos dados de teste...")
test_predictions = model_pipeline.predict(X_test)

# criar o arquivo de submissão
# criar o DataFrame de submissão
submission_df = pd.DataFrame({'id': test_ids, 'labels': test_predictions})

# salvar o arquivo de submissão em formato .csv
submission_df.to_csv('submission.csv', index=False)

print("O arquivo 'submission.csv' foi criado na pasta atual!")

# target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(x=train_df['labels'], palette="Set2")
plt.title("Target Variable Distribution")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.show()

# top 15 Category Code Distribution
plt.figure(figsize=(12,6))
top_categories = train_df['category_code'].value_counts().nlargest(15)
sns.barplot(x=top_categories.index, y=top_categories.values, palette="viridis")
plt.title("Top 15 Category Codes Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# correlation Heatmap (numerical features)
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12,10))
corr = train_df[numerical_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # keep only lower triangle
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            annot=False, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# feature Importance (LightGBM)
# features and target
X = train_df.drop(columns=['id', 'labels'])
y = train_df['labels']

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['category_code']

# preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# model
lgbm_model = lgb.LGBMClassifier(objective='binary', class_weight='balanced', random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm_model)
])

pipeline.fit(X, y)

# feature importance
importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="mako")
plt.title("Top 15 Features - LightGBM Importance")
plt.show()