# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# load data
print("Loading training and test data...")
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
test_ids = test_df['id']

X = train_df.drop(columns=['id', 'labels'])
y = train_df['labels']
X_test = test_df.drop(columns=['id'])
print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Test data: {X_test.shape[0]} rows, {X_test.shape[1]} features\n")

# identify features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['category_code']

# preprocession pipeline
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# model setup
lgbm_model = lgb.LGBMClassifier(
    objective='binary',
    class_weight='balanced',
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgbm_model)
])

# cross-validation metrics
print("Running 5-fold cross-validation to estimate model performance...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = ['accuracy', 'f1', 'roc_auc']
cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

# print per-fold metrics
for i in range(5):
    print(f"Fold {i+1}: Accuracy={cv_results['test_accuracy'][i]:.3f}, "
          f"F1={cv_results['test_f1'][i]:.3f}, ROC-AUC={cv_results['test_roc_auc'][i]:.3f}")

# print average metrics
print("\n=== LightGBM Cross-Validation Metrics (Average) ===")
print(f"Average Accuracy : {np.mean(cv_results['test_accuracy']):.3f}")
print(f"Average F1-score : {np.mean(cv_results['test_f1']):.3f}")
print(f"Average ROC-AUC  : {np.mean(cv_results['test_roc_auc']):.3f}\n")

# train model
print("Training LightGBM on the full training set...")
pipeline.fit(X, y)
print("Training completed!\n")

# feature importance plot
print("Generating feature importance plot...")
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = pipeline.named_steps['classifier'].feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='mako')
plt.title("Top 15 Features - LightGBM")
plt.tight_layout()
plt.show()

# predict test data and submission
print("Making predictions for the test set and creating submission file...")
test_predictions = pipeline.predict(X_test)
submission_df = pd.DataFrame({'id': test_ids, 'labels': test_predictions})
submission_df.to_csv('submission_lightgbm.csv', index=False)
print("✅ Submission file 'submission_lightgbm.csv' created successfully!")