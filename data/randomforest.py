import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # for windows interactive plots

# find files relative to scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where script is
train_path = os.path.join(BASE_DIR, 'train.csv')
test_path = os.path.join(BASE_DIR, 'test.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_ids = test_df['id']

# prepare data
X = train_df.drop(columns=['id', 'labels'])
y = train_df['labels']
X_test = test_df.drop(columns=['id'])

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['category_code']

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

rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1', 'roc_auc']
cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

print("\nCross-Validation Metrics per Fold")
for i in range(5):
    print(f"Fold {i+1}: Accuracy={cv_results['test_accuracy'][i]:.3f}, "
          f"F1={cv_results['test_f1'][i]:.3f}, ROC-AUC={cv_results['test_roc_auc'][i]:.3f}")

print("\nAverage Metrics")
print(f"Accuracy : {np.mean(cv_results['test_accuracy']):.3f}")
print(f"F1-score : {np.mean(cv_results['test_f1']):.3f}")
print(f"ROC-AUC  : {np.mean(cv_results['test_roc_auc']):.3f}")

# train full model and feature importance
pipeline.fit(X, y)
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = pipeline.named_steps['classifier'].feature_importances_

feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}) \
           .sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='mako')
plt.title("Top 15 Features - RandomForest")
plt.tight_layout()
plt.show()

# predict test data and save
test_predictions = pipeline.predict(X_test)
submission_df = pd.DataFrame({'id': test_ids, 'labels': test_predictions})
submission_df.to_csv(os.path.join(BASE_DIR, 'submission_randomforest.csv'), index=False)
print("Submission file 'submission_randomforest.csv' created successfully!")