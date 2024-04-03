import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv('diabetes.csv')

columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_impute:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

ada_classifier = AdaBoostClassifier(
    estimator = RandomForestClassifier(class_weight = 'balanced', random_state = 42),
    random_state = 42,
    algorithm = 'SAMME'
)

ada_param_grid = {
    'estimator__max_depth': [5, 10, None],
    'estimator__n_estimators': [10, 50],
    'estimator__min_samples_split': [2, 5],
    'n_estimators': [30, 50],
    'learning_rate': [0.01, 0.1, 1]
}

ada_grid_search = GridSearchCV(
    estimator = ada_classifier,
    param_grid = ada_param_grid,
    cv = stratified_kfold,
    scoring = 'recall',
    n_jobs = -1
)
ada_grid_search.fit(X_train, y_train)

print("Best Parameters: ", ada_grid_search.best_params_)
print("Best Recall Score: ", ada_grid_search.best_score_)

best_ada_model = ada_grid_search.best_estimator_

y_pred_ada = best_ada_model.predict(X_test)

accuracy_ada = accuracy_score(y_test, y_pred_ada)
con_mat_ada = confusion_matrix(y_test, y_pred_ada)
report_ada = classification_report(y_test, y_pred_ada)

print("AdaBoost with RandomForest Confusion Matrix:\n", con_mat_ada)
print("AdaBoost with RandomForest Accuracy:", accuracy_ada)
print("AdaBoost with RandomForest Classification Report:\n", report_ada)