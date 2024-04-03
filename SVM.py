import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

param_grid = [
    {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.0001, 0.001, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]},
    {'kernel': ['poly'], 'C': [0.0001, 0.001, 0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1], 'degree': [2, 3], 'coef0': [0.0, 1.0]}
]

grid_search = GridSearchCV(
    SVC(class_weight = 'balanced'),
    param_grid,
    cv = stratified_kfold,
    scoring = 'recall',
    n_jobs = -1,
)
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Recall Score: ", grid_search.best_score_)

best_model = SVC(**grid_search.best_params_)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
con_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", con_mat)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)