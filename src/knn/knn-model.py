import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv('academic-stress-level.csv')

print(train_df)
X = train_df.drop(columns=['class'])
y = train_df['class']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "n_neighbors": list(range(1, 60)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Melhores hiperparâmetros:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred, average="weighted"))
print("Recall:", recall_score(y_val, y_val_pred, average="weighted"))
print("F1-score:", f1_score(y_val, y_val_pred, average="weighted"))