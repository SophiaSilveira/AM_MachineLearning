from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

def runRF(X_train, X_test, y_train, y_test):
    # Grid de parâmetros
    param_grid_rf = {
        "n_estimators": [350], # número de árvores 500, 300
        "class_weight": ["balanced_subsample"], # balanceamento de classes , "balanced"
        "criterion": ["gini"], # critério de divisão "gini",

        "min_samples_split": [7], # mínimo de amostras para dividir um nó

        "max_depth": [None], # profundidade máxima 5
        "max_leaf_nodes": [None], # número máximo de folhas
        "max_samples": [0.9], # fração de amostras por árvore
        "max_features": ["sqrt", "log2"], # número de features por divisão , , 1.0

        "random_state": [42], # controle de aleatoriedade
        "n_jobs": [-1] # número de jobs em paralelo
    }

    # Modelo base
    rf = RandomForestClassifier(random_state=42)

    # GridSearch com validação cruzada
    grid_rf = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_rf,
        cv=5,                      # número de folds ≤ menor classe
        scoring="f1_weighted",
        n_jobs=-1
    )

    # Treinamento
    grid_rf.fit(X_train, y_train)

    # Melhor modelo encontrado
    best_rf = grid_rf.best_estimator_
    print(">>> Melhor RandomForest:")
    for key, value in grid_rf.best_params_.items():
        print(f"{key}: {value}")
    print()

    # Predição
    y_pred = best_rf.predict(X_test)

    # Métricas
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    print("\nRelatório detalhado por classe:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

