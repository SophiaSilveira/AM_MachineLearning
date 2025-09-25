from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

def runDT(X_train, X_test, y_train, y_test):
    # Definição da grade de parâmetros

    param_grid = {
        "criterion": ["gini"],          #"entropy",  "log_loss", 
        "splitter": ["random"],         #"best", 
        "max_features": [None],          #"sqrt", "log2"
        "max_depth": [13],                 #20, 15,13,12,10,5, none
        "max_leaf_nodes": [None],           #,10,12, 20, 50,100,150,200

        "min_samples_split": [2],                # 2,4,5,10
        "min_samples_leaf": [1],                  #1, 3,4
        "min_weight_fraction_leaf": [0.0],              # ,0.1, 
        "min_impurity_decrease": [0.0],           # ,  0.01, 0.01

        "random_state": [42],                           # None, 0, 42,
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0], #, 
        "monotonic_cst": [None]
    }

    # Modelo base
    clf = tree.DecisionTreeClassifier(random_state=42)

    # GridSearch com validação cruzada
    grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1) #valores 5 ou 10
    grid.fit(X_train, y_train)

    # Melhor modelo encontrado
    best_model = grid.best_estimator_
    print(">>> Melhores parâmetros encontrados:")

    for key, value in grid.best_params_.items():
        print(f"{key}: {value}")

    print()

    # Predição com o melhor modelo
    y_pred = best_model.predict(X_test)

    # Métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Impressão das métricas
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Precisão: {precisao:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nRelatório detalhado por classe:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
