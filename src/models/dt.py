from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

def runDT(X_train, X_test, y_train, y_test):
    # Definição da grade de parâmetros

    param_grid = {
        "criterion": ["gini","entropy"],                  # medida de qualidade da divisão - "gini", 
        "splitter": ["random", "best"],                   # estratégia de divisão - "best", 

        "max_features": [None],                 # nº de atributos considerados - "sqrt", "log2",
        "max_depth": [5],                       # profundidade máxima
        "max_leaf_nodes": [10, 12],               # número máximo de folhas - None - ,12

        "min_samples_split": [2,4],               # mínimo de amostras para dividir um nó
        "min_samples_leaf": [1],                # mínimo de amostras em uma folha - ,2,3,4
        "min_weight_fraction_leaf": [0.0],      # fração mínima de peso em uma folha
        "min_impurity_decrease": [0.0],         # impureza mínima p/ dividir

        "random_state": [None],                   # semente p/ reprodutibilidade - 42, 
        "class_weight": [None, "balanced"],                 # pesos das classes -  classe1:1, classe2:5
        "ccp_alpha": [0.0],                     # parâmetro de poda (complexity pruning)
        "monotonic_cst": [None]                 # restrições de monotonicidade
    }


    # Modelo base
    clf = tree.DecisionTreeClassifier(random_state=42)

    # GridSearch com validação cruzada
    grid = GridSearchCV(clf, param_grid, cv=4, scoring="f1_weighted", n_jobs=-1)
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
