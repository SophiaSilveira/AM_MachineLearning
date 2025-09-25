from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier

def runKNN(X_train, X_test, y_train, y_test):
    # 1️⃣ Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2️⃣ Oversampling com SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    # 3️⃣ Definir modelo base
    knn = KNeighborsClassifier()

    # 4️⃣ Definir hiperparâmetros para o grid
    param_grid = {
        'n_neighbors': list(range(5, 11)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }

    # 5️⃣ GridSearch com validação cruzada
    grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,                # 5-fold cross validation
        scoring='balanced_accuracy',  # pode trocar para 'accuracy' se preferir
        n_jobs=-1
    )

    grid.fit(X_train_bal, y_train_bal)

    # 6️⃣ Melhor modelo encontrado
    best_knn = grid.best_estimator_
    print("Melhores hiperparâmetros:", grid.best_params_)

    # 7️⃣ Avaliar no conjunto de teste
    y_pred = best_knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print("Acurácia:", acc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

def runKNNBase2(X_train, X_test, y_train, y_test):
    # 1️⃣ Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3️⃣ Definir modelo base
    knn = KNeighborsClassifier()

    # 4️⃣ Definir hiperparâmetros para o grid
    param_grid = {
        'n_neighbors': list(range(3, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }

    # 5️⃣ GridSearch com validação cruzada
    grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,                # 5-fold cross validation
        scoring='accuracy',  # pode trocar para 'accuracy' se preferir
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    # 6️⃣ Melhor modelo encontrado
    best_knn = grid.best_estimator_
    print("Melhores hiperparâmetros:", grid.best_params_)

    # 7️⃣ Avaliar no conjunto de teste
    y_pred = best_knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print("Acurácia:", acc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)