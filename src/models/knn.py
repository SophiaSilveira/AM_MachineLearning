from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import shap
import lime
import lime.lime_tabular


def runKNN(X_train, X_test, y_train, y_test):
    feature_names = list(X_train.columns)
    class_names = np.unique(y_train).astype(str)
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

    explainer = shap.Explainer(best_knn.predict, X_train_scaled)
    shap_values = explainer(X_test_scaled[:50])  # primeiras 50 instâncias

    shap.summary_plot(shap_values, X_test_scaled[:50], feature_names=feature_names)

    # LIME
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_scaled),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    exp = explainer_lime.explain_instance(X_test_scaled[0], best_knn.predict_proba)
    exp.save_to_file("results/knn_inter_base1_lime.html")
