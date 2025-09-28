from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np

def runNaive(X_train, X_test, y_train, y_test):
    feature_names = list(X_train.columns)
    class_names = np.unique(y_train).astype(str)
    # 🔄 Balancear os dados com SMOTE (opcional)
    usar_smote = True
    if usar_smote:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # 🔍 Definir espaço de busca
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    }

    # ⚡ GridSearch com validação cruzada
    grid = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # 🎯 Melhor modelo
    best_nb = grid.best_estimator_
    print("Melhor alpha encontrado:", grid.best_params_['alpha'])

    # 📊 Avaliar no conjunto de teste
    y_pred = best_nb.predict(X_test)
    
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

    if hasattr(best_nb, "feature_log_prob_"):
        probs = np.exp(best_nb.feature_log_prob_)
        for i, cls in enumerate(class_names):
            print(f"\nClasse {cls}:")
            for f, feature in enumerate(feature_names):
                print(f" - {feature}: prob condicional={probs[i][f]:.3f}")


def testNauve(X_train, X_test, y_train, y_test):
    # 📊 Testar diferentes Naive Bayes
    modelos = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "CategoricalNB": CategoricalNB()
    }

    print("==== Avaliação Naive Bayes ====")
    for nome, modelo in modelos.items():
        acc, f1, precision, recall = avaliar_modelo(modelo, X_train, X_test, y_train, y_test, usar_smote=True)
        print(f"\n{nome}")
        print(f"Acurácia:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

   

def avaliar_modelo(modelo, X_train, X_test, y_train, y_test, usar_smote=False):
    if usar_smote:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    return acc, f1, precision, recall