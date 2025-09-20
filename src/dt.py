from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def runDT(X_train, X_test, y_train, y_test):

    #Modelo
    clf = tree.DecisionTreeClassifier(criterion="gini", random_state=42)

    #Treinamento
    clf = clf.fit(X_train, y_train)

    # Predição 
    y_pred = clf.predict(X_test)

    # Métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Impressão das métricas
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Precisão: {precisao:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nRelatório detalhado por classe:\n")
    print(classification_report(y_test, y_pred))