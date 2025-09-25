from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def runNaive(X_train, X_test, y_train, y_test):
    # Inicializar Naive Bayes
    nb = CategoricalNB()

    # Treinar modelo
    nb.fit(X_train, y_train)

    # Prever no conjunto de teste
    y_pred = nb.predict(X_test)

    # Avaliar
    acc = accuracy_score(y_test, y_pred)
    print("Acurácia:", acc)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print("Naive Bayes - Avaliação no teste:")
    print("Acurácia:", acc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)