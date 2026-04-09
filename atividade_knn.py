import numpy as np
import urllib.request as urllib
import math

from algoritmo_KNN import KNN

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

data = np.genfromtxt(urllib.urlopen(url), delimiter=",")

y = data[:, 0]
X = data[:, 1:14]

def train_test_split(X, y, test_size=0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")
    n_samples = len(X)
    print("Quantidade da amostra: ", n_samples)
    indices = np.random.permutation(n_samples)
    print("Indices embaralhados: ", indices)
    n_test = math.ceil(n_samples * test_size)
    print("Quantidade de amostras para teste: ", n_test)
    test_indices = indices[:n_test]
    print("Dados de teste: ", test_indices)
    train_indices = indices[n_test:]
    print("Dados de treino: ", train_indices)

    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices, :], X[test_indices, :]
    y_train, ytest = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, ytest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

modelo = KNN()
modelo.fit(X_train, y_train)
modelo.predict(X_test)





