import numpy as np
import math

x = np.array([34, 5, 6, 4])
random_state = None
np.random.seed(random_state)
print(np.random.permutation(x))  # embaralhamento
test_size = 0.3
n_samples = len(x)
n_test = math.ceil(n_samples * test_size)  # arredonda para cima
print(n_test)


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


X = np.array([4, 3, 2, 1, 8, 5, 23, 43])
y = np.array([32, 23, 54, 43, 54, 3, 23, 22])

dados = np.loadtxt("mtcars.csv", delimiter=",", skiprows=1)

y = dados[:, 0]
X = dados[:, 1:10]

print(y)
X = dados[:,]

X_train, X_test, y_train, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
