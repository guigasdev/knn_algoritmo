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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = KNN()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("Dados previstos: ", y_pred)
print("Dados reais: ", y_test)

#y_true = [1, 0, 1, 1, 0]
#y_pred = [1, 0, 0, 1, 0]

#VP = 2
#VN = 2
#FP = 1
#FN = 0


#def acuracia(vp, fp, fn, vn):
#    return (vp + vn) / (vp + fp + fn + vn)


#def precisao(vp, fp, fn, vn):
#    return vp / (vp + fp)


#def recall(vp, fp, fn, vn):
#    return vp / (vp + fn)


#def f_score(vp, fp, fn, vn):
#    p = precisao(vp, fp, fn, vn)
 #   r = recall(vp, fp, fn, vn)
  #  return 2 * p * r / (p + r)


#print("Acuracia: ", acuracia(VP, FP, FN, VN))
#print("Precisão: ", precisao(VP, FP, FN, VN))
#print("Recall: ", recall(VP, FP, FN, VN))
#print("F_Score: ", f_score(VP, FP, FN, VN))

# acuracia problema multiclass -> mais de duas classes

print("Dados previstos: ", y_pred)
print("Dados reais: ", y_test)

acertos = 0

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        acertos+=1
        

acc = acertos/len(y_test)
print("A acuracia é: ", acc)
classes = [1,2,3]
precisoes = []

for c in classes:
    vp = 0
    fp = 0
    for yt, yp in zip(y_test, y_pred):
        if yp == c:
            if yt == c:
                vp +=1
            else:
                fp += 1
            
    precisao = vp / (vp+fp)
    precisoes.append(precisao)
    
macro_precision = sum(precisoes) / len(precisoes)
print("A macro precision é: ", macro_precision)