import numpy as np

class NaiveBayes:
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)
        self.mean = np.zeros((n_classes, n_features), dtype=float)
        self.var = np.zeros((n_classes, n_features), dtype=float)
        self.prior = np.zeros(n_classes, dtype=float)
        
        
    #def predict(X_test):
X_train = np.array([[3,2,1],
                   [4,5,6]])

n_samples, n_features = X_train.shape

print(n_samples)
print(n_features)

y_train = ["Sim", "Sim", "Não", "Não"]
n_classes = np.unique(y_train)
n_classes = len(n_classes)
mean = np.zeros((n_classes, n_features), dtype=np.float64)
print(mean)
prior = np.zeros(n_classes, dtype=np.float64)
print(prior)




