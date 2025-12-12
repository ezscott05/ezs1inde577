import numpy as np

class Lin:
    '''Basic linear class to use as a base for ridge and LASSO'''
    def __init__(self, lr=0.01, epochs=1000, alpha=0.1):
        '''Inits model params'''
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha

    def _fit_checks(self, X, y):
        '''Checks for valid data and minor config'''
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        # invalid data case handler
        if n_samples == 0 or y.size == 0 or n_samples != y.shape[0]:
            raise ValueError('Invalid data dimensions')

        self.n_samples, self.n_features = X.shape
        return X, y


class Ridge(Lin):
    '''Ridge regression class built on Lin'''
    def fit(self, X, y):
        '''Fits model on X and y, returns self.'''
        X, y = self._fit_checks(X, y)
        # no features case handler
        if self.n_features == 0:
            self.weights = np.zeros(0)
            self.bias = np.mean(y)
            return self
        # inits
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            dw = np.zeros_like(self.weights)
            db = 0.0

            for i in range(self.n_samples):
                diff = y_pred[i] - y[i]
                db += diff
                for j in range(self.n_features):
                    dw[j] += diff * X[i][j]

            dw /= self.n_samples
            db /= self.n_samples

            # penalty
            dw += self.alpha * self.weights / self.n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        '''Predicts new y from model and new X'''
        X = np.array(X)
        if self.n_features == 0:
            return np.full(X.shape[0], self.bias)
        preds = []
        for row in X:
            total = self.bias
            for j in range(self.n_features):
                total += self.weights[j] * row[j]
            preds.append(total)
        return np.array(preds)

class Lasso(Lin):
    '''LASSO regression model built on Lin'''
    def fit(self, X, y):
        '''Fits model on X and y, returns self'''
        X, y = self._fit_checks(X, y)
        # no features case handler
        if self.n_features == 0:
            self.weights = np.zeros(0)
            self.bias = np.mean(y)
            return self
        # inits
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            dw = np.zeros_like(self.weights)
            db = 0.0

            for i in range(self.n_samples):
                diff = y_pred[i] - y[i]
                db += diff
                for j in range(self.n_features):
                    dw[j] += diff * X[i][j]

            dw /= self.n_samples
            db /= self.n_samples

            # penalty
            dw += self.alpha * np.sign(self.weights) / self.n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        '''Predicts new y from model and new X'''
        X = np.array(X)
        if self.n_features == 0:
            return np.full(X.shape[0], self.bias)
        preds = []
        for row in X:
            total = self.bias
            for j in range(self.n_features):
                total += self.weights[j] * row[j]
            preds.append(total)
        return np.array(preds)
