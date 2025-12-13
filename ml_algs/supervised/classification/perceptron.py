import numpy as np

class Perceptron:
    '''Binary Perceptron algorithm class'''

    def __init__(self, lr=0.01, epochs=1000):
        '''Inits object and model parameters'''
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''Fits model to given X data and corresponding binary y'''
        X = np.array(X)
        y = np.array(y)

        # invalid input error handlers
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError('X must be a non-empty 2D array')
        if y.size == 0 or X.shape[0] != y.shape[0]:
            raise ValueError('Invalid data dimensions')
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError('Perceptron supports only binary {0,1}')

        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # training
        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(self.weights, X[i]) + self.bias
                y_pred = 1 if linear_output >= 0 else 0

                error = y[i] - y_pred

                # update weights if pred incorrect
                if error != 0:
                    for j in range(n_features):
                        self.weights[j] += self.lr * error * X[i][j]
                    self.bias += self.lr * error

        return self

    def predict(self, X):
        '''Fairly simple prediction function'''
        # early run error handler
        if self.weights is None or self.bias is None:
            raise AttributeError('Perceptron not fitted yet')

        X = np.array(X)
        # invalid input error handlers
        if X.ndim != 2:
            raise ValueError('X must be 2D')
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')

        preds = []
        # prediction loop
        for row in X:
            linear_output = np.dot(self.weights, row) + self.bias
            preds.append(1 if linear_output >= 0 else 0)

        return np.array(preds)