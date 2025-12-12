import numpy as np

class Log_Reg:
    '''Logistic Regression algorithm class'''
    def __init__(self, lr=0.01, epochs=1000):
        '''initializes object, learning rate, and epochs'''
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        '''computes sigmoid'''
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        '''
        Fits model
        
        Parameters:
        - training regressors X
        - training response y
        
        Uses helper functions to fit a model to the data, returns self.
        '''
        X = np.array(X)
        y = np.array(y)
        # invalid data case handler
        if X.shape[0] == 0 or y.size == 0 or X.shape[0] != y.shape[0]:
            raise(ValueError('Invalid data dimensions'))

        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0
        # no feature case handler
        if self.n_features == 0:
            self.bias = np.mean(y)
            return self

        for _ in range(self.epochs):
            # main diff btwn this and lin is the prediction function
            y_pred = self.predict_p(X)

            dw = np.zeros_like(self.weights)
            db = 0.0

            for i in range(self.n_samples):
                diff = y_pred[i] - y[i]
                db += diff
                for j in range(self.n_features):
                    dw[j] += diff * X[i][j]

            dw /= self.n_samples
            db /= self.n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_p(self, X):
        '''Computes and returns predicted probabilities (np array) from data'''
        X = np.array(X)
        probs = []
        for row in X:
            z = self.bias
            for j in range(self.n_features):
                z += self.weights[j] * row[j]
            probs.append(self._sigmoid(z))
        return np.array(probs)

    def predict(self, X):
        '''Finalizes prediction by converting probabilities to (0,1)'''
        return (self.predict_p(X) >= 0.5).astype(int)
