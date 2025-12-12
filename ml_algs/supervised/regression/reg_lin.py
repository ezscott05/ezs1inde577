import numpy as np

class Lin_Reg:
    '''Linear Regression algorithm class'''
    def __init__(self, lr=0.01, epochs=1000):
        '''initializes object, learning rate, and epochs'''
        self.lr = lr
        self.epochs = epochs

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
        # no features case handler
        if self.n_features == 0:
            self.bias = np.mean(y)
            return self

        for _ in range(self.epochs):
            # main diff btwn this and log is the prediction function
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

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X):
        '''Uses existing weights to predict y for new X'''
        X = np.array(X)
        preds = []
        for row in X:
            total = self.bias
            for j in range(self.n_features):
                total += self.weights[j] * row[j]
            preds.append(total)
        return np.array(preds)
