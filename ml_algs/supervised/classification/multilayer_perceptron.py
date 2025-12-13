import numpy as np

class MLP:
    '''Multilayer Perceptron binary classification algorithm class'''
    def __init__(self, n_hidden=10, lr=0.1, epochs=1000):
        '''Inits model params/values'''
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _sigmoid(self, z):
        '''Sigmoid activation'''
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        '''Derivative of sigmoid given activation'''
        return a * (1 - a)

    def fit(self, X, y):
        '''Fits MLP to given X, y data'''
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
            raise ValueError('Multilayer Perceptron supports only binary {0,1}')

        n_samples, n_features = X.shape

        # param inits
        self.b1 = np.zeros(self.n_hidden)
        self.b2 = 0.0
        self.W1 = np.random.randn(n_features, self.n_hidden) * np.sqrt(1 / n_features)
        self.W2 = np.random.randn(self.n_hidden, 1) * np.sqrt(1 / self.n_hidden)

        # training
        for _ in range(self.epochs):
            # forward pass
            Z1 = np.zeros((n_samples, self.n_hidden))
            A1 = np.zeros_like(Z1)

            for i in range(n_samples):
                for j in range(self.n_hidden):
                    Z1[i, j] = np.dot(X[i], self.W1[:, j]) + self.b1[j]
                    A1[i, j] = self._sigmoid(Z1[i, j])

            Z2 = np.zeros(n_samples)
            A2 = np.zeros(n_samples)

            for i in range(n_samples):
                Z2[i] = np.dot(A1[i], self.W2[:, 0]) + self.b2
                A2[i] = self._sigmoid(Z2[i])

            # backward pass
            dZ2 = A2 - y
            dW2 = np.zeros_like(self.W2)
            db2 = 0.0

            for j in range(self.n_hidden):
                for i in range(n_samples):
                    dW2[j, 0] += dZ2[i] * A1[i, j]
            dW2 /= n_samples
            db2 = np.mean(dZ2)

            dZ1 = np.zeros_like(A1)

            for i in range(n_samples):
                for j in range(self.n_hidden):
                    dZ1[i, j] = (
                        dZ2[i] * self.W2[j, 0] *
                        self._sigmoid_derivative(A1[i, j])
                    )

            dW1 = np.zeros_like(self.W1)
            db1 = np.zeros_like(self.b1)

            for j in range(self.n_hidden):
                for k in range(n_features):
                    for i in range(n_samples):
                        dW1[k, j] += dZ1[i, j] * X[i, k]
                dW1[:, j] /= n_samples
                db1[j] = np.mean(dZ1[:, j])

            # update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        return self

    def predict(self, X):
        '''Predicts y labels from new X'''
        # early prediction error handler
        if self.W1 is None:
            raise AttributeError('MLP not fitted yet')

        X = np.array(X)
        # invalid input error handlers
        if X.ndim != 2:
            raise ValueError('X must be 2D')
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')

        n_samples = X.shape[0]
        preds = []
        # pred loop
        for i in range(n_samples):
            hidden = []
            for j in range(self.n_hidden):
                z = np.dot(X[i], self.W1[:, j]) + self.b1[j]
                hidden.append(self._sigmoid(z))

            z_out = np.dot(hidden, self.W2[:, 0]) + self.b2
            y_hat = self._sigmoid(z_out)
            preds.append(1 if y_hat >= 0.5 else 0)

        return np.array(preds)