import numpy as np
from ml_algs.supervised.classification.cart import Cart

class GradientBoosting:
    '''Gradient boosting classifier class'''
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_split=2):
        '''validates and initializes model parameters'''
        
        # error handlers
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise TypeError('n_estimators must be a positive integer')
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise TypeError('learning_rate must be a positive number')
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise TypeError('max_depth must be a positive integer')
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise TypeError('min_samples_split must be an integer ≥ 2')

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _sigmoid(self, z):
        '''calculates sigmoid'''
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        '''fits model '''
        self.trees = []
        self.initial_log_odds = None
        X = np.asarray(X)
        y = np.asarray(y)

        # error handlers
        if y.ndim != 1:
            raise TypeError('y must be 1D')
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError('GradientBoosting supports only binary {0,1}')
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be non-empty 2D array.")
        if y.size == 0 or X.shape[0] != y.shape[0]:
            raise ValueError('Invalid data dimensions')
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')

        # initial model is log odds
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.initial_log_odds = np.log(p / (1 - p))

        # start with constant score
        f = np.full(len(y), self.initial_log_odds)

        # boosting loop
        for _ in range(self.n_estimators):
            # negative gradient for log-loss
            residual = y - self._sigmoid(f)

            # fit CART on residuals
            tree = Cart(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residual)
            self.trees.append(tree)

            # update model
            f += self.learning_rate * tree.predict(X)

        return self

    def predict_p(self, X):
        X = np.asarray(X)

        f = np.full(X.shape[0], self.initial_log_odds)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)

        proba = self._sigmoid(f)
        return np.vstack((1 - proba, proba)).T

    def predict(self, X):
        if not hasattr(self, 'trees') or not hasattr(self, 'initial_log_odds'):
            raise AttributeError('GradientBoosting instance is not fitted yet. Call fit() first')
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError('X must be non-empty 2D array')
        return (self.predict_p(X)[:, 1] >= 0.5).astype(int)