import numpy as np
from ml_algs.supervised.classification.cart import Cart

class RandomForest:
    '''Random forest classifier built on CART'''
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 max_features=None, bootstrap=True):
        '''inits hyperparameters'''
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap

        self.trees = []
        self.feature_subsets = []

    def fit(self, X, y):
        '''Fits the model using random CART'''
        X = np.array(X)
        y = np.array(y)

        # invalid data case handlers
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be non-empty 2D array.")
        if y.size == 0 or X.shape[0] != y.shape[0]:
            raise ValueError("Invalid data dimensions.")
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError("X must contain numeric values.")

        self.n_samples, self.n_features = X.shape
        self.trees = []
        self.feature_subsets = []

        # determine number of features per tree
        if self.max_features is None:
            m = int(np.sqrt(self.n_features)) or 1
        else:
            m = self.max_features

        for _ in range(self.n_estimators):
            # feature subset
            feat_idx = np.random.choice(self.n_features, m, replace=False)
            self.feature_subsets.append(feat_idx)

            # bootstrap or full sample
            if self.bootstrap:
                idx = np.random.choice(self.n_samples, self.n_samples, replace=True)
                X_boot = X[idx][:, feat_idx]
                y_boot = y[idx]
            else:
                X_boot = X[:, feat_idx]
                y_boot = y

            tree = Cart(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        # error handlers
        if not hasattr(self, 'trees') or len(self.trees) == 0:
            raise AttributeError('RandomForest not fitted yet.')
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError('X must be 2D.')
        if X.shape[1] != self.n_features:
            raise ValueError('Feature count mismatch.')

        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values.')

        all_preds = []
        for tree, feats in zip(self.trees, self.feature_subsets):
            preds = tree.predict(X[:, feats])
            all_preds.append(preds)

        # majority vote across trees
        all_preds = np.array(all_preds)
        final_preds = []
        for col in all_preds.T:
            values, counts = np.unique(col, return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        return np.array(final_preds)