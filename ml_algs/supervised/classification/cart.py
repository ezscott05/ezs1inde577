import numpy as np

class Cart:
    '''CART algorithm classifier'''
    def __init__(self, max_depth=5, min_samples_split=2):
        '''inits classifier max depth and min split'''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        '''Node subclass'''
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def _gini(self, y):
        '''Calculates gini impurity for a set of labels (a node)'''
        _, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        for c in counts:
            p = c / len(y)
            impurity -= p * p
        return impurity

    def _best_split(self, X, y):
        '''Searches for the optimal split at the current node'''
        best_feature = None
        best_threshold = None
        best_gain = 0.0

        # initial gini impurity to check against
        current_impurity = self._gini(y)
        n_samples, n_features = X.shape

        for f in range(n_features):
            values = X[:, f]
            thresholds = np.unique(values)

            for t in thresholds:
                left_idx = [i for i in range(n_samples) if values[i] <= t]
                right_idx = [i for i in range(n_samples) if values[i] > t]

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                left_y = y[left_idx]
                right_y = y[right_idx]

                # proportions sent to left vs right children
                p_left = len(left_y) / n_samples
                p_right = 1.0 - p_left

                # compute impurity of split
                impurity = (p_left * self._gini(left_y) + p_right * self._gini(right_y))
                gain = current_impurity - impurity

                # choose split if best checked
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold, best_gain

    def fit(self, X, y):
        '''Validates and trains model (majority of work done in build_tree and best_split)'''
        X = np.array(X)
        y = np.array(y)

        # improper dimension handlers
        if X.ndim != 2:
            raise ValueError('X must be 2-dimensional (n_samples, n_features)')
        n_samples = X.shape[0]
        if n_samples == 0 or y.size == 0:
            raise ValueError('X and y cannot be empty')
        if n_samples != y.shape[0]:
            raise ValueError('Number of samples in X and y must be equal')

        # improper type handler
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')
        
        # build tree
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        '''Builds the tree :p'''
        num_samples = X.shape[0]
        num_labels = len(np.unique(y))

        # leaf conditions
        if (
            depth >= self.max_depth
            or num_samples < self.min_samples_split
            or num_labels == 1
            or X.shape[1] == 0
        ):
            majority_class = self._majority_class(y)
            return Cart.Node(value=majority_class)

        # chooses best split
        feature, threshold, gain = self._best_split(X, y)

        if gain == 0 or feature is None:
            # no useful split
            return Cart.Node(value=self._majority_class(y))

        # splits data on the split
        left_idx = [i for i in range(num_samples) if X[i, feature] <= threshold]
        right_idx = [i for i in range(num_samples) if X[i, feature] > threshold]

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Cart.Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right
        )

    def _majority_class(self, y):
        '''Default value when conditions are not met'''
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict_one(self, x):
        '''Single value predictor'''
        node = self.tree
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        '''Multi value predictor'''
        X = np.array(X)
        return np.array([self._predict_one(row) for row in X])