import numpy as np

class KMeans:
    '''K-Means clustering algorithm class'''
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        '''Initializes object and model parameters'''

        # invalid input error handlers
        if not isinstance(k, int) or k <= 0:
            raise TypeError('k must be a positive integer')
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise TypeError('max_iters must be a positive integer')
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise TypeError('tol must be a positive number')

        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def _euclidean_distance(self, x1, x2):
        '''Calculates Euclidean distance (replacable modular part)'''
        total = 0.0
        for i in range(len(x1)):
            diff = x1[i] - x2[i]
            total += diff * diff
        return np.sqrt(total)

    def fit(self, X):
        X = np.array(X)

        # invalid input error handlers
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError("X must contain numeric values")

        n_samples, n_features = X.shape
        if self.k > n_samples:
            raise ValueError("k cannot be greater than number of samples")

        # initialize centroids by sampling data points
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]

            # assignment step
            for i in range(n_samples):
                distances = [
                    self._euclidean_distance(X[i], self.centroids[j])
                    for j in range(self.k)
                ]
                cluster_idx = int(np.argmin(distances))
                clusters[cluster_idx].append(i)

            new_centroids = []
            for idxs in clusters:
                if len(idxs) == 0:
                    # empty cluster -> keep old centroid
                    new_centroids.append(self.centroids[len(new_centroids)])
                else:
                    mean = np.zeros(n_features)
                    for i in idxs:
                        mean += X[i]
                    mean /= len(idxs)
                    new_centroids.append(mean)

            new_centroids = np.array(new_centroids)

            # convergence check
            shift = 0.0
            for i in range(self.k):
                shift += self._euclidean_distance(self.centroids[i], new_centroids[i])

            self.centroids = new_centroids
            if shift < self.tol:
                break

        # store final labels
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        '''Predicts y-labels for new X'''

        # invalid input error handlers
        if self.centroids is None:
            raise AttributeError('KMeans instance is not fitted yet')

        X = np.array(X)
        if X.ndim != 2:
            raise ValueError('X must be 2D')

        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')

        labels = []
        for i in range(X.shape[0]):
            distances = [
                self._euclidean_distance(X[i], self.centroids[j])
                for j in range(self.k)
            ]
            labels.append(int(np.argmin(distances)))

        return np.array(labels)
