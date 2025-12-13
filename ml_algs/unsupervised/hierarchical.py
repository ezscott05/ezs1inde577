import numpy as np

class HierarchicalClustering:
    '''Agglomerative hierarchical clustering algorithm class'''
    def __init__(self, n_clusters=2):
        '''Validates and inits n_clusters'''
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise TypeError('n_clusters must be a positive integer')
        self.n_clusters = n_clusters
        self.labels_ = None

    def _euclidean_distance(self, x1, x2):
        '''Same ol' Euclidean distance (interchangeable)'''
        total = 0.0
        for i in range(len(x1)):
            diff = x1[i] - x2[i]
            total += diff * diff
        return np.sqrt(total)

    def _cluster_distance(self, c1, c2, X):
        '''Distance between two clusters as minimum distance between single points in each cluster'''
        # single linkage :3
        min_dist = float('inf')
        for i in c1:
            for j in c2:
                dist = self._euclidean_distance(X[i], X[j])
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def fit(self, X):
        '''Fits model to a set of X'''
        X = np.array(X)

        # input error handlers
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError('X must be a non-empty 2D array')
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')
        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError('n_clusters cannot exceed number of samples')

        # initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]

        # merge until desired number of clusters
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            pair = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(clusters[i], clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        pair = (i, j)

            i, j = pair
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]

        # assign labels
        labels = np.zeros(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = idx

        self.labels_ = labels
        return self

    def predict(self, X):
        '''Straightforward fitting model to points and returning labels'''
        self.fit(X)
        return self.labels_