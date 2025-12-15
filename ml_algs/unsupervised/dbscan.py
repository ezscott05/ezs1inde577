import numpy as np

class DBSCAN:
    '''Density-Based Spatial Clustering of Applications with Noise algorithm class'''
    def __init__(self, eps=0.5, min_samples=5):
        '''Inits model parameters'''
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _euclidean_distance(self, x1, x2):
        '''Basic Euclidean distance, interchangeable'''
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _region_query(self, X, idx):
        '''Finds all points within eps distance of point X[idx]'''
        neighbors = []
        for i in range(len(X)):
            # dist check
            if self._euclidean_distance(X[idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id, visited):
        '''Expands a cluster by recursively adding density-connected points'''
        labels[idx] = cluster_id
        i = 0

        while i < len(neighbors):
            point = neighbors[i]
            # visit unvisited neighbor
            if not visited[point]:
                visited[point] = True
                point_neighbors = self._region_query(X, point)
                # if core, merge nbhds
                if len(point_neighbors) >= self.min_samples:
                    neighbors.extend(
                        p for p in point_neighbors if p not in neighbors
                    )
            # add label if previously labeled noise
            if labels[point] == -1:
                labels[point] = cluster_id

            i += 1

    def fit(self, X):
        '''Fits model to X, designating clusters and noise'''
        X = np.array(X)

        # input error handlers
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError('X must be a non-empty 2D array')
        try:
            X = X.astype(float)
        except Exception:
            raise TypeError('X must contain numeric values')
        # init all to unvisited noise
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(X, i)
            # mark noncore as noise
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                # expand from core
                self._expand_cluster(
                    X, labels, i, neighbors, cluster_id, visited
                )
                cluster_id += 1

        self.labels_ = labels
        return self

    def predict(self, X):
        '''Predict and fit are basically the same. This method exists for consistency'''
        self.fit(X)
        return self.labels_