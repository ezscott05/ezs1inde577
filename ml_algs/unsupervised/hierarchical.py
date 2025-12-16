import numpy as np

class HierarchicalClustering:
    '''Agglomerative hierarchical clustering algorithm class'''
    def __init__(self, n_clusters=2):
        '''Validates and inits n_clusters'''
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise TypeError('n_clusters must be a positive integer')
        self.n_clusters = n_clusters
        self.labels_ = None

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

        # precompute full distance matrix (vectorized)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)

        # initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        cluster_dist = dist_matrix.copy()

        # merge until desired number of clusters
        while len(clusters) > self.n_clusters:
            # find closest pair of clusters
            i, j = np.unravel_index(np.argmin(cluster_dist), cluster_dist.shape)

            # merge clusters
            clusters[i] += clusters[j]
            clusters.pop(j)

            # update distance matrix using single linkage
            cluster_dist[i] = np.minimum(cluster_dist[i], cluster_dist[j])
            cluster_dist[:, i] = cluster_dist[i]

            # remove merged cluster
            cluster_dist = np.delete(cluster_dist, j, axis=0)
            cluster_dist = np.delete(cluster_dist, j, axis=1)
            cluster_dist[i, i] = np.inf

        # assign labels
        labels = np.zeros(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            labels[cluster] = idx

        self.labels_ = labels
        return self

    def predict(self, X):
        '''Straightforward fitting model to points and returning labels'''
        self.fit(X)
        return self.labels_
