import numpy as np

class TSNE:
    '''t-SNE for dimensionality reduction to 2D or 3D'''
    def __init__(self, n_components=2, perplexity=30.0, lr=200.0, n_iter=1000):
        '''Inits model parameters'''
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iter = n_iter
        self.embedding_ = None

    def _pairwise_distances(self, X):
        '''Computes pairwise squared Euclidean distances on given X data'''
        sum_X = np.sum(np.square(X), axis=1)
        D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
        return D

    def _compute_joint_probabilities(self, X, tol=1e-5):
        '''Uses Gaussian kernel to translate distances into probabilities of similarity'''
        n = X.shape[0]
        D = self._pairwise_distances(X)
        P = np.exp(-D / (2.0 * (self.perplexity**2)))
        np.fill_diagonal(P, 0)
        P = P / np.sum(P)
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        return P

    def fit(self, X):
        '''Fits model to data X'''
        try:
            X = np.array(X, dtype=float)
        except ValueError:
            raise TypeError("X must contain numeric values")
        if X.size == 0:
            raise ValueError('X must be non-empty')
        n_samples = X.shape[0]
        self.embedding_ = np.random.randn(n_samples, self.n_components) * 1e-4

        P = self._compute_joint_probabilities(X)

        # gradient descent
        for i in range(self.n_iter):
            sum_Y = np.sum(np.square(self.embedding_), axis=1)
            D = -2.0 * np.dot(self.embedding_, self.embedding_.T)
            D = 1.0 / (1.0 + (D + sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :]))
            np.fill_diagonal(D, 0)

            Q = D / np.sum(D)
            Q = np.maximum(Q, 1e-12)

            grad = np.zeros_like(self.embedding_)
            for i in range(n_samples):
                grad[i] = 4.0 * np.sum((P[i, :, np.newaxis] - Q[i, :, np.newaxis]) * 
                                       (self.embedding_[i] - self.embedding_), axis=0)

            self.embedding_ -= self.lr * grad

        return self

    def transform(self, X=None):
        '''Returns the learned embedding'''
        if self.embedding_ is None:
            raise AttributeError("t-SNE not fitted yet.")
        return self.embedding_

    def fit_transform(self, X):
        '''Convenience function to fit and transform'''
        self.fit(X)
        return self.transform()
