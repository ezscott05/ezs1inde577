import numpy as np

class PCA:
    '''Principal Component Analysis for dimensionality reduction'''

    def __init__(self, n_components=None):
        '''Inits model parameters'''
        self.n_components = n_components
        self.components_ = None  # eigenvectors
        self.mean_ = None        # mean of each feature
        self.explained_variance_ = None  # variance explained by each component

    def fit(self, X):
        '''Computes principal components from data X'''
    
        # invalid input error handlers
        try:
            X = np.asarray(X, dtype=float)
        except Exception:
            raise TypeError('X must contain numeric values')
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError('X must be a non-empty 2D array')
        if X.shape[0] < 2:
            raise ValueError('PCA requires at least two samples')

        # center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        cov_matrix = np.atleast_2d(cov_matrix)


        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvalues and eigenvectors descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store top n_components
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
            eigenvalues = eigenvalues[:self.n_components]

        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues

        return self

    def transform(self, X):
        '''Project data X onto the principal components'''
        X = np.array(X, dtype=float)
        if self.components_ is None:
            raise AttributeError('Cannot transform model that has not been fit yet')

        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        '''Fits to and transforms X'''
        self.fit(X)
        return self.transform(X)
