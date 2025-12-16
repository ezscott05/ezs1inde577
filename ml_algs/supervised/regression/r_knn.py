import numpy as np

class KNN_Reg:
    '''KNN Regression class.'''
    def __init__(self, k):
        '''
        Initializes training data and k.

        Parameters:
        k | int | number of nearest neighbors to find/use
        '''
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        '''
        Stores training data.

        Returns self to enable chaining.
        '''
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        # invalid data case handler
        if self.X_train.size == 0 or self.y_train.size == 0 or self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError('Invalid data dimensions')
        return self

    def _euclidean_distance(self, a, b):
        '''Euclidean distance btwn 2 pts, interchange for alt. distance methods.'''
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        '''
        Predicts the y values for test data X

        Parameters:
        X : array/like, test sample data for prediction.

        Returns np array predictions containing predicted y values.
        '''
        X = np.array(X)
        # invalid data case handler
        if X.size == 0:
            raise ValueError('Data cannot be empty')
        n_test = X.shape[0]
        predictions = np.zeros(n_test)
        for i, x in enumerate(X):
            # compute distances to all training points (vectorized bc old version was slow)
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nn_idx = np.argsort(distances)[:self.k]
            predictions[i] = np.mean(self.y_train[nn_idx])

        return predictions