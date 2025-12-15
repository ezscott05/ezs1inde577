import numpy as np
import pytest
from ml_algs.unsupervised.pca import PCA

# general functionality tests

def test_basic():
    # basic check for reduced dimensionality
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])

    pca = PCA(n_components=1)
    Z = pca.fit_transform(X)

    assert Z.shape == (4, 1)

def test_attributes():
    # checks that attributes are computed and stored
    X = np.random.randn(10, 3)

    pca = PCA(n_components=2)
    pca.fit(X)

    assert pca.components_ is not None
    assert pca.explained_variance_ is not None
    assert pca.components_.shape == (3, 2)

def test_fit_transform():
    # checks that fit_transform is the same as fitting and then transforming. obvious if not black box
    X = np.random.randn(20, 5)

    pca1 = PCA(n_components=3)
    Z1 = pca1.fit_transform(X)

    pca2 = PCA(n_components=3)
    pca2.fit(X)
    Z2 = pca2.transform(X)

    assert np.allclose(Z1, Z2)

# edge case tests

def test_keep():
    # n_components = None should keep all
    X = np.random.randn(8, 4)

    pca = PCA()
    Z = pca.fit_transform(X)

    assert Z.shape == (8, 4)

def test_single_feature():
    # test with single feature input
    X = np.array([[1], [2], [3], [4]])

    pca = PCA(n_components=1)
    Z = pca.fit_transform(X)

    assert Z.shape == (4, 1)

def test_too_many_components():
    # test with too many components (should cap at # of features)
    X = np.random.randn(5, 2)

    pca = PCA(n_components=5)
    Z = pca.fit_transform(X)

    assert Z.shape == (5, 2)

# invalid input tests

def test_empty():
    # test proper error handling with empty input
    X = np.array([])

    pca = PCA()
    with pytest.raises(ValueError):
        pca.fit(X)

def test_wrong_dim():
    # test proper error handling with 1D X when 2D expected
    X = np.array([1, 2, 3, 4])

    pca = PCA()
    with pytest.raises(ValueError):
        pca.fit(X)

def test_nonnumeric():
    # test proper error handling with nonnumeric input
    X = np.array([
        ['blehhh', 'buh'],
        ['guh', 'hyuck']
    ])

    pca = PCA()
    with pytest.raises(TypeError):
        pca.fit(X)

def test_single_sample():
    # test with single sample, while normally an edge case, is not valid for PCA
    X = np.array([[1.0, 2.0, 3.0]])

    pca = PCA(n_components=2)
    with pytest.raises(ValueError):
        pca.fit(X)