import numpy as np
import pytest
from ml_algs.unsupervised.dbscan import DBSCAN

# basic functionality tests

def test_two_clusters():
    # two very obvious clusters
    X = np.array([
        [0, 0], [0, 1], [1, 0],
        [10, 10], [10, 11], [11, 10]
    ])

    model = DBSCAN(eps=1.5, min_samples=2)
    labels = model.predict(X)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    # should find only two clusters
    assert len(unique_labels) == 2

def test_noise():
    # fourth point clear outlier, should be labeled noise
    X = np.array([
        [0, 0], [0, 1], [1, 0],
        [10, 10]
    ])

    model = DBSCAN(eps=1.5, min_samples=2)
    labels = model.predict(X)

    assert labels[-1] == -1

# edge case tests

def test_single():
    # dbscan should not form a cluster from just one point
    X = np.array([[0, 0]])

    model = DBSCAN(eps=1.0, min_samples=2)
    labels = model.predict(X)

    # Cannot form a cluster
    assert labels[0] == -1

def test_one_cluster():
    X = np.array([
        [0, 0],
        [0.1, 0.1],
        [0.2, 0.2]
    ])

    model = DBSCAN(eps=1.0, min_samples=2)
    labels = model.predict(X)

    # should only form the one cluster
    assert len(set(labels)) == 1

def test_eps_too_small():
    # with too small an epsilon, no clusters should be found, resolving all to noise
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ])

    model = DBSCAN(eps=0.01, min_samples=2)
    labels = model.predict(X)

    assert np.all(labels == -1)

# invalid input tests

def test_empty():
    # typical empty input test
    model = DBSCAN()
    with pytest.raises(ValueError):
        model.predict([])

def test_wrong_dimension():
    # 1D X where 2D is needed
    model = DBSCAN()
    with pytest.raises(ValueError):
        model.predict([1, 2, 3])

def test_nonnumeric():
    # nonnumeric X where numbers expected
    X = np.array([
        ['a', 'b'],
        ['c', 'deoxyribonucleic acid']
    ])

    model = DBSCAN()
    with pytest.raises(TypeError):
        model.predict(X)