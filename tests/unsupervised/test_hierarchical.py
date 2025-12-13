import numpy as np
import pytest
from ml_algs.unsupervised.hierarchical import HierarchicalClustering

# general functionality tests

def test_basic():
    # basic model should be valid
    X = np.array([
        [0, 0],
        [0, 1],
        [5, 5],
        [6, 5]
    ])

    model = HierarchicalClustering(n_clusters=2)
    model.fit(X)
    labels = model.predict(X)

    assert labels.shape == (4,)
    assert set(labels) == {0, 1}

def test_deterministic():
    # testing that the model is deterministic by running it twice
    X = np.array([
        [1, 1],
        [1, 2],
        [10, 10]
    ])

    model = HierarchicalClustering(n_clusters=2)
    model.fit(X)

    preds1 = model.predict(X)
    preds2 = model.predict(X)

    assert np.array_equal(preds1, preds2)

# edge case tests

def test_single_sample():
    # test single sample forming one cluster correctly
    X = np.array([[0.0, 0.0]])

    model = HierarchicalClustering(n_clusters=1)
    model.fit(X)
    labels = model.predict(X)

    assert labels.shape == (1,)
    assert labels[0] == 0

def test_identical():
    # test repeated identical samples clustering without error
    X = np.array([[1, 1], [1, 1], [1, 1]])

    model = HierarchicalClustering(n_clusters=2)
    model.fit(X)
    labels = model.predict(X)

    assert labels.shape == (3,)
    assert set(labels).issubset({0, 1})

def test_equal():
    # test 3 pts -> 3 clusters
    X = np.array([
        [0],
        [1],
        [2]
    ])

    model = HierarchicalClustering(n_clusters=3)
    model.fit(X)
    labels = model.predict(X)

    assert set(labels) == {0, 1, 2}

# invalid input test

def test_empty():
    # basic empty input test
    X = np.array([])

    model = HierarchicalClustering(n_clusters=2)

    with pytest.raises(ValueError):
        model.fit(X)

def test_wrong_dim():
    # 1D X where 2D expected
    X = np.array([1, 2, 3])

    model = HierarchicalClustering(n_clusters=2)

    with pytest.raises(ValueError):
        model.fit(X)

def test_nonnumeric():
    # nonnumeric input test where numeric expected
    X = np.array([
        ['antidisestablishmentarianism', 'b'],
        ['c', 'd']
    ])

    model = HierarchicalClustering(n_clusters=2)

    with pytest.raises(TypeError):
        model.fit(X)

def test_too_many_clusters():
    # test error raising when attempting to form more clusters than points
    X = np.array([
        [0],
        [1]
    ])

    model = HierarchicalClustering(n_clusters=5)

    with pytest.raises(ValueError):
        model.fit(X)