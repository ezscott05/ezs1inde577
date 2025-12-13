import numpy as np
import pytest
from ml_algs.unsupervised.kmeans import KMeans

# general functionality tests

def test_basic():
    # basic prediction test
    X = np.array([
        [0, 0],
        [0, 1],
        [5, 5],
        [6, 5]
    ])

    model = KMeans(k=2, max_iters=50)
    model.fit(X)

    labels = model.predict(X)
    assert len(labels) == X.shape[0]
    assert set(labels).issubset({0, 1})

def test_labels_exist():
    # test for existence of labels attr
    X = np.array([[0], [1], [10], [11]])

    model = KMeans(k=2)
    model.fit(X)

    assert hasattr(model, "labels_")
    assert len(model.labels_) == len(X)

def test_predict():
    # another prediction test
    X = np.array([[0], [1], [10], [11]])

    model = KMeans(k=2)
    model.fit(X)

    preds = model.predict([[0], [11]])
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

# edge case tests

def test_single_cluster():
    # test with a single cluster
    X = np.array([[1], [2], [3]])

    model = KMeans(k=1)
    model.fit(X)

    labels = model.predict(X)
    assert all(label == 0 for label in labels)

def test_single_sample():
    # test with a single sample
    X = np.array([[5]])

    model = KMeans(k=1)
    model.fit(X)

    pred = model.predict([[5]])[0]
    assert pred == 0

def test_identical():
    # test with a single cluster of identical X
    X = np.array([[1, 1], [1, 1], [1, 1]])

    model = KMeans(k=2)
    model.fit(X)

    labels = model.predict(X)
    assert len(labels) == 3

def test_empty_cluster():
    # test that creates empty clusters (centroids at 0, 100)
    X = np.array([[0], [100]])

    model = KMeans(k=2, max_iters=5)
    model.fit(X)

    labels = model.predict(X)
    assert set(labels).issubset({0, 1})

# invalid input tests

def test_predict_before_fit():
    # test predicting with a model that has not been fit
    model = KMeans()
    with pytest.raises(AttributeError):
        model.predict([[0, 1]])

def test_empty():
    # test empty input error handling
    model = KMeans()
    with pytest.raises(ValueError):
        model.fit(np.array([]))

def test_nonnumeric():
    # test nonnumeric input error handling
    X = np.array([['a'], ['b']])

    model = KMeans()
    with pytest.raises(TypeError):
        model.fit(X)

def test_wrong_dim():
    # test fitting to 1-dimensional X
    X = np.array([1, 2, 3])

    model = KMeans()
    with pytest.raises(ValueError):
        model.fit(X)

def test_predict_wrong_dim():
    # test predicting with 1-dimensional X
    X = np.array([[1], [2]])

    model = KMeans(k=1)
    model.fit(X)

    with pytest.raises(ValueError):
        model.predict([1, 2])

def test_k_too_large():
    # test k > number of points
    X = np.array([[1], [2]])

    model = KMeans(k=3)
    with pytest.raises(ValueError):
        model.fit(X)