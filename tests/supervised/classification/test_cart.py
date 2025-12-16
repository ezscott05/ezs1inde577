import numpy as np
import pytest
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split

from ml_algs.supervised.classification.cart import Cart

# general functionality tests

def test_simple_split():
    # test on synthetic easily split data
    X = np.array([[0], [1], [2], [10], [11], [12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = Cart(max_depth=2, min_samples_split=2)
    clf.fit(X, y)
    preds = clf.predict([[1], [11]])
    assert preds.tolist() == [0, 1]

def test_iris():
    # test on iris dataset
    iris = load_iris()
    # fairly separable binary data
    X = iris.data[iris.target < 2]
    y = iris.target[iris.target < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    clf = Cart(max_depth=5, min_samples_split=2)
    clf.fit(X_train, y_train)
    acc = (clf.predict(X_test) == y_test).mean()
    assert acc >= 0.9

# hyperparameter behavior tests

def test_max_depth():
    # test max depth behavior
    X, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=0)
    clf_shallow = Cart(max_depth=1)
    clf_deep = Cart(max_depth=10)
    clf_shallow.fit(X, y)
    clf_deep.fit(X, y)

    # deep tree should not be worse than shallow
    train_acc_shallow = (clf_shallow.predict(X) == y).mean()
    train_acc_deep = (clf_deep.predict(X) == y).mean()
    assert train_acc_deep >= train_acc_shallow

def test_no_split():
    X = np.array([[0],[1],[2],[3],[4]])
    y = np.array([0,0,1,1,1])
    clf = Cart(min_samples_split=6)  # larger than n_samples
    clf.fit(X, y)
    # if no split, should predict majority
    preds = clf.predict(X)
    assert np.all(preds == np.bincount(y).argmax())

# edge case tests

def test_single_sample():
    # test with only one sample to learn from
    X = np.array([[42]])
    y = np.array([1])
    clf = Cart()
    clf.fit(X, y)
    assert clf.predict([[42]])[0] == 1

def test_zero_features():
    # test with zero features should fall back to majority
    X = np.empty((4, 0))
    y = np.array([0, 1, 1, 1])
    clf = Cart()
    clf.fit(X, y)
    preds = clf.predict(np.empty((2, 0)))
    assert np.all(preds == np.array([1, 1]))

def test_all_same_label():
    # if all labels are the same the prediction should be that label
    X = np.array([[0],[1],[2]])
    y = np.array([1,1,1])
    clf = Cart()
    clf.fit(X, y)
    assert np.all(clf.predict(X) == 1)

def test_tie():
    # should choose *something* in a tie (instead of error)
    X = np.array([[0],[2]])
    y = np.array([0,1])
    clf = Cart()
    clf.fit(X, y)
    pred = clf.predict([[1]])[0]
    assert pred in (0, 1)

# invalid input tests

def test_predict_before_fit():
    # test error handling when predict run with no model
    clf = Cart()
    with pytest.raises(AttributeError):
        _ = clf.predict([[0]])


def test_empty():
    # test error handling when no data provided
    clf = Cart()
    with pytest.raises(ValueError):
        clf.fit(np.array([]), np.array([]))


def test_mismatch():
    # test error handling when data dimensions are mismatched
    clf = Cart()
    X = np.array([[0], [1]])
    y = np.array([0])
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_nonnumeric():
    # test error handling when data is nonnumeric
    clf = Cart()
    X = np.array([["a"], ["b"]], dtype=object)
    y = np.array([0, 1])
    with pytest.raises(TypeError):
        clf.fit(X, y)