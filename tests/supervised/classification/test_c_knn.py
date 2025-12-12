import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_algs.supervised.classification.c_knn import KNN_Class

# general functionality tests

def test_format():
    # test that the formatting system works
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = KNN_Class(k=1).fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert preds.dtype == y.dtype


def test_k1():
    # test choice of 1 between 2 neighbors
    X = np.array([[0], [10]])
    y = np.array([0, 1])

    model = KNN_Class(k=1).fit(X, y)

    # closer to 0, should be y=0
    assert model.predict([[1]])[0] == 0

    # closer to 10, should be y=1
    assert model.predict([[9]])[0] == 1


def test_majority():
    # test funcionality of basic majority vote
    X = np.array([[0], [1], [2]])
    y = np.array([0, 0, 1])

    model = KNN_Class(k=3).fit(X, y)
    # 3 neighbors, 2 have y=0, so new pt should have y=0
    assert model.predict([[1]])[0] == 0


def test_iris():
    # test on iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    model = KNN_Class(k=5).fit(X_train, y_train)
    preds = model.predict(X_test)

    # expect general high accuracy (model "works")
    accuracy = np.mean(preds == y_test)
    assert accuracy > 0.9

# edge case tests

def test_k_too_large():
    # test with choosing more neighbors than there are points
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    model = KNN_Class(k=10).fit(X, y)
    preds = model.predict([[2]])

    # should still work :p
    assert preds[0] in y

def test_equidistant():
    # test with equidistant neighbors
    X_train = np.array([[0], [2]])
    y_train = np.array([0, 1])
    model = KNN_Class(k=2)
    model.fit(X_train, y_train)

    # test point is equidistant from both training points
    pred = model.predict([[1]])

    # prediction should be either 0 or 1 (majority vote tie-breaker)
    assert pred[0] in [0, 1]

# invalid input

def test_empty():
    # test proper error handling with empty input
    model = KNN_Class(k=1)
    model.fit(np.array([[0]]), np.array([0]))

    with pytest.raises(ValueError):
        model.predict(np.array([]))

def test_nonnumeric():
    # test proper error handling with nonnumeric input
    model = KNN_Class(k=1)
    model.fit(np.array([[0]]), np.array([0]))

    with pytest.raises(TypeError):
        model.predict([["a"]])