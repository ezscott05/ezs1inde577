import numpy as np
import pytest
from ml_algs.supervised.classification.random_forest import RandomForest
from ml_algs.supervised.classification.cart import Cart

# general functionality tests

def test_simple():
    # test with simple single-feature data
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = RandomForest(n_estimators=5, max_depth=3)
    model.fit(X, y)

    preds = model.predict([[0], [3]])
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})


def test_multiple_features():
    # test with multiple features to learn from
    X = np.array([
        [1, 2],
        [1, 3],
        [10, 12],
        [11, 10]
    ])
    y = np.array([0, 0, 1, 1])

    model = RandomForest(n_estimators=7, max_depth=4)
    model.fit(X, y)
    preds = model.predict([[1, 2], [11, 10]])

    assert preds[0] == 0
    assert preds[1] == 1

# edge case tests

def test_single_sample():
    # test with only one sample to learn from
    X = np.array([[1, 2]])
    y = np.array([1])

    model = RandomForest(n_estimators=3)
    model.fit(X, y)
    preds = model.predict([[1, 2]])

    assert preds[0] == 1


def test_k_too_large():
    # test with more trees than samples
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    model = RandomForest(n_estimators=50)
    model.fit(X, y)
    pred = model.predict([[2]])[0]

    assert pred in y

# test invalid input handling

def test_mismatch():
    # test mismatched X, y sizes
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0])

    model = RandomForest()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_empty():
    # test empty input
    model = RandomForest()

    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))


def test_nonnumeric():
    # test nonnumeric input
    X = np.array([['red'], ['blue']])
    y = np.array([0, 1])

    model = RandomForest()

    with pytest.raises(TypeError):
        model.fit(X, y)


def test_predict_before_fit():
    # test attempting to predict with a model that has not been fit yet
    model = RandomForest()

    with pytest.raises(AttributeError):
        model.predict([[1]])
