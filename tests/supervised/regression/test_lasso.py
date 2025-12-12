import numpy as np
import pytest
from ml_algs.supervised.regression.ridge_lasso import Lasso

# general functionality tests

def test_basic():
    # test with basic, predictable synthetic data
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = Lasso(lr=0.01, epochs=2500, alpha=0.1).fit(X, y)
    preds = model.predict([[5]])

    assert np.isclose(preds[0], 10, atol=0.5)

def test_weights():
    # testing that the weights adjust properly
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])

    no_reg = Lasso(alpha=0.0, lr=0.01, epochs=2000).fit(X, y)
    strong_reg = Lasso(alpha=10.0, lr=0.01, epochs=2000).fit(X, y)

    assert abs(strong_reg.weights[0]) < abs(no_reg.weights[0])

# edge case tests

def test_zero_feature():
    # test with no feature data
    X = np.empty((5, 0))
    y = np.array([5, 6, 7, 8, 9])

    model = Lasso(alpha=0.2).fit(X, y)
    preds = model.predict(np.empty((3, 0)))

    assert np.allclose(preds, np.mean(y))

def test_single():
    # test with one observation to learn from
    X = np.array([[4]])
    y = np.array([9])

    model = Lasso(lr=0.01, epochs=600, alpha=0.2).fit(X, y)
    p = model.predict([[4]])[0]

    assert np.isclose(p, 9, atol=1.0)

def test_constant():
    # test with constant feature data
    X = np.array([[2], [2], [2], [2]])
    y = np.array([1, 2, 3, 4])

    model = Lasso(lr=0.001, epochs=3000, alpha=1.0).fit(X, y)
    preds = model.predict([[2]])

    assert np.isclose(preds[0], np.mean(y), atol=0.5)

# invalid input test

def test_mismatch():
    # test proper error handling with mismatched X,y sizes
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])

    with pytest.raises(ValueError):
        Lasso().fit(X, y)

def test_lasso_empty_input():
    # test proper error handling with empty input
    with pytest.raises(ValueError):
        Lasso().fit([], [])