import numpy as np
import pytest
from ml_algs.supervised.regression.ridge_lasso import Ridge

# general functionality tests

def test_basic():
    # test with basic straightforward synthetic data
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = Ridge(lr=0.01, epochs=2000, alpha=0.1).fit(X, y)
    preds = model.predict([[5]])

    assert np.isclose(preds[0], 10, atol=0.5)

def test_weights():
    # test that weights adjust properly
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])

    no_reg = Ridge(alpha=0.0, lr=0.01, epochs=2000).fit(X, y)
    strong_reg = Ridge(alpha=10.0, lr=0.01, epochs=2000).fit(X, y)

    assert abs(strong_reg.weights[0]) < abs(no_reg.weights[0])

# edge case tests

def test_zero_feature():
    # test with no feature data
    X = np.empty((4, 0))
    y = np.array([2, 4, 6, 8])

    model = Ridge(alpha=0.5).fit(X, y)
    preds = model.predict(np.empty((2, 0)))

    assert np.allclose(preds, np.mean(y))


def test_single():
    # test with only one observation to learn from
    X = np.array([[10]])
    y = np.array([20])

    model = Ridge(lr=0.01, epochs=500, alpha=0.3).fit(X, y)
    p = model.predict([[10]])[0]

    assert np.isclose(p, 20, atol=1.0)


def test_constant():
    # test with constant feature data
    X = np.array([[3], [3], [3], [3]])
    y = np.array([1, 2, 3, 4])

    model = Ridge(lr=0.001, epochs=3000, alpha=1.0).fit(X, y)
    preds = model.predict([[3]])

    assert np.isclose(preds[0], np.mean(y), atol=0.5)

# invalid input tests

def test_mismatch():
    # test proper error handling with mismatched X,y sizes
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])

    with pytest.raises(ValueError):
        Ridge().fit(X, y)

def test_empty():
    # test proper error handling with empty input
    with pytest.raises(ValueError):
        Ridge().fit([], [])