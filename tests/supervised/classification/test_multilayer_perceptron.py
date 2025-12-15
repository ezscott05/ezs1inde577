import numpy as np
import pytest
from ml_algs.supervised.classification.multilayer_perceptron import MLP

# general functionality tests

def test_basic():
    # basic fit to AND gate data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    model = MLP(n_hidden=4, lr=0.05, epochs=5000)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y)

def test_xor():
    # test on the same XOR gate data that fails in normal perceptron
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])

    model = MLP(n_hidden=4, lr=0.2, epochs=6000)
    model.fit(X, y)
    preds = model.predict(X)
    print(preds)

    assert np.array_equal(preds, y)

def test_output():
    # test valid output for random data
    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, size=20)

    model = MLP(n_hidden=6, epochs=500)
    model.fit(X, y)
    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})

def test_shape():
    # test properly shaped output for random data
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)

    model = MLP(n_hidden=5, epochs=500)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (10,)

# edge case tests

def test_single_sample():
    # check for valid pred on one sample
    X = np.array([[1.0, 2.0]])
    y = np.array([1])

    model = MLP(n_hidden=3, lr=0.1, epochs=500)
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (1,)
    assert pred[0] in (0, 1)

def test_constant():
    # test same output when fit to constant labels
    X = np.random.rand(20, 3)
    y = np.zeros(20, dtype=int)

    model = MLP(n_hidden=5, lr=0.1, epochs=500)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.all(preds == 0)

# invalid input tests

def test_predict_before_fit():
    # test proper error handling on early prediction
    model = MLP()
    X = np.array([[0, 0]])
    with pytest.raises(AttributeError):
        model.predict(X)

def test_X_dim():
    # test proper error handling on 1D X when 2D expected
    model = MLP()
    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        model.fit(X, y)

def test_mismatch():
    # test proper error handling on X,y with mismatched dims
    model = MLP()
    X = np.array([[0, 0], [1, 1]])
    y = np.array([1])

    with pytest.raises(ValueError):
        model.fit(X, y)

def test_nonnumeric():
    # test proper error handling on nonnumeric X
    model = MLP()
    X = np.array([['apple', 'blueberry'], ['cavendish', 'durian']])
    y = np.array([0, 1])

    with pytest.raises(TypeError):
        model.fit(X, y)