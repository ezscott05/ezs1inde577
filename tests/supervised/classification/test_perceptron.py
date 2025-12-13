import numpy as np
import pytest
from ml_algs.supervised.classification.perceptron import Perceptron

# basic functionality tests

def test_separable():
    # test prediction on model fit to same obviously separable data
    X = np.array([
        [1, 1],
        [2, 2],
        [-1, -1],
        [-2, -2]
    ])
    y = np.array([1, 1, 0, 0])

    model = Perceptron(lr=0.1, epochs=50)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y)

def test_predict_new():
    # test prediction on new data with similar, obviously separable data
    X = np.array([
        [1, 1],
        [-1, -1]
    ])
    y = np.array([1, 0])

    model = Perceptron(lr=0.1, epochs=20)
    model.fit(X, y)

    preds = model.predict([[2, 2], [-2, -2]])
    assert np.array_equal(preds, [1, 0])

# edge case tests

def test_single_sample():
    # if there is only one sample for fitting and prediction, it should be predicted as it is
    X = np.array([[1, 1]])
    y = np.array([1])

    model = Perceptron(lr=0.1, epochs=10)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds[0] == 1

def test_constant():
    # if all feature data is the same, output should still be valid but not necessarily well-informed
    X = np.array([
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    y = np.array([0, 1, 1])

    model = Perceptron(lr=0.1, epochs=20)
    model.fit(X, y)
    preds = model.predict(X)

    assert set(preds).issubset({0, 1})

def test_non_separable():
    # nonseparable XOR gate data, can predict incorrectly but should still be valid
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])

    model = Perceptron(lr=0.1, epochs=50)
    model.fit(X, y)
    preds = model.predict(X)

    assert set(preds).issubset({0, 1})

# invalid input tests

def test_predict_before_fit():
    # simple prediction on model that has not been fit yet
    model = Perceptron()
    with pytest.raises(AttributeError):
        model.predict([[1, 2]])

def test_empty():
    # simple empty input test
    model = Perceptron()
    with pytest.raises(ValueError):
        model.fit([], [])

def test_mismatch():
    # test X and y dimension
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1])

    model = Perceptron()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_nonbinary():
    # once again the model is not woke enough
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 2])

    model = Perceptron()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_nonnumeric():
    # the model should reject nonnumeric data properly (sorry able sisters)
    X = np.array([['sable'], ['mabel']])
    y = np.array([0, 1])

    model = Perceptron()
    with pytest.raises(TypeError):
        model.fit(X, y)