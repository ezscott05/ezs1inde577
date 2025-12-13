import numpy as np
import pytest
from ml_algs.supervised.classification.gradient_boosting import GradientBoosting

# general functionality tests

def test_basic():
    # test on very basic data
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=2)
    model.fit(X, y)

    preds = model.predict([[0], [3]])
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

def test_multiple_features():
    # test with multiple features
    X = np.array([[1, 2], [2, 3], [10, 12], [11, 10]])
    y = np.array([0, 0, 1, 1])

    model = GradientBoosting(n_estimators=5, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    preds = model.predict([[1, 2], [11, 10]])

    assert preds[0] == 0
    assert preds[1] == 1

# edge case tests

def test_single_sample():
    # test with only one sample to learn from
    X = np.array([[1, 2]])
    y = np.array([1])

    model = GradientBoosting(n_estimators=3)
    model.fit(X, y)

    pred = model.predict([[1, 2]])[0]
    assert pred == 1

def test_constant_labels():
    # test with no variability in y
    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 1])

    model = GradientBoosting(n_estimators=5)
    model.fit(X, y)

    preds = model.predict([[0], [2]])
    assert all(pred == 1 for pred in preds)

def test_small_tree_large_learning_rate():
    # test with a small tree and fast learning
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = GradientBoosting(n_estimators=2, learning_rate=1.0, max_depth=1)
    model.fit(X, y)
    preds = model.predict([[0], [3]])
    assert set(preds).issubset({0, 1})

# invalid input tests

def test_empty():
    # test empty input error handling
    model = GradientBoosting()
    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))

def test_nonnumeric():
    # test nonnumeric input error handling
    X = np.array([['a'], ['b']])
    y = np.array([0, 1])

    model = GradientBoosting()
    with pytest.raises(TypeError):
        model.fit(X, y)

def test_nonbinary():
    # unfortunately our class should return an error when faced with the reality of many, fluid genders
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])

    model = GradientBoosting()
    with pytest.raises(TypeError):
        model.fit(X, y)

def test_y_not_1d():
    # test improper y dims error handling
    X = np.array([[0], [1]])
    y = np.array([[0], [1]])

    model = GradientBoosting()
    with pytest.raises(TypeError):
        model.fit(X, y)

def test_predict_before_fit():
    # test attempting to predict on a model that has not been fit yet
    model = GradientBoosting()
    with pytest.raises(AttributeError):
        model.predict([[0]])

def test_mismatch():
    # test mismatched X and y sizes error handling
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1])

    model = GradientBoosting()
    with pytest.raises(ValueError):
        model.fit(X, y)
