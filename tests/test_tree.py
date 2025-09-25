"""
Some (probably too) simple tests for the exercise class.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def _get_data():
    """helper to load and split data for tests"""
    from typing import cast

    # https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.int64],
        ],
        train_test_split(X, y, random_state=2025, train_size=0.8),
    )

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = _get_data()


def test_cprobs():
    from ai4h.models.tree import cprobs

    y1 = np.array([2, 2, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 0, 1])
    y2 = np.array([2, 2, 2, 2, 2, 0, 0, 2, 1, 0, 0, 1, 1, 2, 1, 1, 0, 1, 2, 1])

    assert np.allclose(cprobs(y1), np.array([0.3, 0.4, 0.3]))
    assert np.allclose(cprobs(y2), np.array([0.25, 0.35, 0.4]))


def test_gini():
    from ai4h.models.tree import gini

    y1 = np.array([2, 2, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 0, 1])
    y2 = np.array([2, 2, 2, 2, 2, 0, 0, 2, 1, 0, 0, 1, 1, 2, 1, 1, 0, 1, 2, 1])

    g1 = round(gini(y1), 4)
    g2 = round(gini(y2), 4)

    assert np.allclose(g1, 0.66)
    assert np.allclose(g2, 0.655)


def test_goodness_split():
    from ai4h.models.tree import gini, goodness_split

    y1 = np.array([2, 2, 1, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 0, 1])
    y2 = np.array([2, 2, 2, 2, 2, 0, 0, 2, 1, 0, 0, 1, 1, 2, 1, 1, 0, 1, 2, 1])
    y = np.concat((y1, y2))

    gs = round(goodness_split(y1, y2, imp_parent=gini(y), impurity_fn=gini), 6)
    assert np.allclose(gs, 0.00375)


def test_build_tree():
    """test functions related to impurity"""

    from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifierSkLearn
    from ai4h.models.tree import build_tree, gini

    # Build own tree
    tree = build_tree(X_train, y_train, impurity_fn=gini, min_samples_leaf=5)

    print(tree)

    print(tree.pretty())

    # Build sklearn
    clf = DecisionTreeClassifierSkLearn(min_samples_leaf=5)
    clf.fit(X_train, y_train)

    # Check the prediction from the lecture
    mask_path = (X_test[:, 2] > 2.5) & (X_test[:, 3] < 1.5)
    i = np.where(mask_path)[0][0]
    xi = X_test[i].reshape(1, 4)

    # See https://ai-for-humanity-ucph.github.io/2025/slides/lecture-2/#/9
    assert np.allclose(xi, np.array([[6.1, 2.6, 5.6, 1.4]]))

    yh_own = tree.predict(xi)
    yh_sk = clf.predict(xi)

    # Wrong prediction
    assert y_test[i] == 2
    assert yh_own == 1
    assert yh_own == yh_sk

    # build many trees and check; good enough for the exercises
    min_samples_leaf = 5
    for max_depth in range(1, 6):
        print(f"{min_samples_leaf=}, {max_depth=}")
        root = build_tree(
            X_train,
            y_train,
            impurity_fn=gini,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )
        clf = DecisionTreeClassifierSkLearn(
            min_samples_leaf=min_samples_leaf, max_depth=max_depth
        )
        clf.fit(X_train, y_train)
        pred_sklearn = clf.predict(X_test)
        pred_own = root.predict(X_test)
        assert np.allclose(pred_own, pred_sklearn)


def _get_diabetes_data():
    """helper to load and split data for tests"""
    from typing import cast
    from sklearn.datasets import load_diabetes

    # https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.int64],
        ],
        train_test_split(X, y, random_state=2025, train_size=0.8),
    )
    return X_train, X_test, y_train, y_test


def test_regression_tree():
    """test regression tree"""

    from sklearn.tree import DecisionTreeRegressor
    from ai4h.models.tree import build_tree, mse

    X_train, X_test, y_train, _ = _get_diabetes_data()

    tree = build_tree(
        X_train, y_train, impurity_fn=mse, min_samples_leaf=10, max_depth=4
    )

    clf = DecisionTreeRegressor(min_samples_leaf=10, max_depth=4)
    clf.fit(X_train, y_train)

    pred_sklearn = clf.predict(X_test)
    pred_own = tree.predict(X_test)

    assert np.allclose(pred_own, pred_sklearn)
