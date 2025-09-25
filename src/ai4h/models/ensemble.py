import math
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from ai4h.models.tree import (
    DecisionTreeClassifier,
    RandomStateType,
    check_random_state,
    gini,
)


def _generate_indices(n: int, n_b: int, random_state: RandomStateType = None):
    """draw indices for bootstrapped sample"""
    rng = check_random_state(random_state)
    sample_indices = rng.randint(low=0, high=n, size=n_b, dtype=np.int32)
    return sample_indices


def _fit_one_tree(
    X: NDArray,
    y: NDArray,
    tree: DecisionTreeClassifier,
    n: int,
    n_b: int,
):
    idc = _generate_indices(n, n_b, tree.random_state)
    tree.fit(X[idc], y[idc])
    return tree


class RandomForestClassifier:
    def __init__(
        self,
        max_depth: int = 32,
        min_samples_leaf: int = 1,
        max_samples: int | None = None,
        max_features: Literal["sqrt"] = "sqrt",
        n_estimators: int = 10,
        criterion: Callable[[NDArray], float] = gini,
        random_state: RandomStateType = None,
        n_jobs: int = 4,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.criterion = criterion
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.estimators: list[DecisionTreeClassifier] = []
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs

    def fit(self, X: NDArray, y: NDArray):
        n = X.shape[0]
        if self.max_samples is None:
            n_b = X.shape[0]
        else:
            n_b = self.max_samples
        n_features = X.shape[1]
        match self.max_features:
            case "sqrt":
                n_feat_max = math.floor(math.sqrt(n_features))  # floor
            case _:
                raise ValueError("n max")

        trees = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=n_feat_max,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state.randint(np.iinfo(np.int32).max),
            )
            for _ in range(self.n_estimators)
        ]
        fitted_trees = list(
            Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(_fit_one_tree)(
                    X,
                    y,
                    tree,
                    n,
                    n_b,
                )
                for tree in trees
            )
        )
        trees = cast(list[DecisionTreeClassifier], fitted_trees)
        self.estimators = trees

    def predict(self, X: NDArray):
        yhat = np.zeros(shape=(X.shape[0], len(self.estimators)))
        for i, tree in enumerate(self.estimators):
            yhat[:, i] = tree.predict(X)
        # majority vote
        modes = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=yhat
        )
        return modes