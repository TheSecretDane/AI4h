from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Callable
from numpy.random import RandomState

from dataclasses import dataclass

# The below four functions compute different impurity measures - determining how good a split is. 

def cprobs(y: npt.NDArray[np.integer]):
    # Computes the class probabilities using np.bincount 
    return np.bincount(y) / y.shape[0]

def gini(y: npt.NDArray[np.integer]):
    # Computes the Gini impurity
    p = cprobs(y)
    return np.sum(p * (1 - p))

def entropy(y: npt.NDArray[np.integer]):
    # Computes the entropy
    p = cprobs(y)
    return -np.sum(p * np.log2(p + 1e-15))  # Adding a small constant to avoid log(0)

def misclassification(y: npt.NDArray[np.integer]):
    # Computes the misclassification error
    p = cprobs(y)
    return 1 - np.max(p)

def mse(y: npt.NDArray[np.float64]) -> float:
    # mean squared error impurity measure for regression trees
    return float(np.mean((y-y.mean())**2))

def goodness_split(y_left: npt.NDArray, y_right: npt.NDArray, imp_parent: float, impurity_fn: Callable[[npt.NDArray], float] = gini) -> float:
    # Computes the goodness of a split using the specified impurity measure
    if impurity_fn not in [gini, entropy, misclassification, mse]:
        raise ValueError("impurity_fn must be one of gini, entropy, misclassification, or mse (for regression trees)")
    
    N_left, N_right = y_left.shape[0], y_right.shape[0]
    N = N_left + N_right

    imp_left, imp_right = impurity_fn(y_left), impurity_fn(y_right)
    p_L = N_left / N
    p_R = N_right / N

    if imp_parent is None:
        imp_parent = impurity_fn(np.concatenate([y_left, y_right]))

    return imp_parent - p_L * imp_left - p_R * imp_right

def best_split(
    X: npt.NDArray,
    y: npt.NDArray,
    *,
    impurity_fn: Callable[[npt.NDArray], float] = gini,
    min_samples_leaf: int = 1,
) -> tuple[int, float, float]:
    ''' Find the best split of the data X, y using the specified impurity measure in a greedy manner.'''

    n, d = X.shape

    if n < 2 * min_samples_leaf:
        print(f' Not enough samples to split: {n} < {2 * min_samples_leaf}')
        return -1, np.nan, 0.0
    
    imp_parent = impurity_fn(y)

    # initialize container for best split
    best_feature, best_threshold, best_gain = -1, np.nan, 0.0

    for j in range(d):
        xj = X[:, j]
        x_sorted = np.argsort(xj, kind="stable")
        x_sorted, y_sorted = xj[x_sorted], y[x_sorted]

        for i in range(min_samples_leaf, n - min_samples_leaf):
            if x_sorted[i] == x_sorted[i + 1]:
                continue
            gain = goodness_split(
                y_sorted[: i + 1], y_sorted[i + 1 :], imp_parent, impurity_fn)
            
            if gain > best_gain:
                best_feature, best_threshold, best_gain = j, (x_sorted[i] + x_sorted[i + 1]) / 2, gain
    
    return best_feature, best_threshold, best_gain

# Copied from Exercise 2 as tasked
def compute_pred(y: npt.NDArray) -> float:
    """computes predicted values.

    Guesses whether classification or regression tree
    by inspecting dtype of outcome variable.
    """
    if np.issubdtype(y.dtype, np.integer):
        return int(np.argmax(np.bincount(y)))
    return float(y.mean())


@dataclass
class TerminalNode:
    prediction: float
    impurity: float
    depth: int
    n_samples: int

    def predict_one(self, _):
        return self.prediction

    def predict(self, X):
        return np.full(len(X), self.prediction)

    def pretty(self, indent=""):
        return f"{indent}Leaf(pred={self.prediction:.2f}, n={self.n_samples})\n"


@dataclass
class NonterminalNode:
    feature: int
    threshold: float
    left: Node
    right: Node
    impurity: float
    depth: int
    n_samples: int

    def predict_one(self, x):
        return (
            self.left.predict_one(x)
            if x[self.feature] <= self.threshold
            else self.right.predict_one(x)
        )

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def pretty(self, indent=""):
        out = (
            f"{indent}Node(f{self.feature}<={self.threshold:.2f}, n={self.n_samples})\n"
        )
        return out + self.left.pretty(indent + "  ") + self.right.pretty(indent + "  ")


Node = TerminalNode | NonterminalNode

def build_tree(
    X: npt.NDArray,
    y: npt.NDArray,
    *,
    impurity_fn: Callable[[npt.NDArray], float] = gini,
    max_depth: int = 32,
    min_samples_leaf: int = 1,
    depth: int = 0,
) -> Node:
    
    feature, threshold, gain = best_split(X, y, impurity_fn=impurity_fn, min_samples_leaf=min_samples_leaf)

    if depth >= max_depth or y.shape[0] < 2 * min_samples_leaf or np.unique(y).shape[0] == 1 or feature == -1 or not np.isfinite(threshold) or gain <= 0.0:
        return TerminalNode(prediction=compute_pred(y), impurity=impurity_fn(y), depth=depth, n_samples=y.shape[0])
    else:
        mask = X[:, feature] <= threshold
        return NonterminalNode(
            feature = feature,
            threshold = threshold,
            left = build_tree(X[mask], y[mask], impurity_fn=impurity_fn, max_depth=max_depth, min_samples_leaf=min_samples_leaf, depth=depth+1),
            right = build_tree(X[~mask], y[~mask], impurity_fn=impurity_fn, max_depth=max_depth, min_samples_leaf=min_samples_leaf, depth=depth+1),
            impurity = impurity_fn(y),
            depth = depth,
            n_samples = y.shape[0]
        )

# NOTE: Extra/Optional stuff

RandomStateType = int | None | RandomState


def check_random_state(random_state: RandomStateType):
    """helper function that follows sklearn"""
    match random_state:
        case int():
            return np.random.RandomState(random_state)
        case RandomState():
            return random_state
        case None:
            return np.random.RandomState(0)
    raise ValueError


def draw_features(
    n_features: int, max_features: int, random_state: RandomStateType = None
):
    """draw features."""
    rng = check_random_state(random_state)
    return rng.choice(n_features, size=max_features, replace=False)


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 32,
        min_samples_leaf: int = 1,
        criterion: Callable[[npt.NDArray], float] = gini,
        max_features: int | None = None,
        random_state: RandomStateType = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree_: Node
        self.random_state = check_random_state(random_state)
        self.max_features = max_features

    def fit(self, X: npt.NDArray, y: npt.NDArray):
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.tree_ = build_tree(
            X=X,
            y=y,
            impurity_fn=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"criterion={self.criterion.__name__ if callable(self.criterion) else self.criterion}, "
            f"random_state={self.random_state}"
            f")"
        )

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        if self.tree_ is None:
            raise ValueError("Not fitted yet")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.tree_.predict(X)