from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml


fp_proj = Path(__file__).parents[2]
fp_data = fp_proj.joinpath("data")
fp_data.mkdir(exist_ok=True)
mnist_file = fp_data / "mnist.npz"


def onehot(y: NDArray[np.uint8], n_classes: int = 10) -> NDArray[np.uint8]:
    return np.eye(n_classes, dtype=np.uint8)[y]


def prepare_nielsen(
    X: NDArray[np.int64],
    y: NDArray[np.uint8],
    y_transform=lambda x: x.reshape(10, 1),
    one_hot: bool = True,
) -> list[tuple[NDArray, NDArray]]:
    """Massages data into the shape MN's code expects."""
    return [
        (x.reshape(784, 1) / 255, y_transform(v))
        for x, v in zip(X, onehot(y) if one_hot else y)
    ]


def download_mnist():
    if mnist_file.exists():
        print("Mnist data already exist in data folder")
        return

    mnist = fetch_openml("mnist_784", as_frame=False)
    X = mnist["data"].astype(np.uint8)  # type: ignore
    y = mnist["target"].astype(np.uint8)  # type: ignore

    np.savez_compressed(mnist_file, X=X, y=y)
    print(f"Mnist data saved to {fp_data}")


def load_mnist() -> tuple[NDArray[np.int64], NDArray[np.uint8]]:
    if not mnist_file.exists():
        raise ValueError("Mnist data hasn't been downloaded")
    data = np.load(fp_data / "mnist.npz")
    X, y = data["X"], data["y"]
    return X, y

def softmax(z: NDArray, axis: int = 1) -> NDArray:
    z_shift = z - z.max(axis=axis, keepdims=True)  # numerical stability
    expz = np.exp(z_shift)
    return expz / expz.sum(axis=axis, keepdims=True)

def cross_entropy(y: NDArray, y_pred: NDArray, eps: float = 1e-9) -> float:
    ''' Computes the cross-entropy loss'''
    return -np.sum(y * np.log(y_pred + eps))

def sigmoid(z: NDArray):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z: NDArray):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

class BasicNetwork:
    """Basic network with 1 hidden layer (30 units)."""

    def __init__(self, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.W_1 = rng.normal(0, 0.1, size=(30, 784))
        self.b_1 = np.zeros((30, 1))
        self.W_2 = rng.normal(0, 0.1, size=(10, 30))
        self.b_2 = np.zeros((10, 1))

    def predict(self, x: NDArray) -> NDArray:
        """Forward pass for a single column vector"""
        z1 = self.W_1 @ x + self.b_1
        a1 = sigmoid(z1)
        z2 = self.W_2 @ a1 + self.b_2
        return softmax(z2, axis=0)  # (10,1)

    def train_step(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
        '''Performs one step of gradient descent using backpropagation.'''

        # Forward pass
        a0 = x # (784, 1)
        z1 = self.W_1 @ a0 + self.b_1
        a1 = sigmoid(z1)
        z2 = self.W_2 @ a1 + self.b_2
        yhat = softmax(z2, axis=0)  # (10,1)

        # loss
        loss = cross_entropy(y, yhat)

        # Backpropagation
        delta2 = yhat - y
        dW2 = np.outer(delta2, a1.T)
        db2 = delta2
        delta1 = (self.W_2.T @ delta2) * sigmoid_prime(z1)
        dW1 = np.outer(delta1, a0.T)
        db1 = delta1

        return loss, dW1, db1, dW2, db2

    def train_update(
       self, mini_batch: list[tuple[NDArray, NDArray]], lr: float
   ) -> float:
         '''Updates the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.'''
         n = len(mini_batch)
         total_loss = 0.0
         sum_dW1 = np.zeros_like(self.W_1)
         sum_db1 = np.zeros_like(self.b_1)
         sum_dW2 = np.zeros_like(self.W_2)
         sum_db2 = np.zeros_like(self.b_2)
    
         for x, y in mini_batch:
              loss, dW1, db1, dW2, db2 = self.train_step(x, y)
              total_loss += loss
              sum_dW1 += dW1
              sum_db1 += db1
              sum_dW2 += dW2
              sum_db2 += db2
    
         # Update weights and biases
         self.W_1 -= (lr / n) * sum_dW1
         self.b_1 -= (lr / n) * sum_db1
         self.W_2 -= (lr / n) * sum_dW2
         self.b_2 -= (lr / n) * sum_db2
    
         return total_loss / n