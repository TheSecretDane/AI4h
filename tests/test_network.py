"""
Some (probably too) simple tests for the exercise class.
"""

import numpy as np
from ai4h.models.network import load_mnist, prepare_nielsen
from ai4h.models import network

# Load data:
X, y = load_mnist()
X_train, y_train = X[:50_000], y[:50_000]
X_val, y_val = X[50_000:60_000], y[50_000:60_000]
X_test, y_test = X[60_000:], y[60_000:]  # not used

training_data = prepare_nielsen(X_train, y_train)
test_data = prepare_nielsen(X_test, y_test)
# strictly speaking, MN denotes validation data as test data;
# the `net.evaluate` compares argmax with actual label; hence no one-hot
# encoding.
val_data = prepare_nielsen(
    X_val,
    y_val,
    y_transform=lambda x: x.item(),
    one_hot=False,
)


def test_cross_entropy():
    """test cross entropy"""

    yhat = np.array(
        [
            0.08880715,
            0.11686076,
            0.1296761,
            0.05413173,
            0.08718659,
            0.06143193,
            0.0360487,
            0.24213272,
            0.08954033,
            0.09418397,
        ]
    )
    yi = np.array([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0]], dtype=np.uint8)
    assert np.allclose(
        network.cross_entropy(yi, yhat),
        24.27336582325356,
    )


def test_train_step():
    """test the train step"""

    xi, yi = training_data[0]
    net = network.BasicNetwork(rng=np.random.default_rng(seed=2025))

    loss, dW1, db1, dW2, db2 = net.train_step(xi, yi)

    assert net.predict(xi).argmax().item() == 3

    assert np.allclose(loss, 2.1432136903376966)

    lr = 3.0
    for _ in range(10):
        loss, dW1, db1, dW2, db2 = net.train_step(xi, yi)
        net.W_1 -= lr * dW1
        net.b_1 -= lr * db1
        net.W_2 -= lr * dW2
        net.b_2 -= lr * db2

    yh_c = net.predict(xi).ravel().argmax().item()
    assert yh_c == yi.argmax().item()


def test_predict():
    net = network.BasicNetwork(rng=np.random.default_rng(seed=2025))
    xi, yi = training_data[0]
    yhat = net.predict(xi).ravel()
    assert np.allclose(
        yhat,
        np.array(
            [
                0.10189152,
                0.05425138,
                0.08096501,
                0.18972103,
                0.10339144,
                0.11727734,
                0.12296482,
                0.06127349,
                0.06600587,
                0.10225809,
            ]
        ),
    )


def test_train_update():
    """test the train step"""

    net = network.BasicNetwork(rng=np.random.default_rng(seed=2025))

    for i in range(3):
        print(f"Epoch: {i}")
        net.train_update(training_data, lr=3.0)

    xi, yi = test_data[0]

    yhat = net.predict(xi).ravel()

    assert np.allclose(
        yhat,
        np.array(
            [
                0.08880715,
                0.11686076,
                0.1296761,
                0.05413173,
                0.08718659,
                0.06143193,
                0.0360487,
                0.24213272,
                0.08954033,
                0.09418397,
            ]
        ),
    )

    assert yi.argmax() == yi.argmax()
