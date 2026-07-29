import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from tensorflow.keras.datasets import mnist

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0

def one_hot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

y_train = one_hot(y_train)


def adam_update(param, grad, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * (grad ** 2)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v

class Dense:
    def __init__(self, in_dim, out_dim):
        # He initialization
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros((1, out_dim))
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dz):
        self.dW = self.x.T @ dz
        self.db = np.sum(dz, axis=0, keepdims=True)
        return dz @ self.W.T

    def step(self, t, lr):
        self.W, self.mW, self.vW = adam_update(self.W, self.dW, self.mW, self.vW, t, lr)
        self.b, self.mb, self.vb = adam_update(self.b, self.db, self.mb, self.vb, t, lr)

class BatchNorm:
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, dim))
        self.beta  = np.zeros((1, dim))
        self.run_mean = np.zeros((1, dim))
        self.run_var  = np.ones((1, dim))
        self.momentum = momentum
        self.eps = eps
        self.mg, self.vg = np.zeros_like(self.gamma), np.zeros_like(self.gamma)
        self.mb, self.vb = np.zeros_like(self.beta), np.zeros_like(self.beta)

    def forward(self, z, training=True):
        if training:
            mu = z.mean(axis=0, keepdims=True)
            var = z.var(axis=0, keepdims=True)
            self.std = np.sqrt(var + self.eps)
            self.z_norm = (z - mu) / self.std
            out = self.gamma * self.z_norm + self.beta

            self.run_mean = self.momentum * self.run_mean + (1 - self.momentum) * mu
            self.run_var  = self.momentum * self.run_var  + (1 - self.momentum) * var
        else:
            z_norm = (z - self.run_mean) / np.sqrt(self.run_var + self.eps)
            out = self.gamma * z_norm + self.beta
        return out

    def backward(self, dout):
        N = dout.shape[0]
        self.dgamma = np.sum(dout * self.z_norm, axis=0, keepdims=True)
        self.dbeta  = np.sum(dout, axis=0, keepdims=True)

        dz_norm = dout * self.gamma
        dz = (1. / N) * (1. / self.std) * (
            N * dz_norm
            - np.sum(dz_norm, axis=0, keepdims=True)
            - self.z_norm * np.sum(dz_norm * self.z_norm, axis=0, keepdims=True)
        )
        return dz

    def step(self, t, lr):
        self.gamma, self.mg, self.vg = adam_update(self.gamma, self.dgamma, self.mg, self.vg, t, lr)
        self.beta,  self.mb, self.vb = adam_update(self.beta,  self.dbeta,  self.mb, self.vb, t, lr)

class ReLU:
    def forward(self, x, training=True):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, dz):
        return dz * self.mask

    def step(self, t, lr): pass

class Dropout:
    def __init__(self, p=0.2):
        self.p = p

    def forward(self, x, training=True):
        if training and self.p > 0:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, dz):
        return dz * self.mask

    def step(self, t, lr): pass


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y, pred):
    return -np.mean(np.sum(y * np.log(pred + 1e-9), axis=1))


# Model

np.random.seed(42)

layers = [
    Dense(784, 256), BatchNorm(256), ReLU(), Dropout(0.2),
    Dense(256, 128), BatchNorm(128), ReLU(), Dropout(0.2),
    Dense(128, 64),  BatchNorm(64),  ReLU(), Dropout(0.2),
    Dense(64, 10)
]

def forward(x, training=True):
    for layer in layers:
        x = layer.forward(x, training)
    return x

def backward(dout):
    for layer in reversed(layers):
        dout = layer.backward(dout)

#Training

epochs = 25
batch_size = 64
lr = 0.001
t = 0

n_samples = len(x_train)
n_batches = int(np.ceil(n_samples / batch_size))

for epoch in range(epochs):
    # Shuffle
    idx = np.random.permutation(n_samples)
    x_sh, y_sh = x_train[idx], y_train[idx]

    epoch_loss = 0.0

    for i in range(n_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, n_samples)
        xb, yb = x_sh[start:end], y_sh[start:end]

        logits = forward(xb, training=True)
        probs = softmax(logits)

        epoch_loss += cross_entropy(yb, probs) * len(xb)

        # Softmax + Cross-Entropy gradient
        dz = (probs - yb) / len(xb)
        backward(dz)

        t += 1
        for layer in layers:
            layer.step(t, lr)

    epoch_loss /= n_samples

    # Eval
    test_logits = forward(x_test, training=False)
    test_preds = np.argmax(softmax(test_logits), axis=1)
    acc = np.mean(test_preds == y_test)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Test Acc: {acc:.4f}")

# Final stats
print(f"\nFinal Accuracy: {acc*100:.2f}%")