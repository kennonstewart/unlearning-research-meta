import numpy as np

class OnlineSGD:
    def __init__(self, dim, lr=0.1):
        self.theta = np.zeros(dim)
        self.lr = lr

    def step(self, x, y):
        pred = self.theta @ x
        grad = (pred - y) * x
        self.theta -= self.lr * grad
        return 0.5 * (pred - y) ** 2

class AdaGrad:
    def __init__(self, dim, lr=1.0, eps=1e-8):
        self.theta = np.zeros(dim)
        self.lr = lr
        self.eps = eps
        self.G = np.zeros(dim)

    def step(self, x, y):
        pred = self.theta @ x
        grad = (pred - y) * x
        self.G += grad ** 2
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.eps)
        self.theta -= adjusted_lr * grad
        return 0.5 * (pred - y) ** 2

class OnlineNewtonStep:
    def __init__(self, dim, lam=1.0):
        self.theta = np.zeros(dim)
        self.H = lam * np.eye(dim)

    def step(self, x, y):
        pred = self.theta @ x
        grad = (pred - y) * x
        self.H += np.outer(x, x)
        self.theta -= np.linalg.inv(self.H) @ grad
        return 0.5 * (pred - y) ** 2
