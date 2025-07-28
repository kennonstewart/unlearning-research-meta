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

    def insert(self, x, y):
        pred = self.theta @ x
        grad = (pred - y) * x
        self.G += grad**2
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


class SekhariBatchUnlearning:
    def __init__(self, dim, lam=1.0):
        self.dim = dim
        self.lam = lam
        self.theta = np.zeros(dim)
        self.H_total = None
        self.X_train = None
        self.y_train = None

    def insert(self, X, y):
        self.X_train = X
        self.y_train = y

        # Batch training to find the empirical risk minimizer
        A = self.X_train.T @ self.X_train + self.lam * np.eye(self.dim)
        b = self.X_train.T @ self.y_train
        self.theta = np.linalg.solve(A, b)

        # Pre-compute the Hessian at the learned model for unlearning
        self.H_total = self.X_train.T @ self.X_train

    def delete(self, delete_indices):
        if self.H_total is None:
            raise ValueError("The model must be trained before deletion.")

        X_delete = self.X_train[delete_indices]
        y_delete = self.y_train[delete_indices]

        m = len(X_delete)
        n = len(self.X_train)

        # Compute Hessian of the points to be deleted
        H_delete = X_delete.T @ X_delete

        # Compute the Hessian of the remaining data
        H_remaining = self.H_total - H_delete

        # Compute the gradient of the loss for the deleted points at the current theta
        grad_delete = np.zeros(self.dim)
        for i in range(m):
            pred = self.theta @ X_delete[i]
            grad_delete += (pred - y_delete[i]) * X_delete[i]

        # Apply the Newton step for unlearning
        n_remaining = n - m
        if n_remaining > 0:
            if np.linalg.det(H_remaining) == 0:
                H_remaining_inv = np.linalg.pinv(H_remaining)
            else:
                H_remaining_inv = np.linalg.inv(H_remaining)

            self.theta += (H_remaining_inv @ grad_delete) / n_remaining


class QiaoHessianFree:
    def __init__(self, dim, lr=0.01, epochs=10):
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.theta = np.zeros(dim)
        self.approximators = {}
        self.training_history = []
        self.X_train = None
        self.y_train = None

    def insert(self, X, y):
        self.X_train = X
        self.y_train = y

        # Standard SGD training, storing the trajectory
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i, y_i = X[i], y[i]
                pred = self.theta @ x_i
                grad = (pred - y_i) * x_i
                self.theta -= self.lr * grad
                self.training_history.append(self.theta.copy())

        # Pre-compute approximators for each data point
        H = self.X_train.T @ self.X_train
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        for i in range(len(X)):
            x_i, y_i = X[i], y[i]
            grad_i = (self.theta @ x_i - y_i) * x_i
            self.approximators[i] = H_inv @ grad_i

    def delete(self, delete_indices):
        if not self.approximators:
            raise ValueError(
                "Model must be trained and approximators computed before deletion."
            )

        # Unlearning by adding the pre-computed approximators
        for i in delete_indices:
            if i in self.approximators:
                self.theta += self.approximators[i]


__all__ = [
    "OnlineSGD",
    "AdaGrad",
    "OnlineNewtonStep",
    "SekhariBatchUnlearning",
    "QiaoHessianFree",
]
