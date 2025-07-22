from .mnist import get_rotating_mnist_stream
from .cifar10 import get_cifar10_stream
from .covtype import get_covtype_stream
from .streams import make_stream
from .linear import get_synthetic_linear_stream

__all__ = [
    "get_rotating_mnist_stream",
    "get_cifar10_stream",
    "get_covtype_stream",
    "get_synthetic_linear_stream",
    "make_stream",
]
