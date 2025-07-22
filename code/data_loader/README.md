# Data Loader

Utility loaders for Rotating-MNIST, CIFAR-10 and COVTYPE datasets. Each loader falls back to a deterministic simulation when the real dataset cannot be downloaded.

## Available Loaders

| Key      | Function                     |
|----------|-----------------------------|
| rotmnist | `get_rotating_mnist_stream` |
| cifar10  | `get_cifar10_stream`        |
| covtype  | `get_covtype_stream`        |

Run `python sanity_check.py` to verify deterministic output.
