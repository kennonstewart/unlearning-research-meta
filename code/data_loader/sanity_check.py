import argparse
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import get_rotating_mnist_stream, get_cifar10_stream, get_covtype_stream
from data_loader.utils import set_global_seed

MAP = {
    "rotmnist": get_rotating_mnist_stream,
    "cifar10": get_cifar10_stream,
    "covtype": get_covtype_stream,
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(MAP.keys()))
parser.add_argument("--mode", default="iid", choices=["iid", "drift", "adv"])
parser.add_argument("--T", type=int, default=10)


def main():
    args = parser.parse_args()
    gen = MAP[args.dataset](mode=args.mode)
    for i in range(args.T):
        event = next(gen)
        # Handle both legacy (x, y) tuples and new event record format
        if isinstance(event, dict):
            x, y = event['x'], event['y']
        else:
            x, y = event
        print(i, hash(x.tobytes()) % 1000, int(y))


if __name__ == "__main__":
    main()
