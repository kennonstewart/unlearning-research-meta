import argparse
from . import get_rotating_mnist_stream, get_cifar10_stream, get_covtype_stream
from .utils import set_global_seed

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
    for i, (x, y) in zip(range(args.T), gen):
        print(i, hash(x.tobytes()) % 1000, int(y))


if __name__ == "__main__":
    main()
