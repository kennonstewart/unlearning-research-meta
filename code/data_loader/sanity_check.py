import argparse
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import get_rotating_linear_stream, get_theory_stream

MAP = {
    "linear": get_rotating_linear_stream,
    "theory": get_theory_stream,
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["linear", "theory"])
parser.add_argument("--T", type=int, default=10)


def main():
    args = parser.parse_args()
    # Synthetic streams use different API - no mode parameter
    if args.dataset == "linear":
        gen = MAP[args.dataset](seed=42)
    elif args.dataset == "theory":
        # Theory stream requires more parameters - use minimal set
        gen = MAP[args.dataset](
            dim=10,
            T=args.T,
            target_G=2.0,
            target_D=1.0,
            accountant="zcdp",
            rho_total=1.0,
            delta_total=1e-5,
            seed=42,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    for i in range(args.T):
        event = next(gen)
        # Consume unified event record format
        x, y = event["x"], event["y"]
        print(i, hash(x.tobytes()) % 1000, int(y))


if __name__ == "__main__":
    main()
