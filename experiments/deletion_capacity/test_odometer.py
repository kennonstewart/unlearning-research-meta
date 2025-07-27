import numpy as np
from memory_pair.src.odometer import PrivacyOdometer


def test_odometer():
    np.random.seed(42)
    dim = 10
    warmup_iters = 100

    # Simulate a warmup sequence
    odometer = PrivacyOdometer(
        eps_total=1.0,
        delta_total=1e-5,
        T=10000,
        gamma=0.05,
        lambda_=0.1,
        delta_B=0.05,
    )

    print("\n[Testing] Simulating warmup...")
    theta = np.zeros(dim)
    for _ in range(warmup_iters):
        grad = np.random.normal(0, 1, size=dim)
        step = np.random.normal(0, 0.1, size=dim)
        theta += step
        odometer.observe(grad, theta)

    print("[Testing] Finalizing odometer...")
    odometer.finalize()

    # Simulate deletion phase
    print("\n[Testing] Performing deletions...")
    for i in range(odometer.deletion_capacity):
        try:
            odometer.spend()
            remaining_eps = odometer.remaining()
            print(
                f"  Deletion {i + 1:02d}: ε_spent = {odometer.eps_spent:.4f}, ε_remaining = {remaining_eps:.4f}"
            )
        except RuntimeError as e:
            print(f"  Error: {str(e)}")
            break

    # Attempt one extra deletion to trigger capacity error
    print("\n[Testing] Testing deletion beyond capacity...")
    try:
        odometer.spend()
    except RuntimeError as e:
        print(f"  ✅ Correctly blocked: {str(e)}")

    print("\n[Testing] Noise scale: σ =", odometer.noise_scale())


if __name__ == "__main__":
    test_odometer()
