# Memory Pair Library

This directory contains a minimal implementation of the Memory-Pair online learner used by the experiments.
The algorithm maintains L-BFGS curvature pairs and supports deletion with Gaussian noise calibrated by a privacy odometer.

## L2 Regularization

The learner accepts an optional `lambda_reg` parameter controlling ℓ2 regularization of the weights. It defaults to `0.0`, giving an unregularized objective. For `lambda_reg > 0`, the per-example loss becomes

\[\ell(x, y; w) = \tfrac{1}{2}(w^\top x - y)^2 + \tfrac{\lambda_{\text{reg}}}{2} \lVert w \rVert^2,\]

and gradients include the additional `lambda_reg * w` term. A positive `lambda_reg` makes the objective `lambda_reg`-strongly convex, tightening regret bounds by reducing the optimization term.

## Privacy Odometers

The library now supports **zCDP (zero-Concentrated Differential Privacy)** accounting via the `ZCDPOdometer` class:

```python
from memory_pair.src.odometer import ZCDPOdometer, rho_to_epsilon

# Create zCDP odometer
odometer = ZCDPOdometer(
    rho_total=1.0,      # Total zCDP budget
    delta_total=1e-5,   # Failure probability  
    gamma=0.5,          # Regret constraint
    lambda_=0.1         # Strong convexity
)

# Finalize with calibration statistics
stats = {"G": 2.5, "D": 1.2, "c": 0.8, "C": 1.5}
odometer.finalize_with(stats, T_estimate=1000)

# Perform deletions with per-step sensitivity tracking
for i in range(odometer.deletion_capacity):
    sensitivity = 0.5  # Actual ||d|| for this deletion
    sigma = odometer.noise_scale()
    odometer.spend(sensitivity, sigma)
    
    print(f"Deletion {i+1}: ρ_spent = {odometer.rho_spent:.6f}")

# Convert to (ε, δ) for reporting
epsilon = rho_to_epsilon(odometer.rho_spent, odometer.delta_total)
print(f"Final: ρ = {odometer.rho_spent:.4f} → ε = {epsilon:.4f}")
```

### Key Features

- **Linear budget composition**: zCDP budget adds linearly; we convert to (ε, δ) only once for reporting
- **Per-deletion sensitivity tracking**: Uses actual influence norms rather than global bounds
- **Joint m-σ optimization**: Finds largest deletion capacity with smallest feasible noise scale
- **Adaptive recalibration**: Updates capacity based on observed sensitivity drift

### Backward Compatibility

The legacy `RDPOdometer` is still available but deprecated:

```python
from memory_pair.src.odometer import RDPOdometer  # Issues deprecation warning

# Automatically converts (ε, δ) parameters to zCDP
odometer = RDPOdometer(eps_total=1.0, delta_total=1e-5)  # → ZCDPOdometer
```
