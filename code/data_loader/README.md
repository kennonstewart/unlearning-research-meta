# Data Loader

Utility loaders for Rotating-MNIST, CIFAR-10 and COVTYPE datasets, plus theory-first synthetic linear streams. Each loader falls back to a deterministic simulation when the real dataset cannot be downloaded.

## Available Loaders

| Key      | Function                     | Description |
|----------|------------------------------|-------------|
| rotmnist | `get_rotating_mnist_stream` | Rotating MNIST with configurable drift |
| cifar10  | `get_cifar10_stream`        | CIFAR-10 with optional preprocessing |
| covtype  | `get_covtype_stream`        | Forest cover type classification |
| linear   | `get_synthetic_linear_stream` | Synthetic linear regression (legacy) |
| **theory** | `get_theory_stream`       | **Theory-first synthetic linear regression** |

## Theory-First Data Loader

The new `get_theory_stream()` function provides a theory-driven approach to synthetic data generation, where **theoretical constants are primary inputs** rather than implementation details. This enables experiments parameterized directly by theoretical properties.

### Key Features

- **Theory Constants as Inputs**: Specify gradient bounds (G), domain diameter (D), strong convexity (λ), path length (P_T), AdaGrad energy (S_T), and privacy parameters directly
- **Constraint Enforcement**: Controllers automatically enforce the specified theoretical targets
- **Privacy Integration**: Built-in zCDP and (ε,δ)-DP accounting with per-step noise scheduling
- **Diagnostic Tracking**: Real-time monitoring of constraint adherence and target residuals
- **Backward Compatibility**: Existing linear stream functions remain unchanged

### Example Usage

```python
from code.data_loader import get_theory_stream

# Theory-first configuration
stream = get_theory_stream(
    dim=20,
    T=50000,
    target_G=5.0,           # gradient norm bound
    target_D=4.0,           # parameter domain diameter  
    target_c=1e-3,          # min inverse-Hessian eigenvalue
    target_C=1e3,           # max inverse-Hessian eigenvalue
    target_lambda=1e-2,     # strong convexity parameter
    target_PT=250.0,        # total path length over horizon
    target_ST=2.0e6,        # cumulative squared gradients
    accountant="zcdp",      # privacy accounting method
    rho_total=1.0,          # total zCDP budget
    path_style="rotating",  # parameter path type
    seed=42
)

# Process events with theory constraints enforced
for event in stream:
    x, y = event["x"], event["y"]
    metrics = event["metrics"]
    
    # Access theory targets and diagnostics
    G_hat = metrics["G_hat"]                    # gradient bound for LBFGS
    PT_residual = metrics["PT_target_residual"] # path length tracking error
    clip_applied = metrics["clip_applied"]      # gradient clipping indicator
    privacy_spend = metrics["privacy_spend_running"]
    
    # Your learning algorithm here...
```

### Configuration Example

See `theory_config_example.yaml` for complete configuration examples.

### Demonstration

Run the theory stream demonstration:

```bash
python code/data_loader/demo_theory_stream.py
```

This shows real-time constraint enforcement and generates diagnostic plots.

## Grid Search Integration

Grid cells are now specified in **theory space** $(G,D,c,C,\lambda,P_T,S_T,\text{accountant})$ and the stream **enforces** those targets. This directly supports Experiments A–E (bound adherence, comparator split, sensitivity to $G,C$, replenishment via $S_T$).

## Legacy Usage

Run `python sanity_check.py` to verify deterministic output for existing loaders:

```bash
python sanity_check.py --dataset rotmnist --T 10
```