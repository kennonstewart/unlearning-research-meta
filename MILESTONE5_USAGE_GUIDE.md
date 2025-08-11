# Milestone 5 - Adaptive Capacity Odometer with Alternative Privacy Accounting

## Overview

This implementation provides an adaptive capacity odometer that can switch between different privacy accounting strategies (zCDP, (ε,δ)-DP, and relaxed accounting) to reduce over-conservative noise injection. The solution includes:

1. **Privacy Accountant Strategy Interface** - Abstract base class with concrete implementations
2. **Dynamic Capacity Adjustment** - Recalibration based on observed gradient statistics
3. **Event Logging Extensions** - Enhanced schema to record privacy accounting details
4. **Comprehensive Testing** - Full test suite validating all functionality
5. **Plotting and Analysis Tools** - Utilities for comparing accountant performance

## Quick Start

### Basic Usage

```python
from code.memory_pair.src.adaptive_odometer import AdaptiveCapacityOdometer

# Create an adaptive odometer with zCDP accounting
odometer = AdaptiveCapacityOdometer(
    accountant_type='zCDP',
    budget_params={'rho_total': 2.0, 'delta_total': 1e-5},
    recalibration_enabled=True,
    recalibration_interval=10
)

# Finalize with calibration statistics
stats = {"G": 1.0, "D": 1.0, "c": 1.0, "C": 1.0}
odometer.finalize_with(stats, T_estimate=1000)

# Perform deletions
for i in range(odometer.remaining_capacity()):
    event = odometer.spend(sensitivity=0.5, gradient_magnitude=1.2)
    print(f"Deletion {i+1}: noise_scale={event.noise_scale:.3f}")
```

### Runtime Accountant Selection

```python
# Create different accountant types
accountant_types = ['zCDP', 'eps_delta', 'relaxed']

for acc_type in accountant_types:
    odometer = AdaptiveCapacityOdometer(
        accountant_type=acc_type,
        budget_params={'eps_total': 1.0, 'delta_total': 1e-5}  # Will auto-convert for zCDP
    )
    print(f"Created {acc_type} accountant")
```

### Experimental Comparison

```python
from code.memory_pair.src.experiment_runner import run_comparison_experiment

# Run experiments with matched seeds across all accountant types
results = run_comparison_experiment(
    seeds=[42, 123, 456],
    output_dir="./results"
)
```

## Key Components

### 1. Privacy Accountants (`privacy_accountants.py`)

- **`PrivacyAccountant`** - Abstract base class defining the interface
- **`ZCDPAccountant`** - Zero-concentrated differential privacy accounting
- **`EpsDeltaAccountant`** - Traditional (ε,δ)-DP accounting  
- **`RelaxedAccountant`** - Experimental relaxed accounting with configurable relaxation factor

### 2. Adaptive Odometer (`adaptive_odometer.py`)

- **`AdaptiveCapacityOdometer`** - Main odometer class using strategy pattern
- **`DeletionEvent`** - Event record for comprehensive logging
- Dynamic recalibration every N deletions based on observed statistics
- Integration with pathwise comparator for drift detection

### 3. Event Schema Extensions (`event_schema.py`)

Enhanced schema functions:
- `create_deletion_privacy_info()` - Create privacy information records
- `extract_privacy_metrics()` - Extract metrics for analysis
- Extended `create_event_record()` with privacy_info parameter

### 4. Plotting Utilities (`plotting_utils.py`)

- `plot_regret_comparison()` - Multi-panel comparison plots
- `plot_noise_scale_evolution()` - Noise scale over deletions
- `plot_budget_consumption()` - Budget usage over time
- `create_comprehensive_report()` - Generate complete analysis

### 5. Experiment Runner (`experiment_runner.py`)

- Mock dataset and unlearning algorithm for demonstration
- Configurable experiments with matched seeds
- Automatic results collection and report generation

## Configuration Options

### Accountant-Specific Parameters

**zCDP Accountant:**
```python
budget_params = {'rho_total': 2.0, 'delta_total': 1e-5}
odometer_params = {
    'recalibration_enabled': True,
    'recalibration_interval': 10,
    'drift_threshold': 0.2
}
```

**eps_delta Accountant:**
```python
budget_params = {'eps_total': 1.0, 'delta_total': 1e-5}
odometer_params = {
    'recalibration_enabled': False  # Not supported for eps_delta
}
```

**Relaxed Accountant:**
```python
budget_params = {'eps_total': 1.0, 'delta_total': 1e-5}
odometer_params = {
    'relaxation_factor': 0.5,  # 50% relaxation
    'recalibration_enabled': False
}
```

### Dynamic Recalibration

Recalibration can be triggered by:
1. **Interval-based**: Every N deletions (configurable)
2. **Drift-based**: When pathwise drift exceeds threshold
3. **Manual**: Via explicit `recalibrate_with()` call

## Testing

Run the comprehensive test suite:

```bash
python test_milestone5_adaptive_odometer.py
```

Test categories:
- Privacy accountant interface compliance
- Accountant factory functionality
- Budget conversion utilities
- Basic adaptive odometer functionality
- Dynamic recalibration
- Pathwise comparator integration
- Event schema extensions
- Accountant switching (experimental)
- Metrics export for plotting
- Legacy compatibility
- Relaxed accountant behavior

## Performance Comparison

The implementation allows direct comparison of privacy accounting methods:

| Accountant Type | Budget Tracking | Recalibration | Noise Scale | Use Case |
|-----------------|----------------|---------------|-------------|----------|
| zCDP | ρ (linear composition) | ✅ Supported | Adaptive | Research, tight bounds |
| eps_delta | ε,δ (uniform allocation) | ❌ Not supported | Fixed | Legacy compatibility |
| relaxed | ε,δ (with relaxation) | ❌ Not supported | Reduced | Experimental |

## Integration with Existing Code

The implementation maintains full backward compatibility:

```python
# Existing code continues to work
from code.memory_pair.src.odometer import ZCDPOdometer
odometer = ZCDPOdometer(rho_total=1.0, delta_total=1e-5)

# New code can use adaptive interface
from code.memory_pair.src.adaptive_odometer import create_legacy_odometer
legacy_odometer = create_legacy_odometer(accountant_type='zCDP')
```

## File Structure

```
code/memory_pair/src/
├── adaptive_odometer.py        # Main adaptive odometer implementation
├── privacy_accountants.py      # Privacy accounting strategy interface
├── plotting_utils.py          # Plotting and analysis utilities
├── experiment_runner.py       # Experiment framework
└── odometer.py               # Original odometer (unchanged)

code/data_loader/
└── event_schema.py            # Enhanced event schema (extended)

test_milestone5_adaptive_odometer.py  # Comprehensive test suite
```

## Future Extensions

The modular design enables easy extension:

1. **New Accountant Types**: Implement `PrivacyAccountant` interface
2. **Advanced Recalibration**: Custom recalibration strategies
3. **Integration**: Use with real datasets and unlearning algorithms
4. **Optimization**: Adaptive noise scale selection
5. **Analysis**: Additional plotting and comparison tools

## Example Results

The experiment runner demonstrates clear differences between accountant types:

- **zCDP**: Lower noise scales, adaptive capacity, supports recalibration
- **eps_delta**: Higher noise scales, fixed capacity, uniform budget allocation
- **relaxed**: Intermediate noise scales, experimental relaxation benefits

See generated plots in the experiment output directory for detailed comparisons.