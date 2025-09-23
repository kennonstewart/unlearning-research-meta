# Standardized Analysis Follow-ups for Experiment Notebooks

This document describes the standardized analysis functionality added to experiment notebooks 1-5 to enable automated claim checks and theory validation across all grid experiments.

## Overview

All experiment notebooks now include a new **Section 5.5: Standardized Analysis Follow-ups** that executes five common analyses:

1. **Theory Bound Tracking** - Tests theoretical regret bounds
2. **Stepsize Policy Validation** - Validates stepsize policy adherence  
3. **Privacy & Odometer Sanity Checks** - Checks privacy budget and capacity constraints
4. **Seed Stability Audit** - Identifies high-variability experiments
5. **Enhanced Claim Check Export** - Exports comprehensive analysis results

## Analysis Details

### 1. Theory Bound Tracking
- **Formula**: `theory_ratio = cum_regret / ((G_hat^2/max(lambda_est,λ_min))*(1+ln t) + G_hat*P_T_true)`
- **Analysis**: Computes ratio over last 20% of events
- **Expected**: Ratio ≈ O(1) when theory matches experiment
- **Output**: Median and final values per run

### 2. Stepsize Policy Validation
- **Strong Convexity** (`sc_active=True`): Checks `eta_t*t ≈ 1/max(lambda_est, λ_min)`
- **AdaGrad** (`sc_active=False`): Checks `eta_t ≈ D/√S_t` using `base_eta_t`, `ST_running`
- **Metric**: MAPE (Mean Absolute Percentage Error)
- **Threshold**: 20% for pass/fail classification

### 3. Privacy & Odometer Sanity Checks
- **Capacity**: `m_used ≤ m_capacity`
- **Budget**: `rho_spent ≤ rho_total`
- **Finite Check**: `sigma_step` is finite
- **Noise Consistency**: `cum_regret_with_noise - cum_regret ≈ noise_regret_cum`

### 4. Seed Stability Audit
- **Metrics**: IQR, standard deviation, coefficient of variation
- **Variables**: Final `cum_regret` and `P_T_true` across seeds
- **Flag Threshold**: CV > 0.5 or IQR/mean > 0.5
- **Purpose**: Identify experiments needing replication

### 5. Enhanced Claim Check Export
- **Location**: `artifacts/{notebook_name}_claim_check.json`
- **Content**: Original summary + all analysis results
- **Structure**: Per-run and per-grid statistics
- **Use Case**: CI/CD pipeline consumption

## Usage

The analyses run automatically when executing notebook Section 5.5. No manual intervention required.

### Example Output
```python
=== THEORY BOUND TRACKING ===
Theory ratio - Mean: 1.125, Median: 1.089
Expected: O(1) when theory matches experiment

=== STEPSIZE POLICY VALIDATION ===
Stepsize policy adherence: 14/15 runs passed

=== PRIVACY & ODOMETER SANITY CHECKS ===
Privacy/Odometer checks: 15/15 runs passed

=== SEED STABILITY AUDIT ===
High variability grids: 1/5
Flagged grids: ['grid_003']
```

### Enhanced JSON Export Structure
```json
{
  "notebook": "01_regret_decomposition.ipynb",
  "claim": "R_T ≈ (G^2/λ)·log T + G·P_T with additivity...",
  "summary": [
    {
      "grid_id": "grid_001",
      "seed": 1,
      "final_cum_regret": 156.789,
      "theory_ratio_final": 1.08,
      "stepsize_policy_status": "pass",
      "privacy_odometer_status": "pass",
      "seed_high_variability_flag": false
    }
  ],
  "standardized_analyses": {
    "theory_bound_tracking": {...},
    "stepsize_policy_validation": {...},
    "privacy_odometer_checks": {...},
    "seed_stability_audit": {...}
  }
}
```

## Implementation Files

- **`experiment/utils/standardized_analysis.py`** - Core analysis functions
- **`experiment/tests/test_standardized_analysis.py`** - Unit tests
- **Notebooks 01-05** - Updated with Section 5.5 and enhanced exports

## Testing

Run the test suite to validate functionality:
```bash
cd experiment
python tests/test_standardized_analysis.py
```

## Future Extensions

The modular design allows easy addition of new analyses:
1. Add function to `standardized_analysis.py`
2. Update `run_all_standardized_analyses()` 
3. Modify `enhance_claim_check_export()`
4. Add display logic to Section 5.5 in notebooks