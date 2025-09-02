# Finalized Regret Calculation and Reporting Specification

## Overview

This document describes the finalized regret calculation and reporting specification for deletion capacity experiments. The specification ensures consistent, non-negative, comparator-based regret computation with proper privacy metrics logging and Parquet-native data workflows.

## Key Requirements

### 1. Loss Function and Comparator

**Loss Function**: Squared loss with L2 regularization (ridge regression)
```
loss_t = 0.5 * (y_t - ŷ_t)² + 0.5 * λ * ||θ_t||²
```

**Comparator**: Uses the same λ regularization parameter for fair comparison
- **Static Oracle**: Fixed w₀* computed from initial calibration data
- **Rolling Oracle**: Dynamic w_t* recomputed on sliding window
- Both comparators use identical regularized loss for regret computation

**Regret Definition**: 
```
regret_t = loss_t(θ_t, x_t, y_t) - loss_t(w*, x_t, y_t)
```
where w* is the appropriate comparator (static or dynamic).

### 2. Non-negative Regret Enforcement

**Requirement**: Compute non-negative regret on insert events post-warmup

**Implementation**: 
- Controlled by `enforce_nonnegative_regret` config flag (default: True)
- Applied in comparator `update_regret_accounting` methods:
  ```python
  if cfg.enforce_nonnegative_regret:
      regret_increment = max(0.0, regret_increment)
  ```

**Rationale**: Negative regret can occur due to:
- Oracle approximation errors
- Finite window effects in rolling oracle
- Regularization differences
- Numerical precision issues

Clamping to non-negative ensures meaningful privacy accounting.

### 3. Privacy Metrics on Delete Events

**Requirement**: Log odometer-based privacy metrics on delete events

**Implementation**: 
- Privacy metrics extracted via `get_privacy_metrics(model)` in `phases.py`
- Includes: `eps_spent`, `rho_spent`, `sigma_step`, `m_capacity`, etc.
- Logged for all delete operations in event logs

**Key Metrics**:
- **zCDP**: `rho_spent`, `sigma_step`, `m_capacity`
- **RDP**: `eps_converted`, `delta_total`, `sens_q95`
- **Common**: `eps_spent`, `delta_spent`, `recalibrations_count`

### 4. Parquet-Only Mode by Default

**Requirement**: Write Parquet-only by default, skip legacy CSV

**Implementation**:
- `parquet_only_mode: bool = True` in config defaults
- Controlled by CLI flags: `--parquet-out`, `--no-legacy-csv`
- DuckDB-based aggregation replaces CSV concatenation

**Benefits**:
- Better compression and query performance
- Native columnar analytics
- Type safety and schema evolution
- Integrated with DuckDB for analysis

### 5. DuckDB Views and Analysis

**Core Views**:
- `seeds`: Raw seed summary data
- `events`: Raw event-level data
- `events_summary`: Aggregated metrics per (grid_id, seed)
- `regret_analysis`: Detailed regret statistics by operation type

**Regret-Specific Views**:
```sql
-- Regret analysis view
CREATE VIEW regret_analysis AS
SELECT
    grid_id, seed, op,
    COUNT(*) as event_count,
    AVG(regret) as avg_regret,
    STDDEV(regret) as std_regret,
    MIN(regret) as min_regret,
    MAX(regret) as max_regret,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY regret) as median_regret,
    SUM(CASE WHEN regret < 0 THEN 1 ELSE 0 END) as negative_count,
    AVG(CASE WHEN regret >= 0 THEN regret END) as avg_nonneg_regret
FROM events
WHERE regret IS NOT NULL
GROUP BY grid_id, seed, op;
```

**Helper Functions**:
- `query_regret_analysis()`: Query regret statistics
- `get_negative_regret_summary()`: Summarize negative regret occurrences

## Configuration Keys

### Core Regret Settings
```python
# Finalized regret calculation spec
enforce_nonnegative_regret: bool = True    # Ensure regret >= 0
parquet_only_mode: bool = True             # Write Parquet-only by default
regret_warmup_threshold: Optional[int] = None    # Events before regret tracking
regret_comparator_mode: str = "oracle"     # "oracle", "zero", or "ridge"
```

### Existing Related Settings
```python
# Comparator configuration
comparator: str = "dynamic"                # "static" or "dynamic"
enable_oracle: bool = False               # Enable oracle/comparator functionality

# Regularization
lambda_reg: float = 0.0                   # L2 regularization parameter
```

## Usage Examples

### 1. Basic Experiment with Finalized Spec
```bash
python runner.py \
    --enforce-nonnegative-regret \
    --parquet-only-mode \
    --enable-oracle \
    --comparator dynamic \
    --lambda-reg 0.1
```

### 2. Grid Search with Parquet Output
```bash
python agents/grid_runner.py \
    --grid-file grids.yaml \
    --parquet-out results_parquet \
    --parquet-write-events \
    --no-legacy-csv
```

### 3. Analysis with DuckDB
```python
from exp_engine.engine.duck import create_connection_and_views, query_regret_analysis

# Create connection and views
conn = create_connection_and_views("results_parquet")

# Analyze regret patterns
regret_df = query_regret_analysis(conn, "grid_id = 'my_experiment'")
print(regret_df.groupby('op')['avg_regret'].mean())

# Check for negative regret issues
neg_summary = get_negative_regret_summary(conn)
print(f"Experiments with negative regret: {len(neg_summary)}")
```

## Validation and Testing

### Automated Tests
- `test_finalized_regret_spec.py`: Comprehensive test suite
- Tests non-negative enforcement, regularization consistency, config defaults
- Validates DuckDB views and privacy metrics extraction

### Key Test Cases
1. **Non-negative Enforcement**: Verify regret ≥ 0 when enabled
2. **Regularization Consistency**: Same λ used in all loss computations  
3. **Privacy Metrics**: Odometer metrics logged on delete events
4. **DuckDB Views**: Regret analysis views work correctly
5. **Configuration**: All spec config keys available with correct defaults

### Validation Checklist
- [ ] Regret computation uses identical λ for current and comparator losses
- [ ] Non-negative regret enforced when `enforce_nonnegative_regret=True`
- [ ] Privacy metrics extracted and logged on delete events
- [ ] Parquet-only mode enabled by default
- [ ] DuckDB regret analysis views functional
- [ ] All tests in `test_finalized_regret_spec.py` pass

## Migration Guide

### From Legacy Regret Calculation
1. **Enable finalized spec**: Set `enforce_nonnegative_regret=True`
2. **Switch to Parquet**: Use `--parquet-out` and `--no-legacy-csv`
3. **Update analysis code**: Use DuckDB views instead of CSV concatenation
4. **Verify regularization**: Ensure consistent λ across all components

### Backward Compatibility
- Legacy CSV output still available with `--legacy-csv`
- Non-negative enforcement can be disabled with `enforce_nonnegative_regret=False`
- Existing analysis code works with DuckDB views (same schema)

## Performance Considerations

### Parquet Benefits
- 3-10x smaller file sizes compared to CSV
- Faster query performance with columnar storage
- Better compression with typed data

### DuckDB Advantages
- In-memory analytics without external database
- SQL interface for complex aggregations
- Seamless integration with pandas/numpy workflows

### Scalability
- Partitioned Parquet files enable parallel processing
- DuckDB handles datasets larger than memory
- Views provide lazy evaluation for large-scale analysis

## Troubleshooting

### Common Issues

**Negative Regret**: 
- Check `enforce_nonnegative_regret` setting
- Verify oracle calibration quality
- Review regularization parameters

**Missing Privacy Metrics**:
- Ensure odometer implements `metrics()` method
- Check accountant finalization status
- Verify delete events are properly logged

**DuckDB View Errors**:
- Install DuckDB: `pip install duckdb`
- Check Parquet file structure and schema
- Verify required columns exist in data

### Debug Commands
```bash
# Check regret distribution
python -c "
from exp_engine.engine.duck import *
conn = create_connection_and_views('results_parquet')
df = query_regret_analysis(conn, 'op = \"insert\"')
print(df[['avg_regret', 'negative_count']].describe())
"

# Validate privacy metrics
python -c "
from exp_engine.engine.duck import *
conn = create_connection_and_views('results_parquet')
df = conn.execute('SELECT op, COUNT(*) FROM events WHERE rho_spent IS NOT NULL GROUP BY op').df()
print(df)
"
```

This specification ensures robust, reproducible regret calculation with comprehensive privacy accounting and modern data infrastructure.