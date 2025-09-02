# Experiment Engine (exp_engine)

A minimal, production-ready experiment engine that writes event-level Parquet with HIVE partitioning, instant DuckDB views, and content-addressed parameter hashing.

## What's included

- Event-level Parquet dataset only (no CSV, no seed-level aggregation)
- HIVE-partitioned Parquet by grid_id and seed for scalable storage
- Instant DuckDB views to query events without loading to memory
- Content-addressed hashing for reproducible grid IDs and deduplication

## Quick Start

```python
from exp_engine.engine import write_event_rows, attach_grid_id
from exp_engine.engine.duck import create_connection_and_views

# 1) Define run parameters and attach a content-addressed grid_id
params = {
    "algo": "memorypair",
    "accountant": "zcdp",
    "gamma_bar": 1.0,
    "gamma_split": 0.5,
    "rho_total": 1.0,
}
params = attach_grid_id(params)

# 2) Write event-level data
event_rows = [
    {
        **params,
        "seed": 1,
        "event_id": 0,
        "op": "insert",
        "regret": 0.037,
        "acc": 0.81,
    },
    # ... more events ...
]
write_event_rows(event_rows, base_out="results_parquet", params=params)

# 3) Query with DuckDB
conn = create_connection_and_views("results_parquet")
df = conn.execute("SELECT * FROM events WHERE regret < 0.05 LIMIT 20").df()
print(df)
```

## Data layout

```
{base_out}/
├── events/       # Event-level data only (HIVE partitioned)
│   └── grid_id=abc12345/
│       └── seed=1/
│           └── data.parquet
└── grids/        # Immutable parameter metadata by grid_id
    └── grid_id=abc12345/
        └── params.json
```

## Partition columns (events only)

- grid_id: content-addressed parameter hash
- seed: random seed of the run

## Command Line Interface

```bash
# Query event-level Parquet via DuckDB
python -m exp_engine.cli query --base-out results_parquet --where "regret < 0.1"

# Show dataset information
python -m exp_engine.cli info --base-out results_parquet
```

## Demo

Run the demo to see end-to-end functionality:

```bash
python exp_engine/demo.py
```

## Notes

- CSV is not supported.
- Seed-level or aggregated outputs are not supported.
- There is no singular master manifest file of all events.

# 5. Query with DuckDB
from exp_engine.engine.duck import create_connection_and_views, query_seeds

conn = create_connection_and_views("results")
df = query_seeds(conn, "avg_regret_empirical < 0.2")
print(df)
```

## Architecture

### Data Organization

The engine writes data to a structured directory layout:

```
{base_out}/
├── seeds/          # Seed-level summaries (HIVE partitioned)
│   └── algo=memorypair/
│       └── accountant=zcdp/
│           └── gamma_bar=1.0/
│               └── grid_id=abc12345/
│                   └── seed=1/
│                       └── data.parquet
├── events/         # Event-level data (HIVE partitioned)  
│   └── algo=memorypair/
│       └── ... (same structure)
└── grids/          # Parameter metadata
    └── grid_id=abc12345/
        └── params.json
```

### Partition Columns

Data is partitioned by:
- `algo`: Algorithm name (e.g., "memorypair")
- `accountant`: Privacy accountant (e.g., "zcdp", "rdp")
- `gamma_bar`: Regret threshold
- `gamma_split`: Split parameter  
- `target_PT`, `target_ST`: Theory-first targets
- `delete_ratio`: Deletion workload ratio
- `rho_total`: Privacy budget
- `grid_id`: Content-addressed parameter hash
- `seed`: Random seed

## Module Reference

### `engine.cah` - Content-Addressed Hashing

```python
from exp_engine.engine.cah import canonicalize_params, grid_hash, attach_grid_id

# Remove volatile keys and normalize floats
canonical = canonicalize_params(params)

# Generate 8-character hex hash
hash_str = grid_hash(params)

# Add grid_id to parameters
params_with_id = attach_grid_id(params)
```

### `engine.io` - Parquet I/O

```python
from exp_engine.engine.io import write_seed_rows, write_event_rows

# Write seed summaries
write_seed_rows(seed_data, base_out, params)

# Write event logs
write_event_rows(event_data, base_out, params)
```

### `engine.duck` - DuckDB Views

```python
from exp_engine.engine.duck import create_connection_and_views, query_seeds, query_events

# Create connection with views
conn = create_connection_and_views(base_out)

# Query seeds
seeds_df = query_seeds(conn, "grid_id = 'abc12345'", limit=100)

# Query events
events_df = query_events(conn, "op = 'insert'")

# Custom SQL
result = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()
```

## Command Line Interface

```bash
# Convert CSV files to Parquet
python -m exp_engine.cli convert experiments/results/ parquet_out/

# Query datasets
python -m exp_engine.cli query parquet_out/ seeds --where "avg_regret_empirical < 0.1"

# Show dataset info
python -m exp_engine.cli info parquet_out/

# Run Snakemake rollup
python -m exp_engine.cli rollup --csv-dir experiments/results/ --base-out parquet_out/
```

## Integration with Existing Runners

### Adding to ExperimentRunner

```python
# In runner.py
from exp_engine.engine import write_seed_rows, write_event_rows, attach_grid_id

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        # Generate grid_id from config
        config_params = self._extract_config_params()
        self.params_with_grid = attach_grid_id(config_params)
    
    def aggregate_and_save(self, summaries, csv_paths):
        # Existing CSV save (unchanged)
        # ... existing code ...
        
        # NEW: Also save as Parquet
        base_out = getattr(self.config, "parquet_out", "results/parquet")
        try:
            write_seed_rows(summaries, base_out, self.params_with_grid)
        except Exception as e:
            print(f"Warning: Parquet save failed: {e}")
```

### Adding to Grid Runner

```python
# In grid_runner.py  
from exp_engine.engine import write_seed_rows

def process_seed_output(csv_files, grid_id, output_dir, mandatory_fields, base_out):
    # Existing processing (unchanged)
    processed_files = []  # ... existing code ...
    
    # NEW: Also create Parquet output
    try:
        seed_data = extract_seed_summaries(csv_files, mandatory_fields)
        write_seed_rows(seed_data, base_out, {"grid_id": grid_id, **mandatory_fields})
    except Exception as e:
        print(f"Warning: Parquet conversion failed: {e}")
    
    return processed_files
```

## Converting Legacy Data

Convert existing CSV sweeps to Parquet format:

```bash
# Convert seed-level summaries
python exp_engine/converter.py experiments/results/ parquet_out/ --granularity seed

# Convert event-level data
python exp_engine/converter.py experiments/results/ parquet_out/ --granularity event

# Dry run to preview
python exp_engine/converter.py experiments/results/ parquet_out/ --dry-run
```

## Snakemake Workflows

Process large experiment sweeps incrementally:

```bash
# Run conversion workflow
snakemake -s exp_engine/Snakefile --configfile exp_engine/snakemake_config.yaml

# Custom configuration
snakemake -s exp_engine/Snakefile --config base_out=my_results csv_dir=my_experiments

# Parallel processing
snakemake -s exp_engine/Snakefile --cores 8
```

## Testing

```bash
# Run all tests
python -m pytest exp_engine/tests/ -v

# Run specific test modules
python -m pytest exp_engine/tests/test_cah.py -v
python -m pytest exp_engine/tests/test_io.py -v
python -m pytest exp_engine/tests/test_duck.py -v
```

## Demo

See the complete functionality in action:

```bash
python exp_engine/demo.py
```

This will demonstrate:
- Content-addressed hashing for deduplication
- Parquet writing with HIVE partitioning  
- DuckDB views for instant querying
- File structure organization

## Dependencies

- `pandas >= 1.3`
- `pyarrow >= 6.0` (for Parquet support)
- `duckdb >= 0.8` (for instant views)
- `snakemake >= 7.0` (for workflows, optional)

Install with:
```bash
pip install pandas pyarrow duckdb snakemake
```

## Design Principles

1. **Minimal changes**: Existing code works unchanged
2. **Content-addressed**: Deduplication via stable parameter hashing
3. **Cloud-native**: Parquet + partitioning for efficient storage/querying
4. **Instant access**: DuckDB views without data loading
5. **Incremental**: Snakemake for processing large sweeps
6. **Production-ready**: Robust error handling and testing