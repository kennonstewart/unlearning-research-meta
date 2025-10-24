# Unified Experiment Runner - Migration Summary

## Overview

The experiment execution framework has been refactored from a fragmented system spread across `grid_runner.py`, `runner.py`, and `exp_engine` into a single, unified script: `experiment/run.py`.

## What Changed

### Before (Deprecated)
```bash
# Grid search - old way
python experiment/grid_runner.py \
  --grid-file grids/my_grid.yaml \
  --base-out results/ \
  --parallel 4

# Single run - old way  
python experiment/runner.py \
  --gamma-bar 1.0 \
  --eps-total 1.0 \
  --max-events 1000
```

### After (Recommended)
```bash
# Grid search - new way
python experiment/run.py \
  --grid-file grids/01_theory_grid.yaml \
  --parallel 4

# Single run - new way
python experiment/run.py \
  --target-G 2.0 --target-D 2.0 \
  --target-c 0.1 --target-C 10.0 \
  --target-lambda 0.5 \
  --target-PT 25.0 --target-ST 50000.0 \
  --rho-total 1.0 --max-events 1000
```

## Key Benefits

1. **Theory-First**: Experiments are defined by fundamental constants (G, D, c, C, λ, P_T, S_T)
2. **No Duplication**: One execution path instead of three separate systems
3. **Parquet-Native**: All output via `exp_engine` with content-addressed IDs
4. **Cleaner Code**: 454 lines vs. 813 + 394 = 1,207 lines (62% reduction)
5. **Better Tests**: Comprehensive test coverage for new functionality

## What Stays the Same

- **Existing tests work**: Old imports redirect to compatibility stubs with deprecation warnings
- **CSV helpers available**: `process_seed_output`, `process_event_output`, etc. still work
- **Algorithm unchanged**: Same `MemoryPair` implementation
- **Data loader unchanged**: Same `get_theory_stream` function

## File Changes

### Created
- `experiment/run.py` - New unified runner
- `experiment/config.py` - Simplified config
- `experiment/grids/01_theory_grid.yaml` - Example grid
- `experiment/legacy_csv_helpers.py` - Extracted helpers

### Replaced (now compatibility stubs)
- `experiment/grid_runner.py` - 29 lines (was 813)
- `experiment/runner.py` - 82 lines (was 394)

### Preserved
- `experiment/grid_runner_old.py` - Original code (reference)
- `experiment/runner_old.py` - Original code (reference)

## Migration Checklist

- [ ] Update your scripts to use `experiment/run.py`
- [ ] Convert grids to theory-first format (see `grids/01_theory_grid.yaml`)
- [ ] Update output directory references (`--output-dir` instead of `--base-out`)
- [ ] Switch from gamma-based to theory-first parameters
- [ ] Update CI/CD pipelines to use new runner
- [ ] Remove references to old runners in documentation

## Questions?

- See `experiment/README.md` for detailed usage guide
- Check `experiment/tests/test_unified_runner.py` for examples
- Run `python experiment/run.py --help` for all options
- Old system still works but shows deprecation warnings

## Timeline

- **Now**: Both systems work (new is recommended, old is deprecated)
- **Future**: Old compatibility stubs may be removed after transition period
- **Action**: Migrate at your convenience, but start planning now
