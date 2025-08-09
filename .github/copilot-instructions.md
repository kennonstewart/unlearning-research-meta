# Machine Unlearning Research Meta-Repository

**ALWAYS follow these instructions first and fallback to additional search and context gathering ONLY if the information in these instructions is incomplete or found to be in error.**

## Repository Overview

This is a meta-repository for machine unlearning research containing:
- `code/` - Canonical, installable source code (memory_pair, data_loader, baselines)
- `experiments/` - Independent experimental studies with specific AGENTS.md instructions
- Root-level tests for validation and integration testing

Each experiment has its own AGENTS.md file with detailed instructions for that specific study.

## Environment Setup and Build Process

### 1. Bootstrap Environment
```bash
cd /path/to/unlearning-research-meta
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
**Timing**: ~5 seconds

### 2. Install Dependencies
```bash
pip install -r .devcontainer/requirements.txt
```
**Timing**: ~30 seconds. NEVER CANCEL - includes large ML packages (torch, transformers). Set timeout to 120+ seconds.

### 3. **CRITICAL**: Module Import Method
**DO NOT** use editable installs (`pip install -e code/memory_pair`) - they fail due to PyPI network timeouts.

**ALWAYS use one of these working methods**:

**Method A (Recommended)**: PYTHONPATH
```bash
PYTHONPATH=code python your_script.py
```

**Method B**: sys.path in Python code
```python
import sys
sys.path.insert(0, 'code')
from memory_pair.src.memory_pair import MemoryPair
from data_loader import get_rotating_mnist_stream
```

## Testing and Validation

### Run All Tests
```bash
pytest test_minimal_experiment.py test_schema.py test_integration.py test_flags_default_noop.py -v
```
**Timing**: ~2.5 seconds. 18 tests should pass with 1 warning.

**Note**: Some tests (test_zcdp_odometer.py) have import issues and should be skipped for now.

### Data Loader Validation
**Manual validation (recommended)**:
```bash
python -c "
import sys; sys.path.insert(0, 'code')
from data_loader import get_rotating_mnist_stream
gen = get_rotating_mnist_stream(mode='drift', seed=42)
for i in range(5):
    sample = next(gen)
    x, y = sample['x'], sample['y']
    print(f'Sample {i}: hash {hash(x.tobytes()) % 1000}, y = {int(y)}')
print('Data loader working correctly')
"
```

**Note**: Data loader returns dict format `{'x': array, 'y': int, 'sample_id': str, ...}`, not tuple (x, y).

### Validate MemoryPair Algorithm
```bash
python -c "
import sys; sys.path.insert(0, 'code')
from memory_pair.src.memory_pair import MemoryPair
from data_loader import get_rotating_mnist_stream
print('MemoryPair validation passed')
"
```

## Core Algorithm Usage

### MemoryPair Workflow
```python
import sys; sys.path.insert(0, 'code')
import numpy as np
from memory_pair.src.memory_pair import MemoryPair
from data_loader import get_rotating_mnist_stream

# Initialize
gen = get_rotating_mnist_stream(mode='drift', seed=42)
sample = next(gen)
dim = sample['x'].shape[0]
mp = MemoryPair(dim=dim)

# 1. Calibration Phase
for i in range(50):
    sample = next(gen)
    x, y = sample['x'].astype(np.float32), float(sample['y'])
    mp.calibrate_step(x, y)

# 2. Finalize Calibration (required before insert/delete)
mp.finalize_calibration(gamma=0.5)

# 3. Learning Phase - Insert operations
for i in range(20):
    sample = next(gen)
    x, y = sample['x'].astype(np.float32), float(sample['y'])
    mp.insert(x, y)

# 4. Delete operations (only in INTERLEAVING phase)
# Note: Deletions require transition to INTERLEAVING phase first
```

### Data Loader Format
The data loader returns dictionaries, not tuples:
```python
sample = next(data_stream)
x = sample['x']           # Feature vector
y = sample['y']           # Label
sample_id = sample['sample_id']  # Unique identifier
# Additional metadata: event_id, segment_id, metrics
```

## Experiment Workflows

### Deletion Capacity Experiment
```bash
cd experiments/deletion_capacity
PYTHONPATH=../../code python run.py --help
```
**Note**: Experiment expects tuple format (x, y) but data loader returns dict. Adapt accordingly.

### Sublinear Regret Experiment  
```bash
cd experiments/sublinear_regret
PYTHONPATH=../../code python run.py --dataset rotmnist --stream drift --algo memorypair --T 1000
```

### Post-Deletion Accuracy Experiment
```bash
cd experiments/post_deletion_accuracy
PYTHONPATH=../../code python run_accuracy.py --help
```
**Note**: Uses `MemoryPair` class, not `StreamNewtonMemoryPair`.

## Known Issues and Workarounds

### 1. Editable Install Failures
**Issue**: `pip install -e code/memory_pair` fails with ReadTimeoutError from PyPI
**Solution**: Use PYTHONPATH method shown above

### 2. Network/PyPI Limitations
**Issue**: Fresh dependency installation may fail due to network timeouts
**Symptoms**: `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out`
**Solution**: 
- If on a fresh system, retry installation multiple times
- Consider using conda or alternative package management
- For development, copy a working .venv from another system if available

### 3. Data Format Mismatch
**Issue**: Experiments expect `(x, y)` tuples but data loader returns dicts
**Solution**: Adapt experiment code:
```python
# Instead of: x, y = next(gen)
sample = next(gen)
x, y = sample['x'], sample['y']
```

### 4. Import Errors in Experiments
**Issue**: `ModuleNotFoundError: No module named 'code.memory_pair'`
**Solution**: Use PYTHONPATH or add to experiments:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
```

### 5. Test Suite Limitations
**Issue**: Some tests (test_zcdp_odometer.py) have import issues
**Solution**: Run working subset only:
```bash
pytest test_minimal_experiment.py test_schema.py test_integration.py test_flags_default_noop.py -v
```

## Development Guidelines

### Import Policy
Always use canonical imports:
```python
# Algorithms
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.odometer import PrivacyOdometer

# Data
from data_loader import get_rotating_mnist_stream, get_cifar10_stream, get_covtype_stream
```

### Timing Expectations and Warnings

**NEVER CANCEL long-running operations. Set appropriate timeouts.**

| Operation | Expected Time | Timeout Setting | Notes |
|-----------|---------------|-----------------|-------|
| Virtual environment creation | 3-5 seconds | 30 seconds | `python3 -m venv .venv` |
| Dependency installation | 30 seconds | 180+ seconds | **NEVER CANCEL** - includes torch, transformers |
| Test suite (working subset) | 2.5 seconds | 60 seconds | 18 tests should pass |
| Data loader validation | 1 second | 30 seconds | Manual validation script |
| MemoryPair workflow test | 1.2 seconds | 60 seconds | Calibration + insert operations |
| Single experiment run | 5-30 seconds | 300+ seconds | **NEVER CANCEL** - depends on dataset size |

**CRITICAL**: Dependency installation can take 30+ seconds due to large ML packages (torch, transformers, datasets). Network timeouts are common - always set timeouts to 180+ seconds and never cancel prematurely.

### Reproducibility
All scripts must accept `--seed` parameter and use:
```python
from code.data_loader.utils import set_global_seed
set_global_seed(args.seed)
```

## Experiment-Specific Instructions

Each experiment directory contains its own `AGENTS.md` with:
- Specific CLI commands and parameters
- Expected output formats and commit patterns
- Grid search configurations (for deletion_capacity)
- Dataset and algorithm combinations

**ALWAYS** read the experiment's `AGENTS.md` before making changes to experiment code.

## Validation Checklist

Before making changes, ALWAYS run:

1. **Environment validation**:
   ```bash
   source .venv/bin/activate
   python -c "import sys; sys.path.insert(0, 'code'); from memory_pair.src.memory_pair import MemoryPair; print('OK')"
   ```

2. **Test suite**:
   ```bash
   pytest test_minimal_experiment.py test_schema.py test_integration.py test_flags_default_noop.py -v  # Should show 18 passed, 1 warning
   ```

3. **Data loader**:
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'code')
   from data_loader import get_rotating_mnist_stream
   gen = get_rotating_mnist_stream(mode='iid', seed=42)
   sample = next(gen)
   print(f'Data loader OK: x.shape={sample[\"x\"].shape}, y={sample[\"y\"]}')
   "
   ```

4. **End-to-end workflow**:
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'code')
   import numpy as np
   from memory_pair.src.memory_pair import MemoryPair
   from data_loader import get_rotating_mnist_stream
   
   # Complete workflow test
   gen = get_rotating_mnist_stream(mode='drift', seed=42)
   sample = next(gen)
   dim = sample['x'].shape[0]
   mp = MemoryPair(dim=dim)
   
   # Calibration
   for i in range(10):
       sample = next(gen)
       x, y = sample['x'].astype(np.float32), float(sample['y'])
       mp.calibrate_step(x, y)
   
   # Finalize and test learning
   mp.finalize_calibration(gamma=0.5)
   for i in range(5):
       sample = next(gen)
       x, y = sample['x'].astype(np.float32), float(sample['y'])
       mp.insert(x, y)
   
   print('Complete workflow validation: OK')
   "
   ```
   **Timing**: ~1.2 seconds

## Common Tasks Reference

### Repository Structure
```
.
├── README.md
├── AGENTS.md                    # Meta-repository conventions
├── code/                        # Canonical source code
│   ├── memory_pair/            # Memory-Pair algorithm
│   ├── data_loader/            # Unified dataset loaders  
│   └── baselines/              # Baseline implementations
├── experiments/                # Independent studies
│   ├── deletion_capacity/      # Deletion capacity analysis
│   ├── sublinear_regret/       # Regret analysis
│   └── post_deletion_accuracy/ # Accuracy degradation
└── test_*.py                   # Root-level integration tests
```

### Environment Variables
```bash
export PYTHONPATH=code          # For module imports
```

### Memory-Pair Algorithm Phases
1. **CALIBRATION**: Use `calibrate_step(x, y)`
2. **LEARNING**: Use `insert(x, y)` after `finalize_calibration(gamma)`  
3. **INTERLEAVING**: Deletions allowed with `delete(x, y)`

### Data Loader Modes
- `iid`: Independent and identically distributed
- `drift`: Concept drift every 1000 steps
- `adv`: Adversarial permutations every 500 steps

Always follow these instructions and refer to experiment-specific `AGENTS.md` files for detailed workflows.