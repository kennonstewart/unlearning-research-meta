import pytest
import tempfile
import os
import pandas as pd
from exp_engine.converter import convert_csv_to_parquet


def parse_csv_filename(filename):
    """
    Extract parameters from a CSV filename of the form:
    'seed_001_memorypair_gamma1.0-split0.5_zcdp.csv'
    Returns a dict with keys: seed, algo, gamma, split, accountant.
    """
    import re
    # Example: seed_001_memorypair_gamma1.0-split0.5_zcdp.csv
    pattern = r"seed_(\d+)_(\w+)(?:_gamma([\d\.]+)-split([\d\.]+))?_(\w+)\.csv"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    seed = int(match.group(1))
    algo = match.group(2)
    gamma = float(match.group(3)) if match.group(3) else None
    split = float(match.group(4)) if match.group(4) else None
    accountant = match.group(5)
    result = {"seed": seed, "algo": algo}
    if gamma is not None:
        result["gamma"] = gamma
    if split is not None:
        result["split"] = split
    result["accountant"] = accountant
    return result

def extract_params_from_grid_dir(grid_dir):
    """
    Extract parameters from a grid directory name of the form:
    'split_0.7-0.3_q0.95_k10_default'
    Returns a dict with keys: gamma_bar, gamma_split, quantile, delete_ratio, accountant.
    """
    import os
    import re
    dirname = os.path.basename(grid_dir)
    # Example: split_0.7-0.3_q0.95_k10_default
    pattern = r"split_(\d+\.\d+)-(\d+\.\d+)_q(\d+\.\d+)_k(\d+)_([\w]+)"
    match = re.match(pattern, dirname)
    if not match:
        raise ValueError(f"Grid dir name does not match expected pattern: {dirname}")
    gamma_bar = float(match.group(1))
    gamma_split = float(match.group(2))
    quantile = float(match.group(3))
    delete_ratio = int(match.group(4))
    accountant = match.group(5)
    return {
        "gamma_bar": gamma_bar,
        "gamma_split": gamma_split,
        "quantile": quantile,
        "delete_ratio": delete_ratio,
        "accountant": accountant,
    }
def test_parse_csv_filename():
    """Test extracting parameters from CSV filename."""
    result = parse_csv_filename("seed_001_memorypair_gamma1.0-split0.5_zcdp.csv")
    assert result["seed"] == 1
    assert result["algo"] == "memorypair"


def test_extract_params_from_grid_dir():
    """Test extracting parameters from grid directory name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test directory name parsing
        grid_dir = os.path.join(tmpdir, "split_0.7-0.3_q0.95_k10_default")
        os.makedirs(grid_dir)
        
        params = extract_params_from_grid_dir(grid_dir)
        
        assert params["gamma_bar"] == 0.7
        assert params["gamma_split"] == 0.3
        assert params["quantile"] == 0.95
        assert params["delete_ratio"] == 10
        assert params["accountant"] == "default"


def test_convert_csv_to_parquet():
    """Test converting CSV files to Parquet format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSV file
        csv_dir = os.path.join(tmpdir, "input")
        os.makedirs(csv_dir)
        
        test_data = pd.DataFrame({
            "op": ["insert", "insert", "delete"],
            "regret": [0.1, 0.15, 0.05],
            "acc": [0.8, 0.85, 0.87]
        })
        
        csv_file = os.path.join(csv_dir, "seed_001_memorypair_test.csv")
        test_data.to_csv(csv_file, index=False)
        
        output_dir = os.path.join(tmpdir, "output")
        
        # Convert to Parquet
        convert_csv_to_parquet(csv_dir, output_dir, granularity="seed")
        
        # Check that Parquet files were created
        seeds_dir = os.path.join(output_dir, "seeds")
        assert os.path.exists(seeds_dir)
        
        # Should contain parquet files
        parquet_files = []
        for root, dirs, files in os.walk(seeds_dir):
            parquet_files.extend([f for f in files if f.endswith('.parquet')])
        assert len(parquet_files) > 0