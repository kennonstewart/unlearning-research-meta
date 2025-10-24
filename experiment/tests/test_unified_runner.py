"""
Test the new unified experiment runner.

This test validates that the unified runner (run.py) can successfully:
1. Load grid configurations from YAML
2. Generate parameter combinations
3. Execute experiments with theory-first parameters
4. Write Parquet output via exp_engine
"""

import os
import tempfile
import json
import pytest

# Constants
GRID_ID_LENGTH = 12  # exp_engine uses 12-character SHA-256 prefix
TEST_TIMEOUT = 120  # Conservative timeout for CI environments


def test_unified_runner_dry_run():
    """Test that dry run mode works and generates grid IDs."""
    import subprocess
    
    # Get the test grid file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grid_file = os.path.join(script_dir, "..", "grids", "test_minimal.yaml")
    
    # Run dry run
    result = subprocess.run(
        ["python", "experiment/run.py", "--grid-file", grid_file, "--dry-run"],
        capture_output=True,
        text=True,
        cwd=os.path.join(script_dir, "..", ".."),
    )
    
    assert result.returncode == 0, f"Dry run failed: {result.stderr}"
    assert "Dry run" in result.stdout
    assert "parameter combinations" in result.stdout
    # Check that a grid_id hash is generated (appears as "  1. <hash>:")
    # The hash is followed by a colon in the dry run output
    import re
    grid_id_pattern = rf'\b[0-9a-f]{{{GRID_ID_LENGTH}}}\b'
    matches = re.findall(grid_id_pattern, result.stdout)
    assert len(matches) > 0, f"No grid_id found in output: {result.stdout}"


def test_unified_runner_execution():
    """Test that the runner can execute a minimal experiment and write Parquet."""
    import subprocess
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Get the test grid file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        grid_file = os.path.join(script_dir, "..", "grids", "test_minimal.yaml")
        
        # Run experiment
        result = subprocess.run(
            [
                "python", "experiment/run.py",
                "--grid-file", grid_file,
                "--output-dir", tmpdir,
            ],
            capture_output=True,
            text=True,
            cwd=os.path.join(script_dir, "..", ".."),
            timeout=TEST_TIMEOUT,
        )
        
        assert result.returncode == 0, f"Execution failed: {result.stderr}"
        assert "Grid search complete" in result.stdout
        
        # Check that Parquet files were created
        events_dir = os.path.join(tmpdir, "events")
        assert os.path.exists(events_dir), "Events directory not created"
        
        # Check that params.json was created
        grids_dir = os.path.join(tmpdir, "grids")
        assert os.path.exists(grids_dir), "Grids directory not created"
        
        # Find the grid_id directory
        grid_id_dirs = [d for d in os.listdir(grids_dir) if d.startswith("grid_id=")]
        assert len(grid_id_dirs) > 0, "No grid_id directories found"
        
        # Check params.json exists and is valid
        params_file = os.path.join(grids_dir, grid_id_dirs[0], "params.json")
        assert os.path.exists(params_file), "params.json not created"
        
        with open(params_file) as f:
            params = json.load(f)
        
        # Verify key parameters are present
        assert "grid_id" in params
        assert "target_G" in params
        assert "target_D" in params
        assert "rho_total" in params
        assert params["algo"] == "memorypair"
        assert params["accountant"] == "zcdp"


def test_config_from_dict():
    """Test that ExperimentConfig can be created from dict."""
    from experiment.config import ExperimentConfig
    
    params = {
        "target_G": 2.0,
        "target_D": 2.0,
        "target_c": 0.1,
        "target_C": 10.0,
        "target_lambda": 0.5,
        "target_PT": 25.0,
        "target_ST": 50000.0,
        "rho_total": 1.0,
        "max_events": 1000,
        "dim": 10,
    }
    
    config = ExperimentConfig.from_dict(params)
    
    assert config.target_G == 2.0
    assert config.target_D == 2.0
    assert config.rho_total == 1.0
    assert config.max_events == 1000
    assert config.dim == 10


def test_legacy_helpers_still_work():
    """Test that legacy CSV helpers are still importable."""
    from experiment.grid_runner import process_seed_output
    from experiment.runner import ALGO_MAP, _get_data_stream
    
    # Just test that imports work
    assert process_seed_output is not None
    assert ALGO_MAP is not None
    assert _get_data_stream is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
