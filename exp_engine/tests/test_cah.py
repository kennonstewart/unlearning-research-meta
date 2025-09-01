import pytest
import tempfile
import os
from exp_engine.engine.cah import canonicalize_params, grid_hash, attach_grid_id


def test_canonicalize_params_removes_volatile_keys():
    """Test that volatile keys are removed during canonicalization."""
    params = {
        "algo": "memorypair",
        "gamma_bar": 1.0,
        "base_out": "/tmp/results",  # volatile
        "seed": 42,  # volatile
        "hostname": "test-host",  # volatile
        "custom_path": "/some/path",  # path-like, should be removed
    }
    
    canonical = canonicalize_params(params)
    
    # Should keep non-volatile params
    assert canonical["algo"] == "memorypair"
    assert canonical["gamma_bar"] == 1.0
    
    # Should remove volatile and path-like keys
    assert "base_out" not in canonical
    assert "seed" not in canonical  
    assert "hostname" not in canonical
    assert "custom_path" not in canonical


def test_canonicalize_params_normalizes_floats():
    """Test that floats are normalized to consistent precision."""
    params = {
        "gamma_bar": 1.12345678901234567890,  # Should be rounded to 10 decimal places
        "epsilon": 0.9876543210987654321,  # Should be rounded to 10 decimal places
    }
    
    canonical = canonicalize_params(params)
    
    # Should round to 10 decimal place precision
    assert canonical["gamma_bar"] == 1.1234567890
    assert canonical["epsilon"] == 0.9876543211


def test_grid_hash_deterministic():
    """Test that grid hash is deterministic for same parameters."""
    params = {
        "algo": "memorypair",
        "gamma_bar": 1.0,
        "accountant": "zcdp"
    }
    
    hash1 = grid_hash(params)
    hash2 = grid_hash(params)
    
    assert hash1 == hash2
    assert len(hash1) == 8  # 8-character hex hash


def test_grid_hash_different_for_different_params():
    """Test that different parameters produce different hashes."""
    params1 = {"algo": "memorypair", "gamma_bar": 1.0}
    params2 = {"algo": "memorypair", "gamma_bar": 0.5}
    
    hash1 = grid_hash(params1)
    hash2 = grid_hash(params2)
    
    assert hash1 != hash2


def test_grid_hash_ignores_volatile_keys():
    """Test that volatile keys don't affect the hash."""
    base_params = {"algo": "memorypair", "gamma_bar": 1.0}
    
    params1 = {**base_params, "seed": 42, "base_out": "/tmp/a"}
    params2 = {**base_params, "seed": 99, "base_out": "/tmp/b"}
    
    hash1 = grid_hash(params1)
    hash2 = grid_hash(params2)
    
    assert hash1 == hash2  # Volatile keys shouldn't affect hash


def test_attach_grid_id():
    """Test that attach_grid_id adds a grid_id field."""
    params = {"algo": "memorypair", "gamma_bar": 1.0}
    
    result = attach_grid_id(params)
    
    assert "grid_id" in result
    assert result["algo"] == "memorypair"  # Original params preserved
    assert result["gamma_bar"] == 1.0
    assert len(result["grid_id"]) == 8  # 8-character hash


def test_canonicalize_nested_structures():
    """Test canonicalization of nested dicts and lists."""
    params = {
        "nested": {
            "inner": 1.12345678901234567890,  # Should be rounded to 10 decimal places
            "seed": 42,  # Should be removed as volatile
            "values": [1.5, 2.7, 3.9]
        },
        "list_param": [{"val": 1.12345678901234567890}, {"val": 2.0}]
    }
    
    canonical = canonicalize_params(params)
    
    # Nested volatile keys should be removed
    assert "seed" not in canonical["nested"]
    # Nested floats should be normalized  
    assert canonical["nested"]["inner"] == 1.1234567890
    # Lists should be processed recursively
    assert canonical["nested"]["values"] == [1.5, 2.7, 3.9]
    assert canonical["list_param"][0]["val"] == 1.1234567890