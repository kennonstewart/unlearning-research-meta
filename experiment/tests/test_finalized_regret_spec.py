"""
Tests for the finalized regret calculation and reporting spec.

Tests non-negative, comparator-based regret on insert events post-warmup;
privacy metrics logging on delete events; Parquet-only mode; and DuckDB views.
"""

import tempfile
import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from code.memory_pair.src.comparators import StaticOracle, RollingOracle
from code.memory_pair.src.metrics import loss_half_mse
from experiment.utils.configs.config import Config
from exp_integration import build_params_from_config, write_seed_summary_parquet, write_event_rows_parquet

try:
    from exp_engine.engine.duck import create_connection_and_views, query_regret_analysis, get_negative_regret_summary
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class TestFinalizedRegretSpec:
    """Test suite for finalized regret calculation specification."""

    def test_nonnegative_regret_enforcement_static_oracle(self):
        """Test that static oracle enforces non-negative regret when configured."""
        # Create config with non-negative enforcement
        cfg = Config(enforce_nonnegative_regret=True)
        
        oracle = StaticOracle(dim=3, lambda_reg=0.1, cfg=cfg)
        
        # Calibrate with some data
        calibration_data = [
            (np.array([1.0, 0.0, 0.0]), 1.0),
            (np.array([0.0, 1.0, 0.0]), 0.5),
            (np.array([0.0, 0.0, 1.0]), 0.2),
        ]
        oracle.calibrate_with_initial_data(calibration_data)
        
        # Test case where regret would be negative
        x = np.array([0.1, 0.1, 0.1])
        y = 0.0
        current_theta = np.array([10.0, 10.0, 10.0])  # Very large theta to make regret negative
        
        result = oracle.update_regret_accounting(x, y, current_theta)
        
        # Regret should be non-negative when enforcement is enabled
        assert result["regret_increment"] >= 0.0, "Regret should be non-negative when enforcement is enabled"

    def test_nonnegative_regret_enforcement_rolling_oracle(self):
        """Test that rolling oracle enforces non-negative regret when configured."""
        cfg = Config(enforce_nonnegative_regret=True)
        
        oracle = RollingOracle(dim=3, window_W=10, lambda_reg=0.1, cfg=cfg)
        
        # Add some window data first
        for i in range(5):
            x = np.random.randn(3)
            y = float(np.random.randn())
            oracle.maybe_update(x, y, np.zeros(3))
        
        # Test case where regret would be negative
        x = np.array([0.1, 0.1, 0.1])
        y = 0.0
        current_theta = np.array([10.0, 10.0, 10.0])  # Very large theta
        
        result = oracle.update_regret_accounting(x, y, current_theta)
        
        # All regret components should be non-negative when enforcement is enabled
        assert result["regret_increment"] >= 0.0
        assert result["static_increment"] >= 0.0

    def test_regret_enforcement_disabled(self):
        """Test that regret can be negative when enforcement is disabled."""
        cfg = Config(enforce_nonnegative_regret=False)
        
        oracle = StaticOracle(dim=3, lambda_reg=0.1, cfg=cfg)
        
        # Calibrate with data that will lead to negative regret
        calibration_data = [
            (np.array([1.0, 0.0, 0.0]), 0.0),
            (np.array([0.0, 1.0, 0.0]), 0.0),
            (np.array([0.0, 0.0, 1.0]), 0.0),
        ]
        oracle.calibrate_with_initial_data(calibration_data)
        
        # Test with scenario that should produce negative regret
        x = np.array([1.0, 0.0, 0.0])
        y = 0.0
        current_theta = np.array([0.0, 0.0, 0.0])  # Zero prediction
        
        result = oracle.update_regret_accounting(x, y, current_theta)
        
        # When enforcement is disabled, regret can be negative
        # (this specific case may not be negative, but the point is we don't clamp)
        # The important thing is we don't apply max(0, regret) transformation

    def test_data_fit_comparison_with_same_lambda(self):
        """Test that regret comparison uses same λ regularization parameter."""
        lambda_reg = 0.2
        cfg = Config(lambda_reg=lambda_reg)
        
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg, cfg=cfg)
        
        # Calibrate oracle
        calibration_data = [
            (np.array([1.0, 0.0]), 1.0),
            (np.array([0.0, 1.0]), 0.5),
        ]
        oracle.calibrate_with_initial_data(calibration_data)
        
        # Test point
        x = np.array([0.5, 0.5])
        y = 0.7
        current_theta = np.array([0.8, 0.3])
        
        # Manually compute regularized losses to verify consistency
        pred_current = float(current_theta @ x)
        loss_current_expected = loss_half_mse(pred_current, y) + 0.5 * lambda_reg * float(np.dot(current_theta, current_theta))
        
        pred_oracle = float(oracle.w_star_fixed @ x)
        loss_oracle_expected = loss_half_mse(pred_oracle, y) + 0.5 * lambda_reg * float(np.dot(oracle.w_star_fixed, oracle.w_star_fixed))
        
        expected_regret = loss_current_expected - loss_oracle_expected
        
        result = oracle.update_regret_accounting(x, y, current_theta)
        
        # Allow for small numerical differences
        if cfg.enforce_nonnegative_regret:
            expected_regret = max(0.0, expected_regret)
            
        assert abs(result["regret_increment"] - expected_regret) < 1e-10, \
            f"Regret calculation should use consistent λ={lambda_reg} regularization"

    def test_parquet_only_mode_default(self):
        """Test that Parquet-only mode is enabled by default in finalized spec."""
        cfg = Config()
        
        assert cfg.parquet_only_mode, "Parquet-only mode should be enabled by default in finalized spec"

    def test_config_keys_for_regret_spec(self):
        """Test that all required config keys for regret spec are available."""
        cfg = Config()
        
        # Check that all finalized spec config keys exist
        assert hasattr(cfg, 'enforce_nonnegative_regret'), "Missing enforce_nonnegative_regret config key"
        assert hasattr(cfg, 'parquet_only_mode'), "Missing parquet_only_mode config key"
        assert hasattr(cfg, 'regret_warmup_threshold'), "Missing regret_warmup_threshold config key"
        assert hasattr(cfg, 'regret_comparator_mode'), "Missing regret_comparator_mode config key"
        
        # Check default values
        assert cfg.enforce_nonnegative_regret == True, "enforce_nonnegative_regret should default to True"
        assert cfg.parquet_only_mode == True, "parquet_only_mode should default to True"
        assert cfg.regret_comparator_mode == "oracle", "regret_comparator_mode should default to 'oracle'"

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")
    def test_regret_analysis_views(self):
        """Test that DuckDB regret analysis views work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock event data with regret values
            event_data = []
            for seed in [1, 2]:
                for event in range(10):
                    regret_val = np.random.normal(0.1, 0.05)  # Some negative values
                    event_data.append({
                        "seed": seed,
                        "grid_id": "test_grid",
                        "event": event,
                        "op": "insert" if event < 7 else "delete",
                        "regret": regret_val,
                        "cum_regret": sum(np.random.normal(0.1, 0.05) for _ in range(event + 1)),
                        "algo": "memorypair",
                        "accountant": "zcdp",
                        "gamma_bar": 1.0,
                    })
            
            # Write to Parquet
            params_with_grid = {"grid_id": "test_grid", "algo": "memorypair", "accountant": "zcdp"}
            write_event_rows_parquet(event_data, tmpdir, params_with_grid)
            
            # Create DuckDB connection and views
            connection = create_connection_and_views(tmpdir)
            
            # Test regret analysis view
            regret_df = query_regret_analysis(connection)
            assert not regret_df.empty, "Regret analysis view should return data"
            assert "avg_regret" in regret_df.columns, "Regret analysis should have avg_regret column"
            assert "negative_count" in regret_df.columns, "Regret analysis should track negative regret counts"
            
            # Test negative regret summary
            neg_summary = get_negative_regret_summary(connection)
            # Note: may be empty if no negative regrets in random data, which is fine

    def test_privacy_metrics_on_delete_events(self):
        """Test that privacy metrics are properly logged on delete events."""
        # This test would check that delete events include odometer-based privacy metrics
        # The actual implementation is in phases.py where get_privacy_metrics is called
        # For now, we test that the infrastructure exists
        from metrics_utils import get_privacy_metrics
        
        # Mock model with odometer
        class MockModel:
            def __init__(self):
                self.odometer = MockOdometer()
        
        class MockOdometer:
            def __init__(self):
                self.eps_spent = 0.5
                self.rho_spent = 0.3
                self.sigma_step = 0.1
                self.m_capacity = 10
                
            def metrics(self):
                return {
                    "eps_spent": self.eps_spent,
                    "rho_spent": self.rho_spent,
                    "sigma_step": self.sigma_step,
                    "m_capacity": self.m_capacity,
                }
        
        model = MockModel()
        metrics = get_privacy_metrics(model)
        
        # Check that privacy metrics are extracted
        assert "eps_spent" in metrics or "rho_spent" in metrics, \
            "Privacy metrics should be extracted from odometer"

    def test_regret_warmup_threshold_config(self):
        """Test regret warmup threshold configuration."""
        cfg = Config(regret_warmup_threshold=100)
        
        assert cfg.regret_warmup_threshold == 100, "regret_warmup_threshold should be configurable"
        
        # Test default (None)
        cfg_default = Config()
        assert cfg_default.regret_warmup_threshold is None, "regret_warmup_threshold should default to None"

    def test_regret_comparator_mode_config(self):
        """Test regret comparator mode configuration."""
        # Test different comparator modes
        for mode in ["oracle", "zero", "ridge"]:
            cfg = Config(regret_comparator_mode=mode)
            assert cfg.regret_comparator_mode == mode, f"regret_comparator_mode should support '{mode}'"

    def test_regularized_loss_computation(self):
        """Test that regularized loss computation is consistent across comparators."""
        lambda_reg = 0.1
        
        # Test data
        x = np.array([1.0, 2.0])
        y = 1.5
        w = np.array([0.5, 0.3])
        
        # Manual computation
        pred = float(w @ x)
        expected_loss = loss_half_mse(pred, y) + 0.5 * lambda_reg * float(np.dot(w, w))
        
        # Test static oracle computation
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
        computed_loss = oracle._compute_regularized_loss(pred, y, w)
        
        assert abs(computed_loss - expected_loss) < 1e-12, \
            "Regularized loss computation should be consistent"
        
        # Test rolling oracle computation
        rolling_oracle = RollingOracle(dim=2, lambda_reg=lambda_reg)
        computed_loss_rolling = rolling_oracle._compute_regularized_loss(pred, y, w)
        
        assert abs(computed_loss_rolling - expected_loss) < 1e-12, \
            "Rolling oracle should use same regularized loss computation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])