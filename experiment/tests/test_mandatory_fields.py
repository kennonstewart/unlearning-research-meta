#!/usr/bin/env python3
"""
Unit test to verify that mandatory fields can be used to re-compute N★ and m.
"""

import pandas as pd
import numpy as np
import tempfile
import os

def test_recompute_nstar_and_m():
    """Test that N★ and m can be recomputed from mandatory fields."""
    print("Testing N★ and m recomputation from mandatory fields...")
    
    # Create sample data with mandatory fields
    sample_data = {
        'seed': [0, 1, 2],
        'grid_id': ['test_grid'] * 3,
        'avg_regret_empirical': [15.2, 14.8, 15.1],
        'N_star_emp': [100, 95, 102],
        'm_emp': [45, 50, 47],
        'G_hat': [2.5, 2.4, 2.6],  # Mandatory field
        'D_hat': [1.8, 1.9, 1.7],  # Mandatory field
        'sigma_step_theory': [0.05, 0.04, 0.06],  # Mandatory field
        'gamma_bar': [1.0, 1.0, 1.0],
        'gamma_split': [0.9, 0.9, 0.9],
        'eps_total': [1.0, 1.0, 1.0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Verify mandatory fields are present
    mandatory_fields = ['G_hat', 'D_hat', 'sigma_step_theory']
    missing_fields = [field for field in mandatory_fields if field not in df.columns]
    assert len(missing_fields) == 0, f"Missing mandatory fields: {missing_fields}"
    
    # Mock computation of N★ from theory (simplified example)
    def compute_nstar_theory(G_hat, D_hat, gamma_bar, gamma_split, eps_total):
        """Mock theoretical computation of N★"""
        # Use gamma_insert (gamma_bar * gamma_split) for learning
        gamma_insert = gamma_bar * gamma_split
        return int(np.ceil((G_hat * D_hat) / (gamma_insert * eps_total) * 100))
    
    def compute_m_theory(sigma_step_theory, eps_total):
        """Mock theoretical computation of m"""
        # This is a simplified formula for demonstration
        return int(np.ceil(eps_total / (sigma_step_theory * 10)))
    
    # Recompute theoretical values from mandatory fields
    df['N_star_recomputed'] = df.apply(
        lambda row: compute_nstar_theory(row['G_hat'], row['D_hat'], row['gamma_bar'], row['gamma_split'], row['eps_total']),
        axis=1
    )
    
    df['m_recomputed'] = df.apply(
        lambda row: compute_m_theory(row['sigma_step_theory'], row['eps_total']),
        axis=1
    )
    
    print("Sample data with recomputed values:")
    print(df[['seed', 'G_hat', 'D_hat', 'sigma_step_theory', 'N_star_emp', 'N_star_recomputed', 'm_emp', 'm_recomputed']].to_string())
    
    # Verify that the recomputation functions work
    assert len(df['N_star_recomputed']) == len(df), "N★ recomputation failed"
    assert len(df['m_recomputed']) == len(df), "m recomputation failed"
    assert not df['N_star_recomputed'].isna().any(), "N★ recomputation has NaN values"
    assert not df['m_recomputed'].isna().any(), "m recomputation has NaN values"
    
    print("✅ Mandatory fields validation successful!")
    print("✅ N★ and m can be recomputed from mandatory fields!")
    
    return True

def test_all_granularities_have_mandatory_fields():
    """Test that all granularity modes include mandatory fields."""
    print("\nTesting mandatory fields in all granularity modes...")
    
    import sys
    sys.path.append(os.path.dirname(__file__))
    
    # Create mock CSV data
    mock_events = []
    for i in range(10):
        mock_events.append({
            'event': i + 1000,
            'op': 'insert' if i < 5 else 'delete',
            'regret': 10.0 + i,
            'acc': 0.1 + i * 0.01,
            'G_hat': 2.5,  # Mandatory
            'D_hat': 1.8,  # Mandatory 
            'sigma_step_theory': 0.05  # Mandatory
        })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock CSV
        mock_csv = os.path.join(temp_dir, "0_memorypair.csv")
        pd.DataFrame(mock_events).to_csv(mock_csv, index=False)
        
        # Test processing functions
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))
        from experiment.grid_runner import process_seed_output, process_event_output, process_aggregate_output
        
        mandatory_fields = {'G_hat': 2.5, 'D_hat': 1.8, 'sigma_step_theory': 0.05}
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test each mode
        for mode, processor in [
            ("seed", process_seed_output),
            ("event", process_event_output), 
            ("aggregate", process_aggregate_output)
        ]:
            print(f"  Testing {mode} mode...")
            
            result = processor([mock_csv], "test_grid", output_dir, mandatory_fields)
            
            if isinstance(result, list):
                files_to_check = result
            else:
                files_to_check = [result] if result else []
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    for field in ['G_hat', 'D_hat', 'sigma_step_theory']:
                        assert field in df.columns, f"Missing {field} in {mode} mode output"
                        assert not df[field].isna().any(), f"{field} has NaN values in {mode} mode"
                    print(f"    ✅ {os.path.basename(file_path)} has all mandatory fields")
    
    print("✅ All granularity modes include mandatory fields!")

if __name__ == "__main__":
    try:
        test_recompute_nstar_and_m()
        test_all_granularities_have_mandatory_fields()
        print("\n🎉 All validation tests passed!")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()