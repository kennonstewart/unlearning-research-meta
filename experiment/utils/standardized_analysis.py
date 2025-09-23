"""
Standardized Analysis Functions for Experiment Notebooks 1-5

This module provides reusable functions for the five common analyses
that should be applied across all experiment notebooks:

1. Theory Bound Tracking
   - Computes theory_ratio = cum_regret / ((G_hat^2/max(lambda_est,λ_min))*(1+ln t) + G_hat*P_T_true)
   - Analyzes last 20% of events per run
   - Returns median and final values for theory adherence checking

2. Stepsize Policy Validation  
   - If sc_active=True: checks eta_t*t ≈ 1/max(lambda_est, λ_min)
   - If sc_active=False: checks eta_t ≈ D/√S_t using base_eta_t, ST_running
   - Reports MAPE (Mean Absolute Percentage Error) and pass/fail status

3. Privacy & Odometer Sanity Checks
   - Validates m_used ≤ m_capacity (deletion capacity)
   - Validates rho_spent ≤ rho_total (privacy budget)
   - Checks sigma_step is finite
   - Verifies noise consistency: cum_regret_with_noise - cum_regret ≈ noise_regret_cum

4. Seed Stability Audit
   - Computes dispersion (IQR, std, CV) for final cum_regret and P_T_true across seeds
   - Flags grids with high variability (CV > 0.5 or IQR/mean > 0.5)
   - Useful for identifying runs that need replication or investigation

5. Enhanced Claim Check Export
   - Extends per-notebook claim-check JSON with all analysis results
   - Includes pass/fail summaries and statistical measures
   - Structured for downstream CI consumption

Usage:
    from standardized_analysis import run_all_standardized_analyses, enhance_claim_check_export
    
    # Run all analyses
    results = run_all_standardized_analyses(con, runs_df)
    
    # Enhance claim check export
    enhanced_summary = enhance_claim_check_export(base_summary, 
        results['theory_bound_tracking'], results['stepsize_policy_validation'],
        results['privacy_odometer_checks'], results['seed_stability_audit'])
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json


def compute_theory_bound_tracking(con, runs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute theory ratio = cum_regret / ((G_hat^2/max(lambda_est,λ_min))*(1+ln t) + G_hat*P_T_true)
    over time for last 20% of events. Return median and final values per run.
    
    Args:
        con: DuckDB connection
        runs_df: DataFrame with grid_id and seed columns
        
    Returns:
        Dict with theory ratio statistics per run
    """
    lambda_min = 1e-6  # Minimum lambda value to avoid division by zero
    
    results = {}
    
    for _, run in runs_df.iterrows():
        grid_id, seed = run['grid_id'], run['seed']
        
        try:
            # Get event data for this run
            df = con.execute(f"""
                SELECT event_id, cum_regret, P_T_true, lambda_est, G_hat
                FROM analytics.fact_event 
                WHERE grid_id = ? AND seed = ?
                AND cum_regret IS NOT NULL 
                AND P_T_true IS NOT NULL 
                AND lambda_est IS NOT NULL 
                AND G_hat IS NOT NULL
                ORDER BY event_id
            """, [grid_id, seed]).df()
            
            if df.empty:
                results[f"{grid_id}_{seed}"] = {
                    'theory_ratio_median_tail': None,
                    'theory_ratio_final': None,
                    'status': 'no_data'
                }
                continue
                
            # Take last 20% of events
            tail_start = int(len(df) * 0.8)
            tail_df = df.iloc[tail_start:]
            
            if tail_df.empty:
                results[f"{grid_id}_{seed}"] = {
                    'theory_ratio_median_tail': None,
                    'theory_ratio_final': None,
                    'status': 'insufficient_data'
                }
                continue
            
            # Compute theory ratio for tail events
            ratios = []
            for _, row in tail_df.iterrows():
                t = row['event_id']
                cum_regret = row['cum_regret']
                P_T_true = row['P_T_true']
                lambda_est = max(row['lambda_est'], lambda_min)
                G_hat = row['G_hat']
                
                # theory_ratio = cum_regret / ((G_hat^2/lambda_est)*(1+ln t) + G_hat*P_T_true)
                if t > 0 and G_hat > 0:
                    denominator = (G_hat**2 / lambda_est) * (1 + np.log(t)) + G_hat * P_T_true
                    if denominator > 0:
                        ratio = cum_regret / denominator
                        ratios.append(ratio)
                        
            if ratios:
                results[f"{grid_id}_{seed}"] = {
                    'theory_ratio_median_tail': np.median(ratios),
                    'theory_ratio_final': ratios[-1],
                    'status': 'success'
                }
            else:
                results[f"{grid_id}_{seed}"] = {
                    'theory_ratio_median_tail': None,
                    'theory_ratio_final': None,
                    'status': 'computation_failed'
                }
                
        except Exception as e:
            results[f"{grid_id}_{seed}"] = {
                'theory_ratio_median_tail': None,
                'theory_ratio_final': None,
                'status': f'error: {str(e)}'
            }
            
    return results


def validate_stepsize_policy(con, runs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate stepsize policies:
    - If sc_active=True, check eta_t*t ≈ 1/max(lambda_est, λ_min) over tail
    - If sc_active=False, check eta_t ≈ D/√S_t using base_eta_t, ST_running
    
    Args:
        con: DuckDB connection
        runs_df: DataFrame with grid_id and seed columns
        
    Returns:
        Dict with stepsize policy validation results per run
    """
    lambda_min = 1e-6
    
    results = {}
    
    for _, run in runs_df.iterrows():
        grid_id, seed = run['grid_id'], run['seed']
        
        try:
            # Get tail events (last 20%)
            df = con.execute(f"""
                WITH tail_events AS (
                    SELECT event_id, eta_t, base_eta_t, sc_active, lambda_est, ST_running, D_bound
                    FROM analytics.fact_event 
                    WHERE grid_id = ? AND seed = ?
                    AND eta_t IS NOT NULL
                    ORDER BY event_id
                    LIMIT -1 OFFSET (SELECT CAST(COUNT(*) * 0.8 AS INTEGER) FROM analytics.fact_event WHERE grid_id = ? AND seed = ?)
                )
                SELECT * FROM tail_events
            """, [grid_id, seed, grid_id, seed]).df()
            
            if df.empty:
                results[f"{grid_id}_{seed}"] = {
                    'stepsize_policy_mape': None,
                    'stepsize_policy_status': 'no_data'
                }
                continue
            
            errors = []
            policy_type = None
            
            for _, row in df.iterrows():
                t = row['event_id']
                eta_t = row['eta_t']
                sc_active = row['sc_active']
                
                if pd.isna(eta_t) or eta_t <= 0 or t <= 0:
                    continue
                    
                if sc_active:
                    # Check eta_t*t ≈ 1/max(lambda_est, λ_min)
                    lambda_est = max(row['lambda_est'] if not pd.isna(row['lambda_est']) else lambda_min, lambda_min)
                    expected = 1.0 / lambda_est
                    actual = eta_t * t
                    policy_type = 'strong_convexity'
                else:
                    # Check eta_t ≈ D/√S_t using base_eta_t, ST_running
                    base_eta_t = row['base_eta_t']
                    ST_running = row['ST_running']
                    
                    if not pd.isna(base_eta_t) and not pd.isna(ST_running) and ST_running > 0:
                        expected = base_eta_t / np.sqrt(ST_running)
                        actual = eta_t
                        policy_type = 'adagrad'
                    else:
                        continue
                
                if expected > 0:
                    error = abs(actual - expected) / expected
                    errors.append(error)
            
            if errors:
                mape = np.mean(errors) * 100  # Convert to percentage
                status = 'pass' if mape < 20 else 'fail'  # 20% threshold
                results[f"{grid_id}_{seed}"] = {
                    'stepsize_policy_mape': mape,
                    'stepsize_policy_status': status,
                    'stepsize_policy_type': policy_type
                }
            else:
                results[f"{grid_id}_{seed}"] = {
                    'stepsize_policy_mape': None,
                    'stepsize_policy_status': 'insufficient_data',
                    'stepsize_policy_type': None
                }
                
        except Exception as e:
            results[f"{grid_id}_{seed}"] = {
                'stepsize_policy_mape': None,
                'stepsize_policy_status': f'error: {str(e)}',
                'stepsize_policy_type': None
            }
            
    return results


def check_privacy_odometer_sanity(con, runs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check privacy and odometer sanity at run end:
    - m_used ≤ m_capacity
    - rho_spent ≤ rho_total  
    - sigma_step is finite
    - cum_regret_with_noise - cum_regret ≈ noise_regret_cum
    
    Args:
        con: DuckDB connection
        runs_df: DataFrame with grid_id and seed columns
        
    Returns:
        Dict with sanity check results per run
    """
    results = {}
    
    for _, run in runs_df.iterrows():
        grid_id, seed = run['grid_id'], run['seed']
        
        try:
            # Get final event data
            final_row = con.execute(f"""
                SELECT 
                    m_used, m_capacity, rho_spent, rho_total, sigma_step,
                    cum_regret, cum_regret_with_noise, noise_regret_cum
                FROM analytics.fact_event 
                WHERE grid_id = ? AND seed = ?
                AND event_id = (SELECT MAX(event_id) FROM analytics.fact_event WHERE grid_id = ? AND seed = ?)
            """, [grid_id, seed, grid_id, seed]).df()
            
            if final_row.empty:
                results[f"{grid_id}_{seed}"] = {
                    'privacy_odometer_status': 'no_data',
                    'checks': {}
                }
                continue
                
            row = final_row.iloc[0]
            checks = {}
            all_pass = True
            
            # Check m_used ≤ m_capacity
            if not pd.isna(row['m_used']) and not pd.isna(row['m_capacity']):
                m_check = row['m_used'] <= row['m_capacity']
                checks['m_capacity_check'] = {
                    'pass': m_check,
                    'value': f"{row['m_used']}/{row['m_capacity']}"
                }
                all_pass &= m_check
            
            # Check rho_spent ≤ rho_total
            if not pd.isna(row['rho_spent']) and not pd.isna(row['rho_total']):
                rho_check = row['rho_spent'] <= row['rho_total']
                checks['rho_budget_check'] = {
                    'pass': rho_check,
                    'value': f"{row['rho_spent']:.6f}/{row['rho_total']:.6f}"
                }
                all_pass &= rho_check
            
            # Check sigma_step is finite
            if not pd.isna(row['sigma_step']):
                sigma_check = np.isfinite(row['sigma_step'])
                checks['sigma_finite_check'] = {
                    'pass': sigma_check,
                    'value': f"{row['sigma_step']:.6f}"
                }
                all_pass &= sigma_check
                
            # Check noise consistency: cum_regret_with_noise - cum_regret ≈ noise_regret_cum
            if (not pd.isna(row['cum_regret_with_noise']) and 
                not pd.isna(row['cum_regret']) and 
                not pd.isna(row['noise_regret_cum'])):
                
                noise_diff = row['cum_regret_with_noise'] - row['cum_regret']
                expected_noise = row['noise_regret_cum']
                
                if abs(expected_noise) > 1e-10:
                    noise_error = abs(noise_diff - expected_noise) / abs(expected_noise)
                    noise_check = noise_error < 0.1  # 10% tolerance
                else:
                    noise_check = abs(noise_diff) < 1e-6
                    
                checks['noise_consistency_check'] = {
                    'pass': noise_check,
                    'value': f"diff={noise_diff:.6f}, expected={expected_noise:.6f}"
                }
                all_pass &= noise_check
            
            results[f"{grid_id}_{seed}"] = {
                'privacy_odometer_status': 'pass' if all_pass else 'fail',
                'checks': checks
            }
            
        except Exception as e:
            results[f"{grid_id}_{seed}"] = {
                'privacy_odometer_status': f'error: {str(e)}',
                'checks': {}
            }
            
    return results


def audit_seed_stability(con, grid_ids: List[str]) -> Dict[str, Any]:
    """
    For each grid_id, compute dispersion (IQR and std) for final cum_regret and P_T_true across seeds.
    Flag grids with high variability.
    
    Args:
        con: DuckDB connection
        grid_ids: List of grid_id values to analyze
        
    Returns:
        Dict with seed stability statistics per grid
    """
    results = {}
    
    for grid_id in grid_ids:
        try:
            # Get final values across all seeds for this grid
            df = con.execute(f"""
                SELECT 
                    seed,
                    MAX(cum_regret) FILTER (WHERE event_id = (SELECT MAX(event_id) FROM analytics.fact_event fe2 WHERE fe2.grid_id = fe.grid_id AND fe2.seed = fe.seed)) AS final_cum_regret,
                    MAX(P_T_true) FILTER (WHERE event_id = (SELECT MAX(event_id) FROM analytics.fact_event fe2 WHERE fe2.grid_id = fe.grid_id AND fe2.seed = fe.seed)) AS final_P_T_true
                FROM analytics.fact_event fe
                WHERE grid_id = ?
                GROUP BY seed
                HAVING final_cum_regret IS NOT NULL AND final_P_T_true IS NOT NULL
            """, [grid_id]).df()
            
            if len(df) < 2:
                results[grid_id] = {
                    'status': 'insufficient_seeds',
                    'num_seeds': len(df)
                }
                continue
            
            # Compute statistics for cum_regret
            regret_values = df['final_cum_regret'].dropna()
            regret_stats = {
                'mean': float(regret_values.mean()),
                'std': float(regret_values.std()),
                'iqr': float(regret_values.quantile(0.75) - regret_values.quantile(0.25)),
                'cv': float(regret_values.std() / regret_values.mean()) if regret_values.mean() != 0 else float('inf')
            }
            
            # Compute statistics for P_T_true
            pt_values = df['final_P_T_true'].dropna()
            pt_stats = {
                'mean': float(pt_values.mean()),
                'std': float(pt_values.std()),
                'iqr': float(pt_values.quantile(0.75) - pt_values.quantile(0.25)),
                'cv': float(pt_values.std() / pt_values.mean()) if pt_values.mean() != 0 else float('inf')
            }
            
            # Flag high variability (CV > 0.5 or large IQR relative to mean)
            regret_flag = regret_stats['cv'] > 0.5 or (regret_stats['iqr'] / regret_stats['mean'] > 0.5 if regret_stats['mean'] != 0 else True)
            pt_flag = pt_stats['cv'] > 0.5 or (pt_stats['iqr'] / pt_stats['mean'] > 0.5 if pt_stats['mean'] != 0 else True)
            
            results[grid_id] = {
                'status': 'success',
                'num_seeds': len(df),
                'cum_regret_stats': regret_stats,
                'P_T_true_stats': pt_stats,
                'high_variability_flag': regret_flag or pt_flag,
                'regret_high_var': regret_flag,
                'pt_high_var': pt_flag
            }
            
        except Exception as e:
            results[grid_id] = {
                'status': f'error: {str(e)}',
                'num_seeds': 0
            }
            
    return results


def enhance_claim_check_export(base_summary: List[Dict], theory_results: Dict, 
                              stepsize_results: Dict, privacy_results: Dict, 
                              stability_results: Dict) -> List[Dict]:
    """
    Enhance the base claim check summary with results from all standardized analyses.
    
    Args:
        base_summary: Original summary list from claim check
        theory_results: Results from theory bound tracking
        stepsize_results: Results from stepsize policy validation
        privacy_results: Results from privacy/odometer checks
        stability_results: Results from seed stability audit
        
    Returns:
        Enhanced summary list with additional analysis fields
    """
    enhanced_summary = []
    
    for entry in base_summary:
        enhanced_entry = entry.copy()
        
        grid_id = entry.get('grid_id')
        seed = entry.get('seed')
        run_key = f"{grid_id}_{seed}"
        
        # Add theory bound tracking results
        if run_key in theory_results:
            enhanced_entry.update({
                'theory_ratio_median_tail': theory_results[run_key]['theory_ratio_median_tail'],
                'theory_ratio_final': theory_results[run_key]['theory_ratio_final'],
                'theory_status': theory_results[run_key]['status']
            })
        
        # Add stepsize policy validation results
        if run_key in stepsize_results:
            enhanced_entry.update({
                'stepsize_policy_mape': stepsize_results[run_key]['stepsize_policy_mape'],
                'stepsize_policy_status': stepsize_results[run_key]['stepsize_policy_status'],
                'stepsize_policy_type': stepsize_results[run_key].get('stepsize_policy_type')
            })
        
        # Add privacy/odometer sanity check results
        if run_key in privacy_results:
            enhanced_entry.update({
                'privacy_odometer_status': privacy_results[run_key]['privacy_odometer_status'],
                'privacy_checks': privacy_results[run_key]['checks']
            })
        
        # Add seed stability results for this grid
        if grid_id in stability_results:
            enhanced_entry.update({
                'seed_stability_status': stability_results[grid_id]['status'],
                'seed_high_variability_flag': stability_results[grid_id].get('high_variability_flag', False)
            })
        
        enhanced_summary.append(enhanced_entry)
    
    return enhanced_summary


def run_all_standardized_analyses(con, runs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all standardized analyses and return a consolidated results dictionary.
    
    Args:
        con: DuckDB connection
        runs_df: DataFrame with grid_id and seed columns
        
    Returns:
        Dictionary containing all analysis results
    """
    print("Running Theory Bound Tracking...")
    theory_results = compute_theory_bound_tracking(con, runs_df)
    
    print("Running Stepsize Policy Validation...")
    stepsize_results = validate_stepsize_policy(con, runs_df)
    
    print("Running Privacy & Odometer Sanity Checks...")
    privacy_results = check_privacy_odometer_sanity(con, runs_df)
    
    print("Running Seed Stability Audit...")
    unique_grids = runs_df['grid_id'].unique().tolist()
    stability_results = audit_seed_stability(con, unique_grids)
    
    return {
        'theory_bound_tracking': theory_results,
        'stepsize_policy_validation': stepsize_results,
        'privacy_odometer_checks': privacy_results,
        'seed_stability_audit': stability_results
    }