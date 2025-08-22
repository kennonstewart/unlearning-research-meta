-- PostgreSQL schema for centralizing unlearning experiment results
-- Simplified approach - store CSV data as-is with minimal normalization

-- Drop existing tables
DROP TABLE IF EXISTS fact_event CASCADE;
DROP TABLE IF EXISTS dim_run CASCADE;
DROP TABLE IF EXISTS dim_grid CASCADE;
DROP TABLE IF EXISTS lut_accountant CASCADE;
DROP TABLE IF EXISTS lut_path_type CASCADE;
DROP TABLE IF EXISTS lut_blocked_reason CASCADE;

-- Simple lookup tables
CREATE TABLE lut_accountant (
    accountant_id SERIAL PRIMARY KEY,
    accountant_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE lut_path_type (
    path_type_id SERIAL PRIMARY KEY,
    path_type_name VARCHAR(50) UNIQUE NOT NULL
);

-- Main dimension table for grids (experiment configurations)
CREATE TABLE dim_grid (
    grid_id VARCHAR(255) PRIMARY KEY,
    gamma_bar FLOAT,
    gamma_split FLOAT,
    accountant VARCHAR(50),
    path_type VARCHAR(50),
    rotate_angle FLOAT,
    drift_rate FLOAT,
    feature_scale FLOAT,
    w_scale FLOAT,
    fix_w_norm BOOLEAN,
    noise_std FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main fact table combining run and event data
CREATE TABLE fact_event (
    event_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255),  -- SHA1 hash of grid_id + seed
    grid_id VARCHAR(255),
    seed INTEGER,
    
    -- Experiment parameters
    gamma_bar FLOAT,
    gamma_split FLOAT,
    accountant VARCHAR(50),
    
    -- Theoretical calibration values
    G_hat FLOAT,
    D_hat FLOAT,
    c_hat FLOAT,
    C_hat_upper FLOAT,
    lambda_est FLOAT,
    S_scalar FLOAT,
    sigma_step_theory FLOAT,
    N_star_live FLOAT,
    N_star_theory FLOAT,
    m_theory_live FLOAT,
    blocked_reason VARCHAR(100),
    eta_t FLOAT,
    
    -- Path/drift parameters
    path_type VARCHAR(50),
    rotate_angle FLOAT,
    drift_rate FLOAT,
    feature_scale FLOAT,
    w_scale FLOAT,
    fix_w_norm BOOLEAN,
    noise_std FLOAT,
    
    -- Privacy accounting
    eps_spent FLOAT,
    eps_remaining FLOAT,
    delta_total FLOAT,
    
    -- Summary metrics (same for all events in a run)
    avg_regret_empirical FLOAT,
    N_star_emp INTEGER,
    m_emp INTEGER,
    final_acc FLOAT,
    total_events INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_fact_event_run_id ON fact_event(run_id);
CREATE INDEX idx_fact_event_grid_id ON fact_event(grid_id);
CREATE INDEX idx_fact_event_seed ON fact_event(seed);
CREATE INDEX idx_fact_event_accountant ON fact_event(accountant);

-- Insert lookup values
INSERT INTO lut_accountant (accountant_name) VALUES 
    ('eps_delta'), ('rdp'), ('zcdp'), ('relaxed') 
ON CONFLICT (accountant_name) DO NOTHING;

INSERT INTO lut_path_type (path_type_name) VALUES 
    ('stationary'), ('drift'), ('linear'), ('rotational')
ON CONFLICT (path_type_name) DO NOTHING;

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO unlearning;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO unlearning;