-- PostgreSQL schema for centralizing unlearning experiment results
-- This schema normalizes the flat CSV structure into fact/dimension tables

-- Drop existing tables in dependency order
DROP TABLE IF EXISTS fact_event;
DROP TABLE IF EXISTS dim_run;
DROP TABLE IF EXISTS dim_grid;
DROP TABLE IF EXISTS lut_accountant CASCADE;
DROP TABLE IF EXISTS lut_path_type CASCADE;
DROP TABLE IF EXISTS lut_blocked_reason CASCADE;

-- Lookup tables for enum values
CREATE TABLE lut_accountant (
    accountant_id SERIAL PRIMARY KEY,
    accountant_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE lut_path_type (
    path_type_id SERIAL PRIMARY KEY,
    path_type_name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE lut_blocked_reason (
    blocked_reason_id SERIAL PRIMARY KEY,
    blocked_reason_name VARCHAR(100) UNIQUE NOT NULL
);

-- Dimension table for experiment grid configurations
CREATE TABLE dim_grid (
    grid_id VARCHAR(255) PRIMARY KEY,
    gamma_bar FLOAT,
    gamma_split FLOAT,
    accountant_id INTEGER REFERENCES lut_accountant(accountant_id),
    path_type_id INTEGER REFERENCES lut_path_type(path_type_id),
    rotate_angle FLOAT,
    drift_rate FLOAT,
    feature_scale FLOAT,
    w_scale FLOAT,
    fix_w_norm BOOLEAN,
    noise_std FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension table for individual experiment runs
CREATE TABLE dim_run (
    run_id VARCHAR(255) PRIMARY KEY,  -- SHA1 hash of grid_id + seed
    grid_id VARCHAR(255) REFERENCES dim_grid(grid_id),
    seed INTEGER NOT NULL,
    
    -- Theoretical parameters from calibration
    G_hat FLOAT,
    D_hat FLOAT,
    c_hat FLOAT,
    C_hat FLOAT,
    lambda_est FLOAT,
    S_scalar FLOAT,
    sigma_step_theory FLOAT,
    N_star_live FLOAT,
    N_star_theory FLOAT,
    m_theory_live FLOAT,
    blocked_reason_id INTEGER REFERENCES lut_blocked_reason(blocked_reason_id),
    eta_t FLOAT,
    
    -- Empirical results summary
    avg_regret_empirical FLOAT,
    N_star_emp INTEGER,
    m_emp INTEGER,
    final_acc FLOAT,
    total_events INTEGER,
    eps_spent FLOAT,
    eps_remaining FLOAT,
    delta_total FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(grid_id, seed)
);

-- Fact table for detailed experiment events
CREATE TABLE fact_event (
    event_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES dim_run(run_id),
    event_sequence INTEGER,  -- Event number within the run
    event_type VARCHAR(50),  -- e.g., 'insert', 'delete', 'predict'
    operation VARCHAR(20),   -- 'op' column from CSV
    regret FLOAT,
    accuracy FLOAT,
    timestamp_offset FLOAT,  -- Time offset from start of run
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_dim_run_grid_id ON dim_run(grid_id);
CREATE INDEX idx_dim_run_seed ON dim_run(seed);
CREATE INDEX idx_fact_event_run_id ON fact_event(run_id);
CREATE INDEX idx_fact_event_sequence ON fact_event(event_sequence);
CREATE INDEX idx_fact_event_type ON fact_event(event_type);

-- Insert default lookup values
INSERT INTO lut_accountant (accountant_name) VALUES ('eps_delta'), ('rdp'), ('zcdp'), ('relaxed') 
ON CONFLICT (accountant_name) DO NOTHING;

INSERT INTO lut_path_type (path_type_name) VALUES ('stationary'), ('drift'), ('linear'), ('rotational')
ON CONFLICT (path_type_name) DO NOTHING;

INSERT INTO lut_blocked_reason (blocked_reason_name) VALUES ('budget_exhausted'), ('capacity_reached'), ('none'), ('error')
ON CONFLICT (blocked_reason_name) DO NOTHING;

-- Grant permissions to unlearning user
GRANT ALL ON ALL TABLES IN SCHEMA public TO unlearning;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO unlearning;