"""
Integration example showing how to add exp_engine APIs to existing experiment runner.

This demonstrates the minimal changes needed to start writing Parquet output
alongside existing CSV output.
"""

import os
import sys
from typing import List, Dict, Any

# Add paths for imports (adjust as needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "deletion_capacity"))

# Import exp_engine APIs
from exp_engine.engine import write_seed_rows, write_event_rows, attach_grid_id


def demonstrate_runner_integration():
    """
    Example showing how to integrate exp_engine with existing runner.
    
    This would be added to the ExperimentRunner class in runner.py
    """
    
    # Example: Converting existing seed summary data
    def save_seed_with_parquet(self, seed_summaries: List[Dict[str, Any]], base_out: str, config_params: Dict[str, Any]):
        """Save seed summaries in both CSV (existing) and Parquet (new) formats."""
        
        # Existing CSV save (unchanged)
        self.aggregate_and_save(seed_summaries, [])  # Existing method
        
        # NEW: Also save as Parquet
        try:
            write_seed_rows(seed_summaries, base_out, config_params)
            print(f"✓ Seed data also saved to Parquet: {base_out}/seeds/")
        except Exception as e:
            print(f"Warning: Could not save Parquet seed data: {e}")
    
    # Example: Converting event logs to Parquet
    def save_events_with_parquet(self, logger, base_out: str, config_params: Dict[str, Any]):
        """Save event logs in both CSV (existing) and Parquet (new) formats."""
        
        # Existing CSV save (unchanged)
        csv_path = logger.to_csv("existing_path.csv")  # Existing method
        
        # NEW: Also save as Parquet
        try:
            # Convert logger events to list of dicts
            event_data = []
            for event in logger.events:
                event_data.append(event)
            
            write_event_rows(event_data, base_out, config_params)
            print(f"✓ Event data also saved to Parquet: {base_out}/events/")
        except Exception as e:
            print(f"Warning: Could not save Parquet event data: {e}")


def demonstrate_grid_runner_integration():
    """
    Example showing how to integrate exp_engine with grid_runner.py
    
    This would be added to the grid runner processing functions.
    """
    
    # In process_seed_output function
    def enhanced_process_seed_output(csv_files, grid_id, output_dir, mandatory_fields, base_out):
        """Enhanced version that also writes Parquet."""
        
        # Existing CSV processing (unchanged)
        processed_files = []  # existing logic here...
        
        # NEW: Also create Parquet output
        try:
            seed_data = []
            for csv_file in csv_files:
                # Extract seed summary data (existing logic)
                summary = extract_seed_summary(csv_file, mandatory_fields)
                seed_data.append(summary)
            
            # Write to Parquet with grid_id
            grid_params = {"grid_id": grid_id, **mandatory_fields}
            write_seed_rows(seed_data, base_out, grid_params)
            print(f"✓ Grid {grid_id} seed data saved to Parquet")
            
        except Exception as e:
            print(f"Warning: Could not save grid {grid_id} to Parquet: {e}")
        
        return processed_files


def demonstrate_config_integration():
    """Show how to extract config params for grid_id generation."""
    
    # Example config extraction for existing Config class
    def config_to_params(config):
        """Convert Config object to parameter dict for hashing."""
        return {
            "algo": config.algo,
            "accountant": config.accountant,
            "gamma_bar": getattr(config, "gamma_bar", None),
            "gamma_split": getattr(config, "gamma_split", None),
            "delete_ratio": getattr(config, "delete_ratio", None),
            "target_PT": getattr(config, "target_PT", None),
            "target_ST": getattr(config, "target_ST", None),
            "rho_total": getattr(config, "rho_total", None),
            # Add other relevant parameters
        }
    
    # Usage in experiment runner
    def run_with_parquet_integration(config):
        """Example of running experiment with Parquet integration."""
        
        # Extract params for grid_id
        params = config_to_params(config)
        params_with_grid = attach_grid_id(params)
        grid_id = params_with_grid["grid_id"]
        
        print(f"Running experiment with grid_id: {grid_id}")
        
        # Run experiment (existing logic)
        # ...
        
        # Save results with Parquet integration
        base_out = getattr(config, "base_out", "results/parquet") 
        # save_seed_with_parquet(seed_summaries, base_out, params)
        # save_events_with_parquet(logger, base_out, params)


if __name__ == "__main__":
    print("Exp_engine integration examples:")
    print("1. demonstrate_runner_integration() - Shows runner.py integration")
    print("2. demonstrate_grid_runner_integration() - Shows grid_runner.py integration") 
    print("3. demonstrate_config_integration() - Shows config parameter extraction")
    print("\nThese functions show minimal changes needed for Parquet output alongside existing CSV.")