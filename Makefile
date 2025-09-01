# Makefile for exp_engine operations

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  rollup    - Run Snakemake rollup to convert CSV to Parquet"
	@echo "  clean     - Clean temporary files"

# Rollup target using exp_engine Snakemake
.PHONY: rollup
rollup:
	@echo "Running exp_engine rollup..."
	snakemake -s exp_engine/Snakefile --config base_out=results_parquet csv_dir=experiments/deletion_capacity/results --cores 8

# Clean temporary files
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true