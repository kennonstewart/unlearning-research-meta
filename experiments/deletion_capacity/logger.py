# logger.py
from __future__ import annotations
import csv
from typing import Dict, List
import pathlib


class RunLogger:
    """
    Collects dict records and knows how to flush them to CSV.
    Keeps the run.py main loop free of I/O details.
    """

    def __init__(self, seed: int, algo: str, out_dir: pathlib.Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{seed}_{algo}.csv"
        self._records: List[Dict] = []
        self._fieldnames: set[str] = set()  # grows dynamically

    # ---- public API ------------------------------------------------
    def log(self, record: Dict) -> None:
        """Add a record – order of keys does NOT matter."""
        self._records.append(record)
        self._fieldnames.update(record)

    def flush(self) -> pathlib.Path:
        """Write all accumulated rows to CSV (header written once)."""
        if not self._records:
            return self.path
        fieldnames = sorted(self._fieldnames)
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        return self.path
