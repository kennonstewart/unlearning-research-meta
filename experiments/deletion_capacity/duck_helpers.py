import os
import sys

# Add project root to path for exp_engine imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_engine.engine.duck import create_connection_and_views, query_seeds, query_events

def connect(base_out="results_parquet"):
    return create_connection_and_views(base_out)

def q_seeds(conn, where=None, limit=None):
    return query_seeds(conn, where or "TRUE", limit=limit or 1000)

def q_events(conn, where=None, limit=None):
    return query_events(conn, where or "TRUE", limit=limit)