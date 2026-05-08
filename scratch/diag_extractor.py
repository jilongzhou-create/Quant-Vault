import os
import sys

# Simulation of "direct run"
print("--- Simulating Direct Run ---")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from content_engine import data_extractor
print(f"Direct call results: {len(data_extractor.get_pipeline_a_strategies())}")

# Simulation of "imported run" (like content_director_agent does)
print("\n--- Simulating Imported Run ---")
# Clear cache if any
if 'content_engine.data_extractor' in sys.modules:
    del sys.modules['content_engine.data_extractor']

import content_engine.data_extractor as de
print(f"Imported call results: {len(de.get_pipeline_a_strategies())}")

from config import DB_PATH
import sqlite3
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM strategy_directions sd JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id WHERE sd.source = 'RESEARCHER_AGENT' AND sv.metric_annualized_return < 0 AND sd.publish_status = 'UNPUBLISHED'")
print(f"Manual SQL count: {c.fetchone()[0]}")
conn.close()
