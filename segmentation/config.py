from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CURR_RUN = f"run_{TIMESTAMP}"
METRICS_DIR = f"metrics/{CURR_RUN}/"
