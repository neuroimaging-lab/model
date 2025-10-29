from datetime import datetime
import os
from pathlib import Path

CLUSTER_TRAINING_ENABLED = True  # NOTE: If False, 3D MRI images are center-cropped to fit into GPU memory, reducing their voxel dimensions (e.g. 160x240x240 → 80x160x160).

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def get_dataset_dir():
    scratch = os.environ.get(
        "SCRATCH"
    )  # High-speed storage available in cluster environment
    if CLUSTER_TRAINING_ENABLED and scratch:
        return Path(scratch) / "ndziwak" / "data" / "Task01_BrainTumour"
    else:
        return Path(__file__).parent.parent / "datasets" / "Task01_BrainTumour"


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CURR_RUN = f"run_{TIMESTAMP}"
METRICS_DIR = f"metrics/{CURR_RUN}/"
DATA_LABELS = ["Background", "Edema", "Non-enhancing tumor", "Enhancing tumour"]
DATASET_DIR = get_dataset_dir()

# Whether to save best model while training
SAVE_BEST_MODEL = (
    False  # NOTE: when SUPER_COMPUTER_ENABLED=True, model would be saved anyway
)
