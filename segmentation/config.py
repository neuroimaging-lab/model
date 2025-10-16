from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CURR_RUN = f"run_{TIMESTAMP}"
METRICS_DIR = f"metrics/{CURR_RUN}/"
DATA_LABELS = ["Background", "Edema", "Non-enhancing tumor", "Enhancing tumour"]

SUPER_COMPUTER_ENABLED = True  # NOTE: If False, images passed to the model would be cut to fit into the GPU memory
#                                       also the model size might not be full

# Wheter to save best model while training
SAVE_BEST_MODEL = (
    True  # NOTE: when SUPER_COMPUTER_ENABLED=True, model would be saved anyway
)
