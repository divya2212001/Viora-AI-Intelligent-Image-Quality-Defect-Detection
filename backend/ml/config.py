from pathlib import Path


ML_DIR = Path(__file__).resolve().parent

DATA_DIR = ML_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ML_DIR / "artifacts"

REPORTS_DIR = ML_DIR / "reports"

KONIQ_DIR = RAW_DIR / "koniq"

IMAGE_DIR = KONIQ_DIR / "images"

ANNOTATIONS_FILE = (
    KONIQ_DIR / "koniqplusplus.csv"
)


TOTAL_IMAGES = 10073

IMAGE_SIZE = 224

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_SEED = 42


BATCH_SIZE = 32

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# Keep this portable for local development and sandboxed execution where
# PyTorch shared-memory worker processes are unavailable.
NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 5

DEFECT_NAMES = [
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "corruption",
    "defect",
]

FEATURE_NAMES = [
    "brightness",
    "contrast",
    "sharpness",
    "noise_level",
    "entropy",
    "saturation",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
]

for directory in [
    PROCESSED_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
