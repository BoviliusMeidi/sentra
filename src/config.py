from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MICRO_ROOT = DATA_DIR / "imagery"
SHP_KECAMATAN_DIR = DATA_DIR / "shp/kecamatan"
CONFIG_CSV_PATH = DATA_DIR / "config.csv"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# Constant list of Kecamatan
KECAMATAN_LIST = [
    "Bogor Selatan",
    "Bogor Tengah",
    "Bogor Timur",
    "Bogor Utara",
    "Ciawi",
    "Cisarua",
    "Megamendung",
    "Sukaraja",
    "Tanah Sareal",
]

# Constant list of All Band
ALL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

RANDOM_SEED = 42;
DEFAULT_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]
DEFAULT_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_TEST_RATIOS = [70, 80, 90]
DEFAULT_SAMPLES = [2500, 5000, 10000, 20000, "All"]
