"""Input and output locations, relative to the repository root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
US06_FILE = os.path.join(INPUT_DIR, "us06_epa.txt")
WLTP_FILE = os.path.join(INPUT_DIR, "WLTP-DHC-12-07e.xls")
WLTP_SOURCE = "https://unece.org/dhc-12th-session"

# Everything the scripts write goes under outputs/, which is not version-controlled.
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SHARD_DIR = os.path.join(OUTPUT_DIR, "shards")


def output_path(*parts):
    """Path under outputs/, creating its parent directory if needed."""
    path = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
