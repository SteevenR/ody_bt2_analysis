"""Configuration and constants for BearTrap Analysis"""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_INPUT_DIR = BASE_DIR / "beartrap_data"
DATA_OUTPUT_DIR = BASE_DIR / "data"
DATA_OUTPUT_FILE = DATA_OUTPUT_DIR / "beartrap.json"
PLAYER_EVOLUTION_FILE = DATA_OUTPUT_DIR / "top_10_evolution.json"
FLAG_TEMPLATE_PATH = BASE_DIR / "assets" / "flag.png"
ALIASES_FILE = DATA_OUTPUT_DIR / "player_aliases.json"
OCR_LOG_FILE = DATA_OUTPUT_DIR / "ocr_extraction_log.jsonl"
DAMAGE_AGG_LOG_FILE = DATA_OUTPUT_DIR / "damage_aggregation_log.jsonl"

DATA_OUTPUT_DIR.mkdir(exist_ok=True)

RANK_TEMPLATES = {
    1: BASE_DIR / "assets" / "rank1.png",
    2: BASE_DIR / "assets" / "rank2.png",
    3: BASE_DIR / "assets" / "rank3.png",
}

# Multiprocessing config
NUM_WORKERS = 4
