from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATABASE_PATH = DATABASE_DIR / "parking_system.db"
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"

for directory in (DATABASE_DIR, UPLOADS_DIR, OUTPUTS_DIR, MODEL_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)

APP_NAME = "SMART PARKING AI"
APP_SUBTITLE = "Intelligent Parking Space Detection System"
SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png")
VEHICLE_CONFIDENCE = 0.25
DEFAULT_OVERLAP_THRESHOLD = 0.20
MAX_IMAGE_DIMENSION = 1600
MAX_CANVAS_WIDTH = 960
