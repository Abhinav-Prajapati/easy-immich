from dotenv import load_dotenv
import os

load_dotenv()

IMMICH_URL = os.getenv("IMMICH_URL", "").rstrip("/")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY", "")

DINO_MODEL_SIZE = os.getenv("DINO_MODEL", "base")
DINO_MODEL_ID = f"facebook/dinov2-{DINO_MODEL_SIZE}"

TARGET_YEAR = int(os.getenv("TARGET_YEAR", "2026"))

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = "immich_embeddings"

THUMBNAIL_SIZE = "preview"

PAGE_SIZE = 100

TIME_THRESHOLD_HOURS = float(os.getenv("TIME_THRESHOLD_HOURS", "4.0"))

VISUAL_TOLERANCE_EPS = float(os.getenv("VISUAL_TOLERANCE_EPS", "0.25"))

MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "1"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./review_clusters")
