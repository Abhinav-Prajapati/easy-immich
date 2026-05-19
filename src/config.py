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

# Immich thumbnail size to fetch — "thumbnail" is fast, "preview" is higher res
THUMBNAIL_SIZE = "preview"

# How many images to fetch per page from Immich
PAGE_SIZE = 100
