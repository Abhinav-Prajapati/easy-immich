"""
Immich API client.

Fetches image assets for a given year and downloads their thumbnails.
Uses the Immich REST API:
  - POST /api/search/metadata  → paginated asset search with date filters
  - GET  /api/assets/{id}/thumbnail  → download thumbnail bytes
"""

import httpx
from datetime import datetime, timezone
from typing import Generator
from src.config import (
    IMMICH_URL,
    IMMICH_API_KEY,
    TARGET_YEAR,
    PAGE_SIZE,
    THUMBNAIL_SIZE,
)


HEADERS = {
    "x-api-key": IMMICH_API_KEY,
    "Accept": "application/json",
}


def _date_range(year: int) -> tuple[str, str]:
    """Return ISO-8601 start/end strings for the full year."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc).isoformat()
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).isoformat()
    return start, end


def fetch_assets(year: int = TARGET_YEAR) -> Generator[dict, None, None]:
    """
    Yield asset metadata dicts for all images taken in `year`.

    Each dict contains at minimum:
        id, originalFileName, localDateTime, exifInfo (if available)
    """
    taken_after, taken_before = _date_range(year)
    page = 1

    with httpx.Client(headers=HEADERS, timeout=30) as client:
        while True:
            payload = {
                "type": "IMAGE",
                "takenAfter": taken_after,
                "takenBefore": taken_before,
                "withExif": True,
                "page": page,
                "size": PAGE_SIZE,
            }

            resp = client.post(f"{IMMICH_URL}/api/search/metadata", json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Immich returns { assets: { items: [...], nextPage: int|null } }
            assets = data.get("assets", {})
            items = assets.get("items", [])

            if not items:
                break

            yield from items

            # If no nextPage, we've exhausted all results
            if not assets.get("nextPage"):
                break

            page += 1


def download_thumbnail(asset_id: str, client: httpx.Client) -> bytes | None:
    """
    Download the thumbnail for a single asset.
    Returns raw image bytes, or None on failure.
    """
    try:
        resp = client.get(
            f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail",
            params={"size": THUMBNAIL_SIZE},
        )
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPError as e:
        print(f"  [warn] Could not download thumbnail for {asset_id}: {e}")
        return None


def check_connection() -> dict:
    """Ping the Immich server and return server info."""
    with httpx.Client(headers=HEADERS, timeout=10) as client:
        resp = client.get(f"{IMMICH_URL}/api/server/about")
        resp.raise_for_status()
        return resp.json()
