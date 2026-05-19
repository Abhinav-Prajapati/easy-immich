import os
import httpx
from datetime import datetime
import numpy as np
from sklearn.cluster import DBSCAN
from src.vector_store import VectorStore
from src.config import (
    IMMICH_URL,
    IMMICH_API_KEY,
    TIME_THRESHOLD_HOURS,
    VISUAL_TOLERANCE_EPS,
    MIN_SAMPLES,
    OUTPUT_DIR,
)

def fetch_and_sort_data():
    """Fetches all records from ChromaDB and sorts them chronologically."""
    print("Connecting to ChromaDB...")
    store = VectorStore()
    data = store.collection.get(include=["metadatas", "embeddings"])

    ids, metadatas, embeddings = data["ids"], data["metadatas"], data["embeddings"]
    if not ids:
        print("No data found in ChromaDB. Have you run main.py?")
        return []

    print(f"  → Loaded {len(ids)} records. Parsing timestamps...")
    records = []
    for i in range(len(ids)):
        dt_str = metadatas[i].get("localDateTime")
        if not dt_str:
            continue
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            records.append({
                "id": ids[i],
                "timestamp": dt.timestamp(),
                "datetime": dt,
                "embedding": np.array(embeddings[i]),
                "metadata": metadatas[i]
            })
        except ValueError:
            pass

    records.sort(key=lambda x: x["timestamp"])
    return records

def cluster_temporal(records):
    """Phase 1: Groups chronological records into Events based on time gaps."""
    if not records:
        return []

    print(f"\n--- Phase 1: Temporal Clustering (> {TIME_THRESHOLD_HOURS}h gap) ---")
    threshold_seconds = TIME_THRESHOLD_HOURS * 3600
    events = []
    current_event = []

    for i, record in enumerate(records):
        if i == 0:
            current_event.append(record)
            continue

        time_diff = record["timestamp"] - records[i - 1]["timestamp"]
        if time_diff > threshold_seconds:
            events.append(current_event)
            current_event = [record]
        else:
            current_event.append(record)

    if current_event:
        events.append(current_event)

    print(f"  → Generated {len(events)} distinct time-based events.")
    return events

def cluster_visual(events):
    """Phase 2: Splits Events into Scenes using AI vision embeddings."""
    print(f"\n--- Phase 2: AI Visual Clustering (EPS: {VISUAL_TOLERANCE_EPS}) ---")
    all_scenes = []

    for event_idx, event_cluster in enumerate(events):
        event_id = event_idx + 1

        if len(event_cluster) == 1:
            all_scenes.append({
                "event_id": event_id,
                "scene_id": 1,
                "images": event_cluster
            })
            continue

        embeddings = np.array([img["embedding"] for img in event_cluster])
        clustering = DBSCAN(
            eps=VISUAL_TOLERANCE_EPS, 
            min_samples=MIN_SAMPLES, 
            metric="cosine"
        ).fit(embeddings)

        labels = clustering.labels_
        unique_labels = set(labels)

        print(f"  Event {event_id:02d} ({len(event_cluster)} photos) -> Split into {len(unique_labels)} visual scenes.")

        for scene_label in unique_labels:
            scene_images = [img for i, img in enumerate(event_cluster) if labels[i] == scene_label]
            all_scenes.append({
                "event_id": event_id,
                "scene_id": scene_label + 1,
                "images": scene_images
            })

    return all_scenes

def export_to_folders(scenes):
    """Phase 3: Downloads the images into nested Event/Scene folders, or root if standalone."""
    print(f"\n--- Phase 3: Exporting to '{OUTPUT_DIR}/' ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with httpx.Client(
        headers={"x-api-key": IMMICH_API_KEY},
        timeout=30,
        follow_redirects=True
    ) as client:

        for scene in scenes:
            event_id = scene["event_id"]
            scene_id = scene["scene_id"]
            images = scene["images"]

            start_dt = images[0]["datetime"].strftime('%Y-%m-%d_%H%M')

            if len(images) == 1:
                record = images[0]
                asset_id = record["id"]
                orig_name = record["metadata"].get("fileName", "image")

                safe_name = f"Standalone_Event_{event_id:02d}_{start_dt}_{orig_name}.jpg"
                file_path = os.path.join(OUTPUT_DIR, safe_name)

                print(f"  Downloading   1 image  -> Root (Standalone)")

                if not os.path.exists(file_path):
                    try:
                        resp = client.get(
                            f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail",
                            params={"size": "preview"}
                        )
                        resp.raise_for_status()
                        with open(file_path, "wb") as f:
                            f.write(resp.content)
                    except Exception as e:
                        print(f"    [warn] Failed to download {safe_name}: {e}")

            else:
                folder_name = f"Event_{event_id:02d}_{start_dt}_Scene_{scene_id:02d}"
                folder_path = os.path.join(OUTPUT_DIR, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                print(f"  Downloading {len(images):>3} images -> {folder_name}/")

                for img_idx, record in enumerate(images):
                    asset_id = record["id"]
                    orig_name = record["metadata"].get("fileName", "image")
                    safe_name = f"{img_idx + 1:03d}_{orig_name}.jpg"
                    file_path = os.path.join(folder_path, safe_name)

                    if os.path.exists(file_path):
                        continue

                    try:
                        resp = client.get(
                            f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail",
                            params={"size": "preview"}
                        )
                        resp.raise_for_status()
                        with open(file_path, "wb") as f:
                            f.write(resp.content)
                    except Exception as e:
                        print(f"    [warn] Failed to download {safe_name}: {e}")

    print("\n✓ Pipeline complete! Open the review folder to check your clusters.")

if __name__ == "__main__":
    records = fetch_and_sort_data()
    if records:
        events = cluster_temporal(records)
        scenes = cluster_visual(events)
        export_to_folders(scenes)
