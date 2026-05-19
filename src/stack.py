"""
Duplicate Stacking Pipeline
Finds near-exact duplicate images using strict DINOv2 clustering and 
automatically merges them into Stacks in your Immich library.
"""

import httpx
from datetime import datetime
import numpy as np
from sklearn.cluster import DBSCAN
from src.vector_store import VectorStore
from src.config import IMMICH_URL, IMMICH_API_KEY

TIME_THRESHOLD_HOURS = 2.0   # Bursts and duplicates happen close together
VISUAL_TOLERANCE_EPS = 0.25  # Strict! 0.10-0.15 ensures only near-duplicates match
MIN_SAMPLES = 1              

def fetch_and_sort_data():
    print("Connecting to ChromaDB...")
    store = VectorStore()
    data = store.collection.get(include=["metadatas", "embeddings"])
    
    ids, metadatas, embeddings = data["ids"], data["metadatas"], data["embeddings"]
    if not ids:
        print("No data found in ChromaDB.")
        return []

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
                "embedding": np.array(embeddings[i]),
                "metadata": metadatas[i]
            })
        except ValueError:
            pass

    records.sort(key=lambda x: x["timestamp"])
    return records


def cluster_temporal(records):
    """Phase 1: Group by narrow time gaps."""
    if not records:
        return []

    threshold_seconds = TIME_THRESHOLD_HOURS * 3600
    events = []
    current_event = []

    for i, record in enumerate(records):
        if i == 0:
            current_event.append(record)
            continue
        
        if record["timestamp"] - records[i - 1]["timestamp"] > threshold_seconds:
            events.append(current_event)
            current_event = [record]
        else:
            current_event.append(record)
            
    if current_event:
        events.append(current_event)

    return events


def cluster_visual(events):
    """Phase 2: Strict visual clustering to find exact scenes/bursts."""
    print(f"--- Running Strict AI Clustering (EPS: {VISUAL_TOLERANCE_EPS}) ---")
    all_scenes = []
    
    for event_idx, event_cluster in enumerate(events):
        if len(event_cluster) == 1:
            continue # We don't care about standalone photos for stacking

        embeddings = np.array([img["embedding"] for img in event_cluster])
        clustering = DBSCAN(
            eps=VISUAL_TOLERANCE_EPS, 
            min_samples=MIN_SAMPLES, 
            metric="cosine"
        ).fit(embeddings)

        labels = clustering.labels_
        unique_labels = set(labels)

        for scene_label in unique_labels:
            scene_images = [img for i, img in enumerate(event_cluster) if labels[i] == scene_label]
            # Only keep scenes that have multiple images (duplicates!)
            if len(scene_images) > 1:
                all_scenes.append({
                    "event_id": event_idx + 1,
                    "scene_id": scene_label + 1,
                    "images": scene_images
                })
            
    return all_scenes


def push_stacks_to_immich(scenes):
    """Phase 3: Hit the Immich /stacks API."""
    if not scenes:
        print("\nNo duplicates found to stack!")
        return

    print(f"\n--- Phase 3: Creating {len(scenes)} Stacks in Immich ---")

    headers = {
        "x-api-key": IMMICH_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    stacks_created = 0
    
    with httpx.Client(headers=headers, timeout=30) as client:
        for scene in scenes:
            images = scene["images"]
            
            # Sort chronologically. The API makes the first ID the "Cover" of the stack.
            images.sort(key=lambda x: x["timestamp"])
            asset_ids = [img["id"] for img in images]
            
            print(f"  → Stacking {len(asset_ids)} near-duplicate images...")
            
            try:
                payload = {"assetIds": asset_ids}
                # Immich API routes are prefixed with /api
                resp = client.post(f"{IMMICH_URL}/api/stacks", json=payload)
                resp.raise_for_status()
                stacks_created += 1
            except Exception as e:
                print(f"    [error] Failed to create stack: {e}")
                # Print the Immich error message if available
                if hasattr(e, 'response') and e.response is not None:
                     print(f"    Details: {e.response.text}")
                     
    print(f"\n✓ Done! Successfully created {stacks_created} stacks.")


if __name__ == "__main__":
    records = fetch_and_sort_data()
    if records:
        events = cluster_temporal(records)
        scenes = cluster_visual(events)
        push_stacks_to_immich(scenes)