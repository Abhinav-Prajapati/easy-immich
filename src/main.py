import httpx
from tqdm import tqdm
from src import config
from src.immich_client import fetch_assets, download_thumbnail, check_connection
from src.embedder import DinoEmbedder
from src.vector_store import VectorStore

BATCH_SIZE = 16

def run():

    if not config.IMMICH_URL or not config.IMMICH_API_KEY:
        print("  [error] Missing configuration!")
        print("  Please make sure you have copied .env.example to .env and set IMMICH_URL and IMMICH_API_KEY.")
        return

    print("Connecting to Immich...")
    try:
        info = check_connection()
        print(f"  → Connected: Immich v{info.get('version', '?')}")
    except Exception as e:
        print(f"  [error] Cannot reach Immich: {e}")
        print("  Check IMMICH_URL and IMMICH_API_KEY in your .env file.")
        return

    embedder = DinoEmbedder()
    print(f"  → Embedding dim: {embedder.embedding_dim}")

    store = VectorStore()

    print(f"\nFetching assets from Immich for year {config.TARGET_YEAR}...")
    all_assets = list(fetch_assets(config.TARGET_YEAR))
    print(f"  → Found {len(all_assets)} images")

    if not all_assets:
        print("No images found. Check your date range and API key permissions.")
        return

    all_ids = [a["id"] for a in all_assets]
    already_done = store.already_embedded(all_ids)
    pending = [a for a in all_assets if a["id"] not in already_done]

    print(f"  → {len(already_done)} already embedded, {len(pending)} to process")

    if not pending:
        print("Everything already embedded. Nothing to do.")
        return

    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    embedded_count = 0
    failed_count = 0

    with httpx.Client(
        headers={"x-api-key": config.IMMICH_API_KEY},
        timeout=30,
    ) as http_client:

        for batch_start in tqdm(
            range(0, len(pending), BATCH_SIZE),
            total=total_batches,
            desc="Embedding",
            unit="batch",
        ):
            batch = pending[batch_start : batch_start + BATCH_SIZE]

            thumbnails = [
                download_thumbnail(asset["id"], http_client)
                for asset in batch
            ]

            valid_pairs = [
                (asset, thumb)
                for asset, thumb in zip(batch, thumbnails)
                if thumb is not None
            ]

            if not valid_pairs:
                failed_count += len(batch)
                continue

            valid_assets, valid_thumbs = zip(*valid_pairs)

            embeddings = embedder.embed_batch(list(valid_thumbs))

            good_ids, good_embs, good_metas = [], [], []
            for asset, emb in zip(valid_assets, embeddings):
                if emb is not None:
                    good_ids.append(asset["id"])
                    good_embs.append(emb)
                    good_metas.append(asset)
                else:
                    failed_count += 1

            if good_ids:
                store.upsert_batch(good_ids, good_embs, good_metas)
                embedded_count += len(good_ids)

    print(f"\n✓ Done.")
    print(f"  Embedded this run : {embedded_count}")
    print(f"  Failed            : {failed_count}")
    print(f"  Total in DB       : {store.count()}")

if __name__ == "__main__":
    run()
