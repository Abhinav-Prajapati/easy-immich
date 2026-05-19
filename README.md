# 🌟 Immich Clustering & Embedding Pipeline

Fetch images from your [Immich](https://immich.app/) instance, extract high-quality vision embeddings using Meta AI's **DINOv2** model, and store them locally in a **ChromaDB** vector database. 

This project acts as the foundational first step toward visual clustering, duplicate detection, and automated smart album generation for your personal photo library.

---

## 🚀 Setup & Installation

### 1. Prerequisite: Install `uv`
This project utilizes [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver. If you don't have it installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone & Prepare Environment
Clone the repository and copy the environment template:
```bash
# Copy the env template
cp .env.example .env
```

### 3. Configure `.env`
Edit the `.env` file and fill in your details:
*   `IMMICH_URL` — Address of your Immich instance (e.g., `http://192.168.1.10:2283`).
*   `IMMICH_API_KEY` — Your secret Immich API key (see [How to Create an Immich API Key](#key-immich) below).
*   `DINO_MODEL` — The size of the vision transformer model (see [Model Trade-offs](#model-sizes)).

### 4. Run the Pipeline
`uv` will automatically set up a virtual environment, install the required dependencies (including PyTorch, Hugging Face Transformers, and ChromaDB), and run the ingestion pipeline:
```bash
uv run python -m src.main
```

---

## 🔑 Key & Authentication Guides

<a id="key-immich"></a>
### 1. How to Generate an Immich API Key

Immich uses API keys to allow external applications to interact securely with your photo library. Follow these steps to generate one:

1. Open your **Immich Web UI** in a browser (e.g., `http://your-immich-ip:2283`).
2. In the top-right corner, click on your **User Profile Picture** or initials, then select **Account Settings**.
3. In the sidebar of the settings modal that appears, click on the **API Keys** section.
4. Click the **Create API Key** button on the right.
5. Provide a friendly name for your key (e.g., `easy-immich-clustering`) so you remember what it is used for.
6. Click **Create**.
7. **Important:** Copy the generated API key immediately. Immich will *never* show it to you again.
8. Paste this key into your `.env` file as `IMMICH_API_KEY`.

---

<a id="key-dino"></a>
### 2. How to Get an API Key for DINOv2

**Short Answer:** You don't need one! 🎉

*   **Meta's DINOv2** is a completely open-source, public self-supervised vision transformer model.
*   It is hosted publicly on **Hugging Face** under Meta's account (e.g., `facebook/dinov2-base`).
*   The Hugging Face `transformers` library will **automatically download and cache the model** to your local machine the very first time you run the script. No sign-ups, accounts, or API tokens are required.
*   **Optional HF Token:** If you are running this in a highly firewalled environment, hitting rate limits, or want to use private Hugging Face models, you can log in to Hugging Face or set a Hugging Face token in your environment:
    ```bash
    export HF_TOKEN="your_hugging_face_token_here"
    ```
    Again, this is **completely optional** and not needed for standard DINOv2 runs.

---

## ⚙️ Configuration Reference (.env)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IMMICH_URL` | *Required* | The full URL of your Immich instance (without trailing slash). |
| `IMMICH_API_KEY` | *Required* | The API key generated from the Immich Web UI. |
| `DINO_MODEL` | `base` | The size/power of the model: `small`, `base`, `large`, or `giant`. |
| `TARGET_YEAR` | `2026` | The specific year of photos to ingest and embed. |
| `CHROMA_PATH` | `./chroma_db`| Local folder where the ChromaDB vector database will be stored. |

---

<a id="model-sizes"></a>
## 🧠 DINOv2 Model Size Trade-offs

| Size | Embedding Dimension | Speed (CPU) | GPU Required? | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`small`** | `384` | Very Fast | No (works great on CPU) | Low-resource setups, quick testing |
| **`base`** | `768` | Moderate | No (recommended on GPU) | **Standard use-case (best balance)** |
| **`large`** | `1024` | Slow | Yes (highly recommended) | High-accuracy clustering and search |
| **`giant`** | `1536` | Extremely Slow | Yes (mandatory) | Maximum precision retrieval tasks |

---

## 🛡️ Robust & Incremental Execution

*   **Idempotency & Re-runs:** Running the pipeline multiple times is entirely safe! Already-embedded assets are tracked in ChromaDB and automatically skipped on subsequent runs.
*   **Error Tolerance:** If a thumbnail download fails or an image is corrupted, the script logs a warning, skips the single file, and continues processing the rest of the batch without crashing.
*   **Resource Friendly:** Images are processed in configurable batches to prevent out-of-memory (OOM) issues on CPU or low-VRAM GPUs.

---

## 🗺️ Next Steps & Roadmap

Once your images are embedded into ChromaDB, you can run downstream analysis tasks:
- **`cluster.py`** — Apply algorithms like HDBSCAN or K-Means on the stored embeddings to group similar images.
- **`dedup.py`** — Detect and report near-duplicate photos.
- **`albums.py`** — Push the discovered clusters back to Immich as curated albums.
