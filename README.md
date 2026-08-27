# ollama_rag

Telegram RAG bot powered by local Ollama models. Ingest PDFs, build a retrieval index, run a bot.

## Product pipeline

```text
ref_db/*.pdf
    |
    v
build_index.py chunk  -->  long_chunks.pkl, short_chunks.pkl
    |
    +-- build_index.py qdrant  -->  qdrant/  -------------------->  telebot_v1_mistral.py
    |
    +-- build_index.py hdf5    -->  *.pkl + nomic_*.hdf5  ------>  telebot_v3.py
    |
    +-- build_index.py umap    -->  short_umap*.pkl + hdf5  --->  telebot_v4_umap_nomic.py
```

The notebooks (`create_database.ipynb`, `hdf5_search.ipynb`, `qdrant.ipynb`) are the original exploratory versions of the same steps. `build_index.py` is the runnable script that produces the artifacts each bot expects.

## Quick start

1. Put source PDFs in `ref_db/`.
2. Install deps: `pip install -r requirements.txt`
3. Pull models: `ollama pull nomic-embed-text` and the LLM your bot uses (e.g. `ollama pull mistral-nemo` for v1).
4. Build index for your bot version:

```bash
python build_index.py chunk
python build_index.py qdrant   # for v1
# or
python build_index.py hdf5     # for v3
# or
python build_index.py umap     # for v4
```

5. Set your bot token and run:

```bash
export TELEGRAM_BOT_TOKEN=your_token
python telebot_v1_mistral.py
```

## Bots

| version | script | index backend | LLM |
|---------|--------|---------------|-----|
| v1 | `telebot_v1_mistral.py` | Qdrant + LangChain | mistral-nemo |
| v3 | `telebot_v3.py` | sklearn NN + hdf5, double RAG | mistral-small + phi4 |
| v4 | `telebot_v4_umap_nomic.py` | UMAP + sklearn NN + hdf5 | phi4 |

v4 commands: `/long` (chain-of-thought), `/short` (fast), `/clear` (reset session).

## Artifacts

| file | produced by | consumed by |
|------|-------------|-------------|
| `long_chunks.pkl` | chunk | qdrant, hdf5, umap |
| `short_chunks.pkl` | chunk | hdf5, umap |
| `qdrant/` | qdrant | v1 |
| `long_search_sk_nomic.pkl` | hdf5 | v3 |
| `short_search_sk_nomic.pkl` | hdf5 | v3 |
| `nomic_db.hdf5` | hdf5 | v3 |
| `nomic_short.hdf5` | hdf5 | v3, v4 |
| `short_umap.pkl` | umap | v4 |
| `short_umap_search.pkl` | umap | v4 |

## Notes

- CUDA helps for embedding large corpora but is not required.
- v3/v4 keep conversation state under `telebot_v3_cache/` and `telebot_v4_cache/`.
- Original notebooks may have interrupted runs; prefer `build_index.py` for a clean build.
