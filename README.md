# ollama_rag
my implementation of a telegram bot powered by local RAG with ollama and langchain

some jupyter notebooks have some interrupted errors or changes that are not consistent


dependency:
+ ollama
+ langchain
+ unstructured
+ cuda (of course)
+ python / conda / jupyter

vector database:
+ qdrant (or your own choice)

telegram:
+ telebot

## bots

| version | file | description |
|---------|------|-------------|
| v1 | `telebot_v1_mistral.py` | Qdrant + LangChain retrieval with mistral-nemo |
| v3 | `telebot_v3.py` | long-context double RAG using sklearn and hdf5 |
| v4 | `telebot_v4_umap_nomic.py` | UMAP-reduced nearest-neighbor retrieval with nomic embeddings |

## notebooks

| file | purpose |
|------|---------|
| `create_database.ipynb` | build document chunks and vector stores |
| `hdf5_search.ipynb` | embed chunks and build sklearn/hdf5 retrieval artifacts |
| `qdrant.ipynb` | load chunks into qdrant |

## artifact naming

- `long_chunks.pkl` / `short_chunks.pkl` — document chunk stores
- `nomic_db.hdf5` / `nomic_short.hdf5` — hdf5 text stores
- `long_search_sk_nomic.pkl` / `short_search_sk_nomic.pkl` — sklearn nearest-neighbor indexes (v3)
- `short_umap.pkl` / `short_umap_search.pkl` — UMAP reducer and nearest-neighbor index (v4)
- `telebot_v3_cache/` / `telebot_v4_cache/` — per-version conversation state

original version using phi4 has surprisingly good performance even with shorter context window, can be used with lower vram

V4 added, using UMAP with nearest neighbor allows much faster retrieval and lower memory use (~95% reduction in my use)
