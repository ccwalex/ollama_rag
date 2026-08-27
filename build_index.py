#!/usr/bin/env python3
"""Build RAG index artifacts consumed by the telegram bots.

Pipeline trace
--------------
ref_db/ (PDFs)
    -> long_chunks.pkl, short_chunks.pkl          [chunk]
    -> qdrant/ + nomic_multicore collection       [qdrant]  -> telebot_v1_mistral.py
    -> long/short sklearn indexes + hdf5 stores   [hdf5]    -> telebot_v3.py
    -> UMAP reducer + reduced NN index            [umap]    -> telebot_v4_umap_nomic.py

Example:
    python build_index.py chunk
    python build_index.py qdrant
    python telebot_v1_mistral.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_PATH = Path("ref_db")
LONG_CHUNKS_PATH = Path("long_chunks.pkl")
SHORT_CHUNKS_PATH = Path("short_chunks.pkl")
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


def load_documents():
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_core.documents import Document
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH}/ not found. Add PDFs there before building the index."
        )
    loader = DirectoryLoader(
        str(DATA_PATH),
        glob=["*.pdf", "*.nxml"],
        silent_errors=True,
        show_progress=True,
        use_multithreading=True,
        max_concurrency=12,
    )
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks (size={chunk_size}).")
    return chunks


def chunk_documents():
    import joblib
    documents = load_documents()
    long_chunks = split_documents(documents, chunk_size=750, chunk_overlap=200)
    short_chunks = split_documents(documents, chunk_size=250, chunk_overlap=50)
    joblib.dump(long_chunks, LONG_CHUNKS_PATH)
    joblib.dump(short_chunks, SHORT_CHUNKS_PATH)
    print(f"Wrote {LONG_CHUNKS_PATH} ({len(long_chunks)} chunks)")
    print(f"Wrote {SHORT_CHUNKS_PATH} ({len(short_chunks)} chunks)")
    return long_chunks, short_chunks


def load_chunk_pair():
    import joblib
    if not LONG_CHUNKS_PATH.exists() or not SHORT_CHUNKS_PATH.exists():
        return chunk_documents()
    long_chunks = joblib.load(LONG_CHUNKS_PATH)
    short_chunks = joblib.load(SHORT_CHUNKS_PATH)
    print(f"Loaded {len(long_chunks)} long chunks and {len(short_chunks)} short chunks")
    return long_chunks, short_chunks


def embed_texts(texts):
    import numpy as np
    import ollama
    from tqdm import tqdm
    vectors = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, text in enumerate(tqdm(texts, desc="embedding")):
        vectors[i] = np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"], dtype=np.float32)
    return vectors


def chunk_texts(chunks):
    import json
    return [json.dumps(chunk.page_content) for chunk in chunks]


def build_qdrant(long_chunks):
    from uuid import uuid4

    from langchain_ollama import OllamaEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    client = QdrantClient(path="qdrant")
    collection_name = "nomic_multicore"

    if collection_name not in {c.name for c in client.get_collections().collections}:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    ids = [str(uuid4()) for _ in range(len(long_chunks))]
    vector_store.add_documents(documents=long_chunks, ids=ids)
    print(f"Qdrant collection '{collection_name}' ready under qdrant/")


def write_hdf5(path: Path, dataset_name: str, strings):
    import h5py
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            dataset_name,
            (len(strings),),
            maxshape=(None,),
            dtype=h5py.string_dtype(),
        )
        dataset[:] = strings


def fit_search(vectors, output_path: Path):
    import joblib
    from sklearn.neighbors import NearestNeighbors
    search = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="cosine")
    search.fit(vectors)
    joblib.dump(search, output_path)
    print(f"Wrote {output_path}")
    return search


def build_hdf5(long_chunks, short_chunks):
    long_strings = chunk_texts(long_chunks)
    short_strings = chunk_texts(short_chunks)

    long_vectors = embed_texts(long_strings)
    short_vectors = embed_texts(short_strings)

    fit_search(long_vectors, Path("long_search_sk_nomic.pkl"))
    fit_search(short_vectors, Path("short_search_sk_nomic.pkl"))

    write_hdf5(Path("nomic_db.hdf5"), "long_chunks", long_strings)
    write_hdf5(Path("nomic_short.hdf5"), "short_chunks", short_strings)
    print("HDF5 stores ready for telebot_v3.py")


def build_umap(long_chunks, short_chunks):
    build_hdf5(long_chunks, short_chunks)

    from sklearn.model_selection import train_test_split
    from umap import UMAP

    short_strings = chunk_texts(short_chunks)
    short_vectors = embed_texts(short_strings)

    reducer = UMAP(n_components=50, n_jobs=-1)
    sample, _ = train_test_split(short_vectors, test_size=0.95, random_state=42)
    reducer.fit(sample)
    reduced = reducer.transform(short_vectors)

    search = NearestNeighbors(n_neighbors=8, algorithm="auto", metric="cosine")
    search.fit(reduced)
    joblib.dump(reducer, "short_umap.pkl")
    joblib.dump(search, "short_umap_search.pkl")
    print("UMAP artifacts ready for telebot_v4_umap_nomic.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=["chunk", "qdrant", "hdf5", "umap", "all"],
        help="chunk: PDFs -> pkl | qdrant/hdf5/umap: pkl -> bot artifacts | all: full v1 stack",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.step == "chunk":
        chunk_documents()
        return

    long_chunks, short_chunks = load_chunk_pair()

    if args.step == "qdrant":
        build_qdrant(long_chunks)
    elif args.step == "hdf5":
        build_hdf5(long_chunks, short_chunks)
    elif args.step == "umap":
        build_umap(long_chunks, short_chunks)
    elif args.step == "all":
        build_qdrant(long_chunks)
        build_hdf5(long_chunks, short_chunks)
        build_umap(long_chunks, short_chunks)


if __name__ == "__main__":
    main()
