"""Build a local LanceDB index for the RAG application."""

from __future__ import annotations

import os

import lancedb
from fastembed import TextEmbedding


# =========================================================
# CONFIG
# =========================================================

DB_PATH = "data/lancedb"

TABLE_NAME = "msmarco_xi"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# =========================================================
# LOCAL SAMPLE DATA
# =========================================================

SAMPLE_DATA = [
    {
        "text": "Machine learning is a branch of artificial intelligence that allows computers to learn patterns from data.",
        "parent_context": "Machine learning is a branch of artificial intelligence that allows computers to learn patterns from data.",
        "source_id": "1",
        "source_title": "Machine Learning",
        "child_index": 0,
    },
    {
        "text": "Supervised learning uses labeled training data to learn a mapping between inputs and outputs.",
        "parent_context": "Supervised learning uses labeled training data to learn a mapping between inputs and outputs.",
        "source_id": "2",
        "source_title": "Supervised Learning",
        "child_index": 0,
    },
    {
        "text": "Unsupervised learning finds patterns or structures in data without using labeled outputs.",
        "parent_context": "Unsupervised learning finds patterns or structures in data without using labeled outputs.",
        "source_id": "3",
        "source_title": "Unsupervised Learning",
        "child_index": 0,
    },
    {
        "text": "Deep learning uses neural networks with multiple layers to learn complex representations from data.",
        "parent_context": "Deep learning uses neural networks with multiple layers to learn complex representations from data.",
        "source_id": "4",
        "source_title": "Deep Learning",
        "child_index": 0,
    },
    {
        "text": "Natural language processing enables computers to process and understand human language.",
        "parent_context": "Natural language processing enables computers to process and understand human language.",
        "source_id": "5",
        "source_title": "Natural Language Processing",
        "child_index": 0,
    },
    {
        "text": "Retrieval augmented generation combines information retrieval with a language model to produce answers using external information.",
        "parent_context": "Retrieval augmented generation combines information retrieval with a language model to produce answers using external information.",
        "source_id": "6",
        "source_title": "RAG",
        "child_index": 0,
    },
    {
        "text": "Vector databases store numerical vector representations of information and support similarity search.",
        "parent_context": "Vector databases store numerical vector representations of information and support similarity search.",
        "source_id": "7",
        "source_title": "Vector Databases",
        "child_index": 0,
    },
    {
        "text": "Embeddings convert text into numerical vectors so that semantically similar text can be compared.",
        "parent_context": "Embeddings convert text into numerical vectors so that semantically similar text can be compared.",
        "source_id": "8",
        "source_title": "Text Embeddings",
        "child_index": 0,
    },
    {
        "text": "FastAPI is a Python framework for building APIs using modern Python type hints.",
        "parent_context": "FastAPI is a Python framework for building APIs using modern Python type hints.",
        "source_id": "9",
        "source_title": "FastAPI",
        "child_index": 0,
    },
    {
        "text": "LanceDB is a vector database that can be used to store embeddings and perform similarity searches.",
        "parent_context": "LanceDB is a vector database that can be used to store embeddings and perform similarity searches.",
        "source_id": "10",
        "source_title": "LanceDB",
        "child_index": 0,
    },
]


# =========================================================
# CREATE LANCEDB INDEX
# =========================================================

def main():

    print()
    print("=" * 60)
    print("LOCAL DATA → LANCEDB")
    print("=" * 60)

    print()
    print(
        f"Using {len(SAMPLE_DATA)} local documents."
    )

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("CREATING EMBEDDINGS")
    print("=" * 60)

    texts = [
        record["text"]
        for record in SAMPLE_DATA
    ]

    print(
        f"Embedding {len(texts)} documents..."
    )

    print(
        "Loading embedding model..."
    )

    embedder = TextEmbedding(
        model_name=MODEL_NAME
    )

    vectors = list(
        embedder.embed(
            texts,
            batch_size=32,
            parallel=4
        )
    )

    print(
        f"Created {len(vectors)} embeddings."
    )

    # -----------------------------------------------------
    # PREPARE DATABASE ROWS
    # -----------------------------------------------------

    db_rows = []

    for record, vector in zip(
        SAMPLE_DATA,
        vectors
    ):

        db_rows.append(
            record
            | {
                "vector": vector.tolist()
            }
        )

    # -----------------------------------------------------
    # CREATE DATABASE DIRECTORY
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("CREATING LANCEDB TABLE")
    print("=" * 60)

    os.makedirs(
        "data",
        exist_ok=True
    )

    db = lancedb.connect(
        DB_PATH
    )

    # -----------------------------------------------------
    # REMOVE OLD TABLE
    # -----------------------------------------------------

    if TABLE_NAME in db.table_names():

        print(
            "Removing existing table..."
        )

        db.drop_table(
            TABLE_NAME
        )

    # -----------------------------------------------------
    # CREATE TABLE
    # -----------------------------------------------------

    db.create_table(
        TABLE_NAME,
        data=db_rows
    )

    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(
        f"Table: {TABLE_NAME}"
    )

    print(
        f"Indexed documents: "
        f"{len(db_rows)}"
    )

    print(
        f"Database: {DB_PATH}"
    )

    print()
    print(
        "LanceDB retrieval is ready."
    )

    print("=" * 60)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()