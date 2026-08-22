"""Safety, retrieval, grounded generation, and retry orchestration."""

from __future__ import annotations

import json
import os
import re
import time

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import lancedb
from fastembed import TextEmbedding
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# CONFIG
# =========================================================

REFUSAL = "I cannot answer based on the provided documents."

DB_PATH = Path("data/lancedb")

TABLE_NAME = "msmarco_xi"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

INJECTION_PATTERNS = re.compile(
    r"ignore (all|any|previous)|system prompt|"
    r"developer message|reveal your instructions|"
    r"jailbreak|dan mode",
    re.I
)


# =========================================================
# CLIENTS
# =========================================================

EMBEDDER = TextEmbedding(
    model_name=MODEL_NAME
)

VECTOR_DB = lancedb.connect(
    str(DB_PATH)
)

VECTOR_TABLE = (
    VECTOR_DB.open_table(TABLE_NAME)
    if TABLE_NAME in VECTOR_DB.table_names()
    else None
)

GROQ_CLIENT = (
    Groq(
        api_key=os.environ["GROQ_API_KEY"]
    )
    if os.getenv("GROQ_API_KEY")
    else None
)


# =========================================================
# RESPONSE MODEL
# =========================================================

class RagResponse(BaseModel):

    answer: str

    is_grounded: bool

    confidence: float = Field(
        ge=0,
        le=1
    )

    citations: list[str] = Field(
        default_factory=list
    )


# =========================================================
# RETRIEVAL MODEL
# =========================================================

@dataclass
class Retrieval:

    context: str

    citations: list[str]

    distance: float


# =========================================================
# RETRY
# =========================================================

def retry(
    operation: Callable[[], Any],
    attempts: int = 3,
    base_delay: float = 0.5
) -> Any:

    for attempt in range(attempts):

        try:

            return operation()

        except Exception:

            if attempt == attempts - 1:

                raise

            time.sleep(
                base_delay * (2 ** attempt)
            )


# =========================================================
# RAG HARNESS
# =========================================================

class RagHarness:

    def __init__(
        self,
        db_path: Path = DB_PATH,
        distance_threshold: float = 0.85
    ):

        self.embedder = EMBEDDER

        self.db = VECTOR_DB

        self.table = VECTOR_TABLE

        self.distance_threshold = distance_threshold

        self.client = GROQ_CLIENT

        print()
        print("=" * 60)
        print("RAG HARNESS INITIALIZED")
        print("=" * 60)

        print(
            f"Database: {db_path}"
        )

        print(
            f"Table available: "
            f"{self.table is not None}"
        )

        print(
            f"Groq configured: "
            f"{self.client is not None}"
        )

        print(
            f"Groq model: {GROQ_MODEL}"
        )

        print("=" * 60)


    # =====================================================
    # RETRIEVAL
    # =====================================================

    @lru_cache(maxsize=512)
    def retrieve(
        self,
        query: str,
        limit: int = 10
    ) -> Retrieval:

        print()
        print(
            f"[RETRIEVAL] Query: {query}"
        )

        if self.table is None:

            print(
                "[RETRIEVAL] ERROR: "
                "LanceDB table is not available."
            )

            return Retrieval(
                "",
                [],
                float("inf")
            )

        # -------------------------------------------------
        # CREATE QUERY EMBEDDING
        # -------------------------------------------------

        vector = next(
            self.embedder.embed(
                [query],
                batch_size=32
            )
        ).tolist()

        print(
            "[RETRIEVAL] Query embedding created."
        )

        # -------------------------------------------------
        # SEARCH
        # -------------------------------------------------

        results = (
            self.table
            .search(vector)
            .metric("cosine")
            .limit(limit)
            .to_list()
        )

        print(
            f"[RETRIEVAL] Raw results: "
            f"{len(results)}"
        )

        # -------------------------------------------------
        # DEBUG RESULTS
        # -------------------------------------------------

        for index, row in enumerate(results):

            print(
                f"[RETRIEVAL] Result {index + 1}: "
                f"distance={row.get('_distance')}"
            )

            print(
                f"[RETRIEVAL] Text: "
                f"{row.get('text', '')[:150]}"
            )

        # -------------------------------------------------
        # FILTER BY DISTANCE
        # -------------------------------------------------

        filtered_results = [
            row
            for row in results
            if float(
                row.get(
                    "_distance",
                    999
                )
            ) <= self.distance_threshold
        ]

        print(
            f"[RETRIEVAL] Results after "
            f"threshold: {len(filtered_results)}"
        )

        if not filtered_results:

            print(
                "[RETRIEVAL] No relevant documents found."
            )

            return Retrieval(
                "",
                [],
                float("inf")
            )

        # -------------------------------------------------
        # BUILD CONTEXT
        # -------------------------------------------------

        context = "\n\n".join(
            row["parent_context"]
            for row in filtered_results
        )

        citations = sorted(
            {
                f"MSMARCO-XI:{row['source_id']}"
                for row in filtered_results
            }
        )

        distance = min(
            float(
                row["_distance"]
            )
            for row in filtered_results
        )

        print(
            f"[RETRIEVAL] Final distance: "
            f"{distance}"
        )

        print(
            f"[RETRIEVAL] Citations: "
            f"{citations}"
        )

        print(
            f"[RETRIEVAL] Context length: "
            f"{len(context)}"
        )

        return Retrieval(
            context,
            citations,
            distance
        )


    # =====================================================
    # ANSWER
    # =====================================================

    def answer(
        self,
        query: str,
        retrieval: Retrieval | None = None
    ) -> RagResponse:

        print()
        print(
            "[ANSWER] Starting answer generation..."
        )

        # -------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------

        if INJECTION_PATTERNS.search(query):

            print(
                "[ANSWER] Refused because of "
                "injection pattern."
            )

            return RagResponse(
                answer=REFUSAL,
                is_grounded=False,
                confidence=0,
                citations=[]
            )

        # -------------------------------------------------
        # RETRIEVE
        # -------------------------------------------------

        retrieval = (
            retrieval
            or self.retrieve(
                query,
                limit=10
            )
        )

        # -------------------------------------------------
        # CHECK CONTEXT
        # -------------------------------------------------

        if not retrieval.context:

            print(
                "[ANSWER] No retrieval context."
            )

            return RagResponse(
                answer=REFUSAL,
                is_grounded=False,
                confidence=0,
                citations=[]
            )

        # -------------------------------------------------
        # CHECK GROQ
        # -------------------------------------------------

        if self.client is None:

            print(
                "[ANSWER] ERROR: GROQ_API_KEY "
                "is not configured."
            )

            return RagResponse(
                answer=(
                    "GROQ_API_KEY is not configured."
                ),
                is_grounded=False,
                confidence=0,
                citations=[]
            )

        print(
            "[ANSWER] Context found."
        )

        print(
            "[ANSWER] Groq client available."
        )

        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------

        prompt = f"""
Answer the question using ONLY the provided context.

Return exactly one JSON object with:

- answer: string
- is_grounded: boolean
- confidence: number between 0 and 1
- citations: array of citation strings

Context:
{retrieval.context}

Question:
{query}

Allowed citations:
{retrieval.citations}
"""

        # -------------------------------------------------
        # GROQ
        # -------------------------------------------------

        completion = retry(
            lambda: self.client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.0,
                max_tokens=500,
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer in one concise sentence "
                            "using only the provided context. "
                            "Always return valid JSON with "
                            "keys answer, is_grounded, "
                            "confidence, and citations."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        )

        # -------------------------------------------------
        # PARSE RESPONSE
        # -------------------------------------------------

        try:

            result = RagResponse.model_validate(
                json.loads(
                    completion
                    .choices[0]
                    .message
                    .content
                )
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError
        ):

            print(
                "[ANSWER] Invalid Groq JSON response."
            )

            return RagResponse(
                answer=REFUSAL,
                is_grounded=False,
                confidence=0,
                citations=[]
            )

        # -------------------------------------------------
        # VALIDATE GROUNDING
        # -------------------------------------------------

        if not result.is_grounded:

            print(
                "[ANSWER] Model marked answer "
                "as not grounded."
            )

            return RagResponse(
                answer=REFUSAL,
                is_grounded=False,
                confidence=0,
                citations=[]
            )

        if not set(
            result.citations
        ).issubset(
            set(retrieval.citations)
        ):

            print(
                "[ANSWER] Invalid citation returned."
            )

            return RagResponse(
                answer=REFUSAL,
                is_grounded=False,
                confidence=0,
                citations=[]
            )

        print(
            "[ANSWER] Answer generated successfully."
        )

        return result