"""FastAPI voice endpoint and single-page dashboard."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from harness import RagHarness


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="HH Goa Voice RAG")


# =========================================================
# RAG HARNESS
# =========================================================

harness = RagHarness()


# =========================================================
# SPEECH TO TEXT
# =========================================================

def transcribe(audio: UploadFile) -> str:

    if not os.getenv("SARVAM_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is not configured"
        )

    print("[STT] Sending audio to Sarvam...")

    response = requests.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={
            "api-subscription-key": os.environ["SARVAM_API_KEY"]
        },
        files={
            "file": (
                audio.filename or "question.wav",
                audio.file,
                "audio/wav"
            )
        },
        data={
            "model": os.getenv(
                "SARVAM_MODEL",
                "saaras:v3"
            ),
            "language_code": "en-IN",
            "mode": "transcribe",
        },
        timeout=30,
    )

    if not response.ok:
        try:
            detail = response.json().get(
                "error",
                response.text
            )
        except ValueError:
            detail = response.text

        raise HTTPException(
            status_code=502,
            detail=(
                f"Sarvam transcription failed "
                f"({response.status_code}): {detail}"
            )
        )

    data = response.json()

    print("[STT] Sarvam response received.")

    transcript = data.get(
        "transcript",
        ""
    ).strip()

    print(f"[STT] Transcript: {transcript}")

    return transcript


# =========================================================
# VOICE RAG ENDPOINT
# =========================================================

@app.post("/api/voice-rag")
def voice_rag(
    audio: UploadFile = File(...)
) -> dict:

    print()
    print("=" * 65)
    print("VOICE RAG REQUEST STARTED")
    print("=" * 65)

    started = time.perf_counter_ns()

    try:

        # =================================================
        # STEP 1 — SPEECH TO TEXT
        # =================================================

        print()
        print("[1/4] Starting speech recognition...")

        stt_start = time.perf_counter_ns()

        query = transcribe(audio)

        stt_ms = (
            time.perf_counter_ns() - stt_start
        ) / 1_000_000

        print(
            f"[1/4] STT completed: "
            f"{stt_ms:.2f} ms"
        )

        print(
            f"[1/4] Query: {query}"
        )

        if not query:
            raise ValueError(
                "Sarvam returned an empty transcript."
            )

        # =================================================
        # STEP 2 — VECTOR RETRIEVAL
        # =================================================

        print()
        print("[2/4] Starting vector retrieval...")

        retrieval_start = time.perf_counter_ns()

        retrieval = harness.retrieve(query)

        retrieval_ms = (
            time.perf_counter_ns() - retrieval_start
        ) / 1_000_000

        print(
            f"[2/4] Retrieval completed: "
            f"{retrieval_ms:.2f} ms"
        )

        print(
            f"[2/4] Citations: "
            f"{retrieval.citations}"
        )

        print(
            f"[2/4] Distance: "
            f"{retrieval.distance}"
        )

        if not retrieval.context:
            print(
                "[2/4] WARNING: "
                "No matching document context found."
            )

        # =================================================
        # STEP 3 — GROQ GENERATION
        # =================================================

        print()
        print("[3/4] Starting answer generation...")

        generation_start = time.perf_counter_ns()

        result = harness.answer(
            query,
            retrieval=retrieval
        )

        generation_ms = (
            time.perf_counter_ns() - generation_start
        ) / 1_000_000

        print(
            f"[3/4] Generation completed: "
            f"{generation_ms:.2f} ms"
        )

        # =================================================
        # STEP 4 — FINAL RESPONSE
        # =================================================

        total_ms = (
            time.perf_counter_ns() - started
        ) / 1_000_000

        print()
        print("[4/4] PIPELINE COMPLETED SUCCESSFULLY")
        print(
            f"[4/4] Total time: "
            f"{total_ms:.2f} ms"
        )
        print("=" * 65)
        print()

        return {
            "transcript": query,
            **result.model_dump(),
            "latency_ms": {
                "stt_ms": round(stt_ms, 3),
                "retrieval_ms": round(retrieval_ms, 3),
                "generation_ms": round(generation_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "retrieval_distance": retrieval.distance,
        }

    # =====================================================
    # PIPELINE ERROR
    # =====================================================

    except Exception as error:

        print()
        print("!" * 65)
        print("VOICE RAG PIPELINE FAILED")
        print("!" * 65)

        print(
            f"ERROR TYPE: {type(error).__name__}"
        )

        print(
            f"ERROR MESSAGE: {error}"
        )

        print()
        print("FULL TRACEBACK:")

        traceback.print_exc()

        print("!" * 65)
        print()

        return {
            "transcript": "Error processing voice request",
            "result": {
                "answer": (
                    "Backend pipeline error. "
                    "Check the VS Code terminal."
                ),
                "is_grounded": False,
                "confidence": 0.0,
                "citations": [],
            },
            "latency_ms": {
                "stt_ms": 0,
                "retrieval_ms": 0,
                "generation_ms": 0,
                "total_ms": 0,
            },
        }


# =========================================================
# DASHBOARD
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def dashboard() -> str:

    html = (
        Path(__file__).parent / "dashboard.html"
    ).read_text(
        encoding="utf-8"
    )

    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-store"
        }
    )