# Voice RAG - Low-Latency Multilingual Retrieval Pipeline

A production-ready, low-latency Voice Retrieval-Augmented Generation (RAG) system using **FastAPI**, **LanceDB**, **Sarvam AI (STT)**, and **Groq (Llama-3)**.

---

## Features

- **Voice-to-Answer Interface**: Fast end-to-end speech-to-text, vector search, and grounded response synthesis.
- **Strict Grounding Guardrails**: Zero-hallucination policy that safely refuses out-of-corpus queries.
- **Adversarial Input Defense**: Instant guardrail short-circuiting (0.01 ms) against prompt injection attempts.
- **Vector Database**: High-speed hybrid vector retrieval powered by LanceDB.
- **Real-Time Telemetry**: Granular latency breakdowns for STT, retrieval, and generation.

---

## Latency Benchmark

Ran across 50 iterations:

| Category | P50 (ms) | P70 (ms) | P100 (ms) |
| :--- | :--- | :--- | :--- |
| **Grounded** | 3994.82 | 4062.39 | 4953.84 |
| **Off-Topic** | 4034.06 | 4635.46 | 5033.06 |
| **Adversarial** | 0.01 | 0.01 | 0.02 |
| **Overall** | 3910.06 | 4062.39 | 5033.06 |

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/riyasasaha/voice-rag-generater.git](https://github.com/riyasasaha/voice-rag-generater.git)
cd voice-rag-generater
