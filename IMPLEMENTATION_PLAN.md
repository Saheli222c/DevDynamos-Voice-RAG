# Voice-Enabled RAG Implementation Plan

1. Load the English portion of `ai4bharat/MSMARCO-XI`, cap it at 3,000 rows, and normalize passage fields.
2. Create 4-sentence parent contexts and 2-sentence child search blocks, embed children with `BAAI/bge-small-en-v1.5`, and persist them in `./data/lancedb`.
3. Apply prompt-injection filtering, distance-filtered retrieval, Groq structured generation, grounding validation, and exponential retries.
4. Expose Sarvam `saaras:v1` transcription and RAG at `POST /api/voice-rag`, with a microphone dashboard at `GET /`.
5. Run 50 grounded, off-topic, and adversarial cases and report P50/P70/P100 latency.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe ingest.py
.\venv\Scripts\python.exe benchmark.py
```

Groq and Sarvam calls require `GROQ_API_KEY` and `SARVAM_API_KEY`. Without them, local safety, retrieval, and refusal checks still run.