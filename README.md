# ⚗️ ChemBot — RAG-Powered AI Tutor for Chem 110

An AI chatbot that answers student questions strictly from uploaded course material, built as part of the **MCREU (Multi-Campus Research Experience for Undergraduates)** program at Penn State Harrisburg under Professor Abu Asaduzzaman.

> **Research focus:** Measuring Retrieval-Augmented Generation (RAG) as a hallucination-reduction strategy in educational AI.

---

## Features

- **Syllabus-grounded answers** — bot only responds from course PDFs, PPTXs, and DOCXs; politely refuses off-topic questions
- **Conversation memory** — retains last 5 exchanges per session for natural follow-up questions
- **Voice input** — dual-path: Web Speech API (Chrome/Edge) + Groq Whisper (Safari-compatible)
- **Text-to-speech** — read-aloud toggle with per-message speak button
- **Topic detection** — 35+ chemistry keywords auto-labeled on each response
- **Free stack** — no paid APIs beyond Groq's free tier

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| LLM | Groq LLaMA 3.3 70B |
| Voice | Groq Whisper Large v3 Turbo |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, free) |
| Vector search | FAISS |
| Frontend | Vanilla HTML/CSS/JS |

## How It Works

1. Course files (PDF/PPTX/DOCX) are chunked and embedded locally using `sentence-transformers`
2. Embeddings are stored in a FAISS index
3. On each student question, the top-5 most relevant chunks are retrieved
4. Retrieved context + conversation history are passed to LLaMA 3.3 70B via Groq
5. The model is instructed to answer **only** from the retrieved course material

## Setup

```bash
# 1. Clone and install
git clone https://github.com/your-username/chembot.git
cd chembot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env

# 3. Add course files and build vectorstore
# Drop PDF/PPTX/DOCX files into data/
python ingest_free.py

# 4. Run
python app.py
```

## Research Context

This project is part of the **MCREU Summer 2026** research program. The research topic is:

*"Design and Development of a Syllabus-Grounded Retrieval-Augmented AI Chatbot for Middle School Coursework"* — extended to college chemistry (Chem 110).

The core research question: **Does RAG measurably reduce hallucination rates in educational AI compared to a base LLM?**

---

Built by **Om Patel** · Penn State Harrisburg · Research Assistant under Prof. Abu Asaduzzaman
