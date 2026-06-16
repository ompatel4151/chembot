# ChemBot - AI Chatbot for Chem 110

Built this as part of my summer research under Professor Abu Asaduzzaman at Penn State Harrisburg (MCREU Summer 2026). The idea was to create a chatbot that only answers from the actual course material — so it can't hallucinate or pull stuff from outside the syllabus.

## What it does

- Answers chemistry questions using the uploaded course slides and notes
- If a question isn't covered in the material, it tells you instead of making something up
- Supports voice input (works on both Chrome and Safari)
- Has a read-aloud button on responses
- Remembers the last few messages so follow-up questions make sense

## Stack

- Flask (Python) for the backend
- Groq API — LLaMA 3.3 70B for answers, Whisper for voice transcription
- sentence-transformers + FAISS for searching through course content
- Plain HTML/CSS/JS frontend, no framework

## How the RAG part works

When you ask a question, it searches through the embedded course material (PDFs, PowerPoints, Word docs) and pulls the most relevant chunks. Those chunks get passed to the LLM along with your question, and the model is told to only use that content to answer. This is the core of the research — testing whether this approach actually reduces hallucinations compared to just asking an LLM directly.

## Running it locally

```bash
git clone https://github.com/ompatel4151/chembot.git
cd chembot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) and add your Groq API key.

Then add your course files to the `data/` folder and run:
```bash
python ingest_free.py
python app.py
```

## Research background

This is part of my MCREU research project at Penn State Harrisburg. The research looks at using RAG (Retrieval-Augmented Generation) to make educational AI more accurate and less likely to give students wrong information. Professor Asaduzzaman is my faculty mentor.

---

Om Patel | Penn State Harrisburg
