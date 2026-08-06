# ChemBot - AI Chatbot for Chem 110

Built this as part of my summer research under Professor Abu Asaduzzaman at Penn State Harrisburg (MCREU Summer 2026). The idea was to create a chatbot that only answers from the actual course material — so it can't hallucinate or pull stuff from outside the syllabus.

**Live demo:** https://chembot-kvo3.onrender.com

## What it does

- Answers chemistry questions using the uploaded course slides and notes
- If a question isn't covered in the material, it says so instead of making something up
- Students log in with their name, section, and a class password
- Voice input (works on Chrome and Safari) and a read-aloud button on responses
- Remembers the last few messages so follow-up questions make sense
- Logs every question so the instructor can see what's being asked

## Instructor dashboard

There's a separate password-protected dashboard at `/teacher` that shows:

- Most-asked topics and where the class has knowledge gaps
- A per-student breakdown — what each student asked, topics they repeat, and questions the bot couldn't answer
- Recent activity across the whole class

The goal is to help the instructor spot what students struggle with most.

## Stack

- Flask (Python) backend
- Groq API — LLaMA 3.3 70B for answers, Whisper for voice transcription
- fastembed (ONNX) + FAISS for semantic search over the course content
- Postgres (Supabase) for storing questions
- Plain HTML/CSS/JS frontend, no framework
- Deployed on Render

## How the RAG part works

When you ask a question, it searches the embedded course material (PDFs, PowerPoints, Word docs) and pulls the most relevant chunks. Those chunks get passed to the LLM along with your question, and the model is told to only use that content to answer. This is the core of the research — testing whether this approach actually reduces hallucinations compared to just asking an LLM directly.

## Running it locally

```bash
git clone https://github.com/ompatel4151/chembot.git
cd chembot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your values. Then add your course files to the `data/` folder and run:

```bash
python ingest_free.py   # builds the search index from data/
python app.py
```

Without a `DATABASE_URL` it just logs questions to a local file, so you don't need a database to try it out locally.

## Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key (required) |
| `SECRET_KEY` | Any random string, used for Flask sessions |
| `TEACHER_PASSWORD` | Password for the instructor dashboard |
| `STUDENT_PASSWORD` | Class password students use to log in |
| `DATABASE_URL` | Postgres connection string (optional — falls back to a local file if unset) |

## Research background

This is part of my MCREU research project at Penn State Harrisburg. The research looks at using RAG (Retrieval-Augmented Generation) to make educational AI more accurate and less likely to give students wrong information. Professor Asaduzzaman is my faculty mentor.

---

Om Patel | Penn State Harrisburg
