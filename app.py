import os
import json
import pickle
import tempfile
import datetime
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, session
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "chembot-secret-2024")

DOCS_FILE  = "vectorstore/docs.pkl"
INDEX_FILE = "vectorstore/index.faiss"

docs      = []
index     = None
index_dim = None
embedder  = None
EMBEDDER_DIM = 384

def load_vectorstore():
    global docs, index, index_dim
    try:
        import faiss
        if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
            index = faiss.read_index(INDEX_FILE)
            index_dim = index.d
            with open(DOCS_FILE, "rb") as f:
                docs = pickle.load(f)
            print(f"Loaded vectorstore: {len(docs)} chunks")
            return True
    except Exception as e:
        print(f"Vectorstore load error: {e}")
    return False

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedder():
    global embedder
    try:
        from fastembed import TextEmbedding
        print("Loading embedding model...")
        embedder = TextEmbedding(model_name=EMBED_MODEL)
        print("Embedder ready")
        return True
    except Exception as e:
        print(f"Embedder load error: {e}")
        return False

def embed_query(text):
    """Return a single 384-dim float32 vector for the given text."""
    return np.array(list(embedder.embed([text]))[0], dtype=np.float32)

FAISS_READY    = load_vectorstore()
EMBEDDER_READY = load_embedder()
DIM_MATCH      = FAISS_READY and EMBEDDER_READY and (index_dim == EMBEDDER_DIM)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

TOPIC_MAP = {
    "acid":          "Acids & Bases",
    "base":          "Acids & Bases",
    "buffer":        "Acids & Bases",
    "ph":            "pH Scale",
    "neutral":       "pH Scale",
    "titration":     "Acid-Base Titrations",
    "indicator":     "Acid-Base Titrations",
    "molarity":      "Concentration",
    "concentration": "Concentration",
    "solution":      "Solutions",
    "solute":        "Solutions",
    "solvent":       "Solutions",
    "colligative":   "Colligative Properties",
    "boiling point": "Colligative Properties",
    "freezing":      "Colligative Properties",
    "osmosis":       "Colligative Properties",
    "enthalpy":      "Thermochemistry",
    "hess":          "Hess's Law",
    "heat":          "Heat & Temperature",
    "temperature":   "Heat & Temperature",
    "exothermic":    "Thermochemistry",
    "endothermic":   "Thermochemistry",
    "reaction":      "Chemical Reactions",
    "neutralization":"Chemical Reactions",
    "salt":          "Chemical Reactions",
    "imf":           "Intermolecular Forces",
    "intermolecular":"Intermolecular Forces",
    "hydrogen bond": "Intermolecular Forces",
    "dipole":        "Intermolecular Forces",
    "van der waals": "Intermolecular Forces",
    "phase":         "Phase Changes",
    "vaporization":  "Phase Changes",
    "condensation":  "Phase Changes",
    "sublimation":   "Phase Changes",
    "specific heat": "Calorimetry",
    "calorimetry":   "Calorimetry",
    "energy":        "Energy",
    "joule":         "Energy",
    "calorie":       "Energy",
    "stoichiometry": "Stoichiometry",
    "mole":          "Stoichiometry",
}

def detect_topic(text):
    lower = text.lower()
    for keyword, topic in TOPIC_MAP.items():
        if keyword in lower:
            return topic
    return "General Chemistry"

LOG_FILE = "logs/questions.json"

# When DATABASE_URL is set (e.g. on Render, pointing at Supabase Postgres),
# questions are stored in the database so they survive server restarts.
# Locally, with no DATABASE_URL, they fall back to logs/questions.json.
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_READY = False

def init_db():
    global DB_READY
    if not DATABASE_URL:
        print("No DATABASE_URL — logging to local JSON file")
        return False
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id        SERIAL PRIMARY KEY,
                student   TEXT,
                section   TEXT,
                question  TEXT,
                topic     TEXT,
                answered  BOOLEAN,
                ts        TIMESTAMPTZ DEFAULT now()
            )
        """)
        conn.commit()
        cur.close(); conn.close()
        DB_READY = True
        print("Database ready (Postgres)")
        return True
    except Exception as e:
        print(f"DB init error, falling back to JSON: {e}")
        return False

def write_log(entry):
    if DB_READY:
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL, sslmode="require")
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO questions (student, section, question, topic, answered, ts) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (entry["student"], entry["section"], entry["question"],
                 entry["topic"], entry["answered"], entry["timestamp"])
            )
            conn.commit(); cur.close(); conn.close()
            return
        except Exception as e:
            print(f"DB write error, falling back to JSON: {e}")
    os.makedirs("logs", exist_ok=True)
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except Exception:
        logs = []
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def read_logs():
    if DB_READY:
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL, sslmode="require")
            cur = conn.cursor()
            cur.execute("SELECT student, section, question, topic, answered, ts "
                        "FROM questions ORDER BY id ASC")
            rows = cur.fetchall(); cur.close(); conn.close()
            return [{
                "student":   r[0],
                "section":   r[1],
                "question":  r[2],
                "topic":     r[3],
                "answered":  r[4],
                "timestamp": r[5].isoformat() if r[5] else "",
            } for r in rows]
        except Exception as e:
            print(f"DB read error, falling back to JSON: {e}")
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def log_question(student, section, question, topic, answered):
    write_log({
        "student":   student,
        "section":   section,
        "question":  question,
        "topic":     topic,
        "answered":  answered,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    })

init_db()

def keyword_search(query, top_k=5):
    q_words = set(query.lower().split())
    scored = [(len(q_words & set(d["text"].lower().split())), i) for i, d in enumerate(docs)]
    scored.sort(reverse=True)
    return [docs[i] for s, i in scored[:top_k] if s > 0]

def search_top_k(query, k=5):
    if DIM_MATCH:
        try:
            vec = embed_query(query)
            D, I = index.search(vec.reshape(1, -1), k)
            results = [docs[i] for i in I[0] if 0 <= i < len(docs)]
            if results:
                return results
        except Exception as e:
            print(f"Search error: {e}")
    return keyword_search(query, k)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    mime   = audio_file.mimetype or ""
    suffix = ".mp4" if "mp4" in mime else ".ogg" if "ogg" in mime else ".webm"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(os.path.basename(tmp_path), f),
                response_format="text"
            )
        os.unlink(tmp_path)
        text = result if isinstance(result, str) else result.text
        return jsonify({"text": text})
    except Exception as e:
        try: os.unlink(tmp_path)
        except: pass
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    if not session.get("student_auth"):
        return jsonify({"error": "Please log in first.", "auth": False}), 401

    data     = request.get_json()
    question = (data.get("message") or "").strip()
    student  = session.get("student_name", "Student")
    section  = session.get("student_section", "Unknown")
    history  = data.get("history") or []

    if not question:
        return jsonify({"error": "Empty message"}), 400

    topic = detect_topic(question)

    if not docs:
        return jsonify({"reply": "No course material loaded. Run ingest_free.py first."})

    retrieved = search_top_k(question, k=5)
    answered  = bool(retrieved)

    log_question(student, section, question, topic, answered)

    if not retrieved:
        return jsonify({
            "reply": "That topic doesn't appear in our course material. Try asking about acids and bases, pH, titrations, solutions, enthalpy, or intermolecular forces.",
            "topic": topic
        })

    context = "\n\n---\n\n".join(
        f"[{d['meta'].get('source_file','?')} p.{d['meta'].get('page_number','?')}]\n{d['text']}"
        for d in retrieved
    )

    system_prompt = (
        "You are ChemBot, a helpful tutor for the Penn State Harrisburg Chem 110 course.\n"
        "Rules:\n"
        "1. Answer ONLY using the course material provided below. Do not use outside knowledge.\n"
        "2. If the student asks which file or source a topic comes from, check the filenames in "
        "brackets (e.g. [Copy of Acids, Bases, pH Scale.pptx]) and tell them.\n"
        "3. If the topic is not in the material, say so and suggest a related topic from Chem 110.\n"
        "4. Keep answers clear and concise. Use bullet points where it helps.\n\n"
        f"Course material:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-5:]:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        return jsonify({"reply": f"Error reaching AI service: {str(e)[:120]}"}), 500

    return jsonify({"reply": reply, "topic": topic})

def _serve_static_html(filename):
    path = os.path.join(os.path.dirname(__file__), "static", filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/student/login", methods=["POST"])
def student_login():
    data    = request.get_json() or {}
    name    = (data.get("name") or "").strip()
    section = (data.get("section") or "").strip()
    pw      = data.get("password", "")
    if len(name) < 2 or not section:
        return jsonify({"ok": False, "error": "Name and section are required."}), 400
    if pw != os.environ.get("STUDENT_PASSWORD", "chem110"):
        return jsonify({"ok": False, "error": "Incorrect class password."}), 401
    session["student_auth"]    = True
    session["student_name"]    = name
    session["student_section"] = section
    return jsonify({"ok": True})

@app.route("/teacher")
def teacher_page():
    return _serve_static_html("teacher.html")

@app.route("/teacher/login", methods=["POST"])
def teacher_login():
    pw = (request.get_json() or {}).get("password", "")
    if pw == os.environ.get("TEACHER_PASSWORD", "teacher123"):
        session["teacher"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 401

def _norm_q(q):
    """Normalize a question for repeat detection."""
    return " ".join((q or "").lower().split()).rstrip("?.! ")

@app.route("/teacher/data")
def teacher_data():
    if not session.get("teacher"):
        return jsonify({"error": "Unauthorized"}), 401
    logs = read_logs()

    # ── Class-wide topic frequency ──
    topic_counts = {}
    unanswered_topics = {}
    for entry in logs:
        t = entry.get("topic", "General")
        topic_counts[t] = topic_counts.get(t, 0) + 1
        if not entry.get("answered"):
            unanswered_topics[t] = unanswered_topics.get(t, 0) + 1
    topics_sorted = sorted(topic_counts.items(), key=lambda x: -x[1])
    unanswered_topics_sorted = sorted(unanswered_topics.items(), key=lambda x: -x[1])

    # ── Per-student breakdown ──
    student_map = {}
    for entry in logs:
        s = entry.get("student", "Unknown")
        if s not in student_map:
            student_map[s] = {
                "section":    entry.get("section", "?"),
                "count":      0,
                "last":       "",
                "topics":     {},
                "unanswered": 0,
                "_qseen":     {},   # normalized question -> count (internal)
                "repeats":    [],
                "questions":  [],   # full list of what this student asked
            }
        rec = student_map[s]
        rec["count"] += 1
        rec["last"]    = entry.get("timestamp", "")
        rec["section"] = entry.get("section", rec["section"])
        topic = entry.get("topic", "General")
        rec["topics"][topic] = rec["topics"].get(topic, 0) + 1
        if not entry.get("answered"):
            rec["unanswered"] += 1
        rec["questions"].append({
            "question":  entry.get("question", ""),
            "topic":     topic,
            "answered":  bool(entry.get("answered")),
            "timestamp": entry.get("timestamp", ""),
        })
        nq = _norm_q(entry.get("question", ""))
        if nq:
            rec["_qseen"][nq] = rec["_qseen"].get(nq, 0) + 1

    # Finalize per-student: sort topics, extract repeated questions, weakest topic
    for s, rec in student_map.items():
        rec["top_topics"] = sorted(rec["topics"].items(), key=lambda x: -x[1])
        rec["repeats"] = sorted(
            [{"question": q, "count": c} for q, c in rec["_qseen"].items() if c > 1],
            key=lambda x: -x["count"]
        )
        rec["weak_topic"] = rec["top_topics"][0][0] if rec["top_topics"] else "—"
        del rec["_qseen"]

    unanswered = [e for e in logs if not e.get("answered")]

    return jsonify({
        "total":             len(logs),
        "topics":            topics_sorted,
        "unanswered_topics": unanswered_topics_sorted,
        "students":          student_map,
        "unanswered":        unanswered[-30:],
        "recent":            logs[-30:],
    })

@app.route("/status")
def status():
    return jsonify({
        "chunks":         len(docs),
        "faiss_ready":    FAISS_READY,
        "dim_match":      DIM_MATCH,
        "embedder_ready": EMBEDDER_READY,
        "groq_key":       bool(os.environ.get("GROQ_API_KEY")),
        "database":       DB_READY,
    })

@app.route("/")
def index_page():
    return _serve_static_html("index.html")

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") != "production"
    if debug:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
