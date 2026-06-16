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

def load_embedder():
    global embedder
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedder ready")
        return True
    except Exception as e:
        print(f"Embedder load error: {e}")
        return False

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

def log_question(student, section, question, topic, answered):
    os.makedirs("logs", exist_ok=True)
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except Exception:
        logs = []
    logs.append({
        "student":   student,
        "section":   section,
        "question":  question,
        "topic":     topic,
        "answered":  answered,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    })
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def keyword_search(query, top_k=5):
    q_words = set(query.lower().split())
    scored = [(len(q_words & set(d["text"].lower().split())), i) for i, d in enumerate(docs)]
    scored.sort(reverse=True)
    return [docs[i] for s, i in scored[:top_k] if s > 0]

def search_top_k(query, k=5):
    if DIM_MATCH:
        try:
            vec = embedder.encode([query])[0].astype(np.float32)
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
    data     = request.get_json()
    question = (data.get("message") or "").strip()
    student  = (data.get("student") or "Student").strip()
    section  = (data.get("section") or "Unknown").strip()
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

@app.route("/teacher")
def teacher_page():
    return send_from_directory("static", "teacher.html")

@app.route("/teacher/login", methods=["POST"])
def teacher_login():
    pw = (request.get_json() or {}).get("password", "")
    if pw == os.environ.get("TEACHER_PASSWORD", "teacher123"):
        session["teacher"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 401

@app.route("/teacher/data")
def teacher_data():
    if not session.get("teacher"):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        with open(LOG_FILE) as f:
            logs = json.load(f)
    except Exception:
        logs = []

    topic_counts = {}
    for entry in logs:
        t = entry.get("topic", "General")
        topic_counts[t] = topic_counts.get(t, 0) + 1
    topics_sorted = sorted(topic_counts.items(), key=lambda x: -x[1])

    student_map = {}
    for entry in logs:
        s = entry.get("student", "Unknown")
        if s not in student_map:
            student_map[s] = {"count": 0, "section": entry.get("section", "?"), "last": ""}
        student_map[s]["count"] += 1
        student_map[s]["last"] = entry.get("timestamp", "")

    unanswered = [e for e in logs if not e.get("answered")]

    return jsonify({
        "total":      len(logs),
        "topics":     topics_sorted,
        "students":   student_map,
        "unanswered": unanswered[-20:],
        "recent":     logs[-20:]
    })

@app.route("/status")
def status():
    return jsonify({
        "chunks":         len(docs),
        "faiss_ready":    FAISS_READY,
        "dim_match":      DIM_MATCH,
        "embedder_ready": EMBEDDER_READY,
        "groq_key":       bool(os.environ.get("GROQ_API_KEY")),
    })

@app.route("/")
def index_page():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    if debug:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
