"""
ingest_free.py — Rebuild vectorstore from data/ folder.
Supports PDF, PPTX, PPT (via python-pptx), and DOCX.

Usage:
    python ingest_free.py

Output:
    vectorstore/index.faiss  (384-dim)
    vectorstore/docs.pkl
"""

import os
import pickle
import re
import numpy as np

DATA_PATH  = "data"
OCR_DIR    = os.path.join(DATA_PATH, "ocr_review")   # hand-checked text for scanned PDF pages
VECTOR_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
DOCS_FILE  = os.path.join(VECTOR_DIR, "docs.pkl")

CHUNK_SIZE = 900
OVERLAP    = 200


MIN_CHUNK_CHARS = 50   # drop slide-title fragments and stray OCR noise

def is_useful_chunk(text: str) -> bool:
    """Keep only chunks with real content: long enough and mostly letters/words."""
    if len(text) < MIN_CHUNK_CHARS:
        return False
    letters = sum(c.isalpha() for c in text)
    words   = [w for w in text.split() if len(w) > 1 and any(c.isalpha() for c in w)]
    # reject garbled fragments that are mostly digits/symbols or too few real words
    return letters >= 25 and len(words) >= 6

SENTENCE_END = re.compile(r"[.!?;:](?=\s)")

def chunk_text(text: str):
    """Split into overlapping chunks that break between sentences, or at worst between words."""
    text = " ".join(text.split())   # normalize whitespace
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            ends = [m.end() for m in SENTENCE_END.finditer(text, start + CHUNK_SIZE // 2, end + 1)]
            space = text.rfind(" ", start + 1, end + 1)
            end = ends[-1] if ends else (space if space != -1 else end)
        chunk = text[start:end].strip()
        if chunk and is_useful_chunk(chunk):
            chunks.append(chunk)
        if end >= len(text):
            break
        back = max(end - OVERLAP, start + 1)
        sentence = SENTENCE_END.search(text, back, end)
        space = text.find(" ", back, end)
        start = sentence.end() + 1 if sentence else (space + 1 if space != -1 else end)
    return chunks


def ocr_pages(filename: str):
    """Hand-checked text for a PDF's scanned pages, as {page_num: text}."""
    path = os.path.join(OCR_DIR, os.path.splitext(filename)[0] + ".txt")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        parts = re.split(r"^--- page (\d+) ---$", f.read(), flags=re.M)
    return {int(n): text.strip() for n, text in zip(parts[1::2], parts[2::2]) if text.strip()}


def extract_pdf(path: str):
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = {}
    for i, page in enumerate(reader.pages):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages[i + 1] = txt
    pages.update(ocr_pages(os.path.basename(path)))
    return sorted(pages.items())   # list of (page_num, text)


def slide_lines(shapes):
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from slide_lines(shape.shapes)
        elif getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                line = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if line:
                    yield line
        elif shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                line = para.text.strip()   # keeps subscripts attached: H2SO4, not H 2 SO 4
                if line:
                    yield line


def extract_pptx(path: str):
    from pptx import Presentation
    prs = Presentation(path)
    pages = []
    for i, slide in enumerate(prs.slides):
        texts = list(slide_lines(slide.shapes))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                texts.append(notes)
        combined = " ".join(texts).strip()
        if combined:
            pages.append((i + 1, combined))
    return pages


def extract_docx(path: str):
    from docx import Document
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            seen, cells = set(), []
            for cell in row.cells:           # merged cells repeat in row.cells
                if cell._tc not in seen and cell.text.strip():
                    seen.add(cell._tc)
                    cells.append(cell.text.strip())
            if cells:
                parts.append(" | ".join(cells))
    text = " ".join(parts)
    return [(1, text)] if text.strip() else []


def extract_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in (".pptx", ".ppt"):
        return extract_pptx(path)
    if ext == ".docx":
        return extract_docx(path)
    return []


def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

    supported = (".pdf", ".pptx", ".ppt", ".docx")
    files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(supported)]

    if not files:
        print(f"❌ No supported files found in '{DATA_PATH}/'.")
        print(f"   Add PDF, PPTX, or DOCX files and run again.")
        return

    docs = []
    for filename in sorted(files):
        path = os.path.join(DATA_PATH, filename)
        print(f"📄 {filename}")
        try:
            pages = extract_file(path)
        except Exception as e:
            print(f"   ⚠️  Skipped ({e})")
            continue

        for page_num, text in pages:
            for chunk_i, chunk in enumerate(chunk_text(text)):
                docs.append({
                    "text": chunk,
                    "meta": {
                        "source_file": filename,
                        "page_number": page_num,
                        "chunk_id":    f"{filename}_p{page_num}_c{chunk_i+1}"
                    }
                })

    print(f"\n✅ {len(docs)} chunks from {len(files)} file(s)")

    print("⏳ Loading embedding model (first run downloads ~90 MB)…")
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("⏳ Embedding chunks…")
    texts    = [d["text"] for d in docs]
    all_vecs = list(model.embed(texts))
    emb      = np.array(all_vecs, dtype=np.float32)

    import faiss
    dim   = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    faiss.write_index(index, INDEX_FILE)

    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

    print(f"\n✅ Vectorstore saved!")
    print(f"   {INDEX_FILE}  ({dim}-dim, {len(docs)} vectors)")
    print(f"   {DOCS_FILE}")
    print(f"\nRestart app.py and you're good to go 🚀")


if __name__ == "__main__":
    main()
