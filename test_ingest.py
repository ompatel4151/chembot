import tempfile, os
from docx import Document
from pptx import Presentation
from pptx.util import Inches
import ingest_free
from ingest_free import chunk_text, extract_docx, extract_pptx

with tempfile.TemporaryDirectory() as tmp:
    doc = Document()
    t = doc.add_table(rows=1, cols=2)
    t.rows[0].cells[0].merge(t.rows[0].cells[1]).text = "merged cell"
    path = os.path.join(tmp, "a.docx")
    doc.save(path)
    assert extract_docx(path) == [(1, "merged cell")]   # table read once, not twice

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_table(1, 1, 0, 0, Inches(2), Inches(1)).table.cell(0, 0).text = "table text"
    g = s.shapes.add_group_shape()
    g.shapes.add_textbox(0, 0, Inches(1), Inches(1)).text_frame.text = "group text"
    s.notes_slide.notes_text_frame.text = "note text"
    path = os.path.join(tmp, "a.pptx")
    prs.save(path)
    text = extract_pptx(path)[0][1]
    assert all(w in text for w in ("table text", "group text", "note text")), text

    prs = Presentation()
    p = prs.slides.add_slide(prs.slide_layouts[6]).shapes.add_textbox(0, 0, Inches(1), Inches(1)).text_frame.paragraphs[0]
    for part in ("H", "2", "SO", "4"):
        p.add_run().text = part
    path = os.path.join(tmp, "b.pptx")
    prs.save(path)
    assert extract_pptx(path) == [(1, "H2SO4")]         # subscripts stay attached

    ingest_free.OCR_DIR = tmp
    with open(os.path.join(tmp, "scan.txt"), "w", encoding="utf-8") as f:
        f.write("SOURCE: scan.pdf\n\n--- page 2 ---\nreal text\n\n--- page 5 ---\n\n")
    assert ingest_free.ocr_pages("scan.pdf") == {2: "real text"}   # header and blank pages dropped
    assert ingest_free.ocr_pages("other.pdf") == {}

sentences = [f"Sentence number {i} talks about molar mass and limiting reagents." for i in range(60)]
chunks = chunk_text(" ".join(sentences))
assert all(len(c) <= 900 for c in chunks)
assert all(c.startswith("Sentence") and c.endswith(".") for c in chunks), chunks   # whole sentences only
assert all(any(s in c for c in chunks) for s in sentences)                         # nothing lost

words = " ".join(f"word{i}" for i in range(600))                                   # no punctuation at all
chunks = chunk_text(words)
assert all(len(c) <= 900 and c.startswith("word") for c in chunks)
assert set(" ".join(chunks).split()) == set(words.split())

print("ok")
