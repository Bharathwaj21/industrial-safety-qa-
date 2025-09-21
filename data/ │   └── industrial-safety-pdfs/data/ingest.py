import os
import sqlite3
import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "data/industrial-safety-pdfs"
DB_PATH = "db/chunks.sqlite"

os.makedirs("db", exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text, max_words=120):
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for p in paragraphs:
        words = p.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i+max_words])
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
    return chunks

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        doc_name TEXT,
        chunk TEXT
    )""")

    for pdf_file in tqdm(Path(DATA_DIR).glob("*.pdf")):
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        for chunk in chunks:
            c.execute("INSERT INTO chunks (doc_name, chunk) VALUES (?, ?)", (pdf_file.name, chunk))

    conn.commit()
    conn.close()
    print(f"✅ Ingested and chunked PDFs into {DB_PATH}")

if __name__ == "__main__":
    main()
