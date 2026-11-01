
import os
import re
import numpy as np
import gradio as gr
import faiss
import torch

from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# --------------------------------
# 1. LOAD AI MODELS
# --------------------------------

print("Loading AI models...")

device = 0 if torch.cuda.is_available() else -1

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=device
)

# --------------------------------
# 2. DOCUMENT PROCESSING
# --------------------------------

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=180, overlap=40):
    words = text.split()
    result = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        result.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return result


def process_pdfs(files):
    documents = []

    if not files:
        return [], "Please upload at least one PDF."

    for file_path in files:
        if not file_path.lower().endswith(".pdf"):
            continue

        reader = PdfReader(file_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")

            if text:
                documents.append({
                    "text": text,
                    "source": os.path.basename(file_path),
                    "page": page_number
                })

    if not documents:
        return [], "No readable text found. Check your PDF files."

    chunks = []

    for doc in documents:
        for part in chunk_text(doc["text"]):
            chunks.append({
                "text": part,
                "source": doc["source"],
                "page": doc["page"]
            })

    return chunks, f"Successfully processed {len(chunks)} text chunks."


# --------------------------------
# 3. CREATE VECTOR DATABASE
# --------------------------------

def create_database(chunks):

    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    vector_db = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_db.add(embeddings)

    bm25 = BM25Okapi([
        text.lower().split() for text in texts
    ])

    return vector_db, bm25


# --------------------------------
# 4. HYBRID RETRIEVAL
# --------------------------------

def retrieve(question, chunks, vector_db, bm25, k=5):

    if not chunks:
        return []

    k = min(k, len(chunks))

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    _, semantic_ids = vector_db.search(
        query_embedding, k
    )

    semantic_rank = {
        int(idx): rank + 1
        for rank, idx in enumerate(semantic_ids[0])
        if idx >= 0
    }

    keyword_scores = bm25.get_scores(
        question.lower().split()
    )

    keyword_ids = np.argsort(
        keyword_scores
    )[::-1][:k]

    keyword_rank = {
        int(idx): rank + 1
        for rank, idx in enumerate(keyword_ids)
    }

    candidates = set(semantic_rank) | set(keyword_rank)

    ranked_ids = sorted(
        candidates,
        key=lambda idx: (
            1 / (60 + semantic_rank.get(idx, 10000))
            + 1 / (60 + keyword_rank.get(idx, 10000))
        ),
        reverse=True
    )

    return [chunks[idx] for idx in ranked_ids[:5]]


# --------------------------------
# 5. RERANKING
# --------------------------------

def rerank(question, candidates):

    if not candidates:
        return []

    pairs = [
        (question, item["text"])
        for item in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: float(item[0]),
        reverse=True
    )

    return [
        item for score, item in ranked[:3]
    ]


# --------------------------------
# 6. ANSWER GENERATION
# --------------------------------

def answer_question(question, chunks, vector_db, bm25):

    if not question.strip():
        return "Please enter your question."

    retrieved = retrieve(
        question, chunks, vector_db, bm25
    )

    best_chunks = rerank(question, retrieved)

    if not best_chunks:
        return "No relevant information was found."

    context = "\n\n".join(
        f"{item['text']}\n"
        for item in best_chunks
    )

    prompt = f"""
    Answer using only the following document context.
    Do not invent facts.
    If the answer is missing, say the documents
    do not contain enough information.

    Context:
    {context[:3000]}

    Question:
    {question}

    Answer:
    """

    output = generator(
        prompt,
        max_new_tokens=160,
        do_sample=False,
        truncation=True
    )

    answer = output[0]["generated_text"]

    sources = sorted(set(
        f"{item['source']} - Page {item['page']}"
        for item in best_chunks
    ))

    return (
        f"{answer}\n\n"
        f"Sources:\n" +
        "\n".join(sources)
    )


# --------------------------------
# 7. CHATBOT INTERFACE
# --------------------------------

def chatbot(message, history, files):

    chunks, status = process_pdfs(files)

    if not chunks:
        return status

    vector_db, bm25 = create_database(chunks)

    return answer_question(
        message, chunks, vector_db, bm25
    )


# --------------------------------
# 8. GRADIO DASHBOARD
# --------------------------------

demo = gr.ChatInterface(
    fn=chatbot,
    additional_inputs=[
        gr.File(
            label="Upload PDF Documents",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath"
        )
    ],
    title="Advanced RAG Document Chatbot",
    description=(
        "Upload PDF documents and ask questions. "
        "Get AI-generated answers with source references."
    )
)

# --------------------------------
# 9. LAUNCH APPLICATION
# --------------------------------

if __name__ == "__main__":
    demo.launch()
