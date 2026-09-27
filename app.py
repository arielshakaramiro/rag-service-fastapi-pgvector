import io
import os
import re
import secrets
import threading
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import psycopg2
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from pypdf import PdfReader

# Urutan import penting: torch (lewat sentence_transformers) dimuat sebelum llama_cpp,
# supaya library CUDA/NCCL milik llama.cpp tidak mendahului library yang dibutuhkan torch.
from sentence_transformers import CrossEncoder
from llama_cpp import Llama

from config import DB_CONFIG

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "60"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "15"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))  # kandidat dari pgvector
TOP_K = int(os.getenv("TOP_K", "3"))             # chunk yang masuk prompt setelah rerank
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0"))  # 0 = nonaktif; kalibrasi dari /search
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")  # wajib diset untuk menambah knowledge

FALLBACK = "Maaf, informasi tersebut belum tersedia. Silakan hubungi customer service Emerald."
# Keputusan "relevan atau tidak" sekarang diambil reranker (RERANK_THRESHOLD), bukan LLM.
# LLM cukup menjawab dari konteks yang sudah lolos saringan.
PERSONA = "Kamu adalah customer service perusahaan bernama Emerald."
# Versi sebelumnya: persona + aturan + kalimat penolakan wajib di slot Instruction, pertanyaan di slot Input.
# Pada model 8B ini, kalimat penolakan yang ditulis persis ternyata terlalu mudah "dipilih" model,
# bahkan saat konteksnya benar. Disimpan untuk perbandingan di notebook.
STRICT_INSTRUCTION = (
    PERSONA + " Jawab pertanyaan pelanggan HANYA berdasarkan konteks yang diberikan, singkat dan sopan. "
    "Jika konteks memuat jawabannya, sebutkan angka atau detailnya persis (harga, durasi, jumlah). "
    "Jika konteks hanya menjawab sebagian pertanyaan, jawab bagian yang tersedia. "
    f'Jika konteks sama sekali tidak berhubungan dengan pertanyaan, jawab persis: "{FALLBACK}"'
)
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "question_as_instruction")

CHAT_TEMPERATURE = 0.0        # /chat deterministik: pertanyaan yang sama, jawaban yang sama
REGENERATE_TEMPERATURE = 0.7  # /regenerate sengaja lebih bervariasi
ALPACA_PROMPT = """Di bawah ini adalah instruksi yang menjelaskan tugas, dipasangkan dengan masukan yang memberikan konteks lebih lanjut. Tulis tanggapan yang melengkapi permintaan dengan tepat.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
STOP = ["### Instruction:", "### Input:", "### Response:", "<|eot_id|>", "<|end_of_text|>"]
# repeat_penalty mencegah model terjebak mengulang kalimat yang sama sampai max_tokens habis
GEN_KWARGS = {"max_tokens": 256, "repeat_penalty": 1.15, "stop": STOP}

# ---------------------------------------------------------------------------
# Model (objek Llama tidak thread-safe, jadi setiap model dijaga lock)
# ---------------------------------------------------------------------------
llm = Llama.from_pretrained(
    repo_id="rubythalib33/llama3_1_8b_finetuned_bahasa_indonesia",
    filename="unsloth.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)
text_embedder = Llama.from_pretrained(
    repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
    embedding=True,
    n_gpu_layers=-1,
    verbose=False,
)
# Reranker cross-encoder multilingual: menilai pasangan (pertanyaan, chunk) secara langsung,
# sehingga skornya jauh lebih mudah dipisahkan dengan threshold dibanding cosine similarity.
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

llm_lock = threading.Lock()
embed_lock = threading.Lock()
rerank_lock = threading.Lock()

app = FastAPI(title="Emerald RAG Service")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
@contextmanager
def get_cursor(dict_rows=False):
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor if dict_rows else None) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def embed(text: str, kind: str) -> str:
    """Embedding nomic dengan prefix task, dinormalisasi, dikembalikan sebagai literal pgvector."""
    prefix = {"doc": "search_document: ", "query": "search_query: "}[kind]
    with embed_lock:
        vec = np.array(text_embedder.create_embedding(prefix + text)["data"][0]["embedding"], dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def chunk_words(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP) -> List[str]:
    if not (0 <= overlap < size):
        raise ValueError("Butuh 0 <= overlap < size")
    words, chunks = text.split(), []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return chunks


def retrieve(query_vec: str, k: int = RETRIEVE_K):
    with get_cursor(dict_rows=True) as cur:
        cur.execute(
            """
            SELECT k.document_id, kv.chunk_text,
                   1 - (kv.chunk_vector <=> %s::vector) AS similarity
            FROM knowledge_vector kv JOIN knowledge k ON k.id = kv.knowledge_id
            ORDER BY kv.chunk_vector <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, k),
        )
        return [dict(r) for r in cur.fetchall()]


def rerank(question: str, candidates: List[dict]) -> List[dict]:
    """Tambahkan rerank_score (0-1) ke setiap kandidat lalu urutkan dari yang paling relevan."""
    if not candidates:
        return []
    with rerank_lock:
        scores = np.asarray(reranker.predict([(question, c["chunk_text"]) for c in candidates]), dtype=np.float32)
    if scores.min() < 0 or scores.max() > 1:  # sebagian versi mengembalikan logit mentah
        scores = 1 / (1 + np.exp(-scores))
    for c, sc in zip(candidates, scores):
        c["rerank_score"] = float(sc)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def is_fallback(answer: str) -> bool:
    return FALLBACK.lower()[:40] in answer.lower()


NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_answer(answer: str, question: str, truncated: bool = False) -> str:
    """Rapikan pola rusak yang sering muncul pada model 8B ini: jawaban diawali salinan pertanyaan,
    kalimat yang sama diulang-ulang, dan kalimat terakhir yang terpotong karena max_tokens habis."""
    text = answer.strip()
    if truncated and not text.endswith((".", "!", "?")):
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if cut > 0:
            text = text[:cut + 1]
    q = question.strip().rstrip("?.! ")
    if q and normalize(text).startswith(normalize(q)):
        text = text[len(q):].lstrip(" ?.!:-\n")
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    seen, unique = set(), []
    for s in sentences:
        key = normalize(s)
        if key not in seen:
            seen.add(key)
            unique.append(s.strip())
    if len(unique) < len(sentences):  # ada kalimat berulang: pakai versi tanpa duplikat
        text = " ".join(unique)
    return text.strip()


def ungrounded_numbers(answer: str, context: str) -> List[str]:
    """Angka di jawaban yang tidak muncul di konteks. Harga, durasi, dan jumlah yang tidak ada
    di knowledge base tidak boleh disampaikan ke pelanggan."""
    ctx = set(NUM_RE.findall(context))
    ctx |= {n.replace(".", "").replace(",", "") for n in ctx}
    return [n for n in NUM_RE.findall(answer) if n not in ctx and n.replace(".", "").replace(",", "") not in ctx]


def is_unusable(answer: str, question: str) -> bool:
    """Jawaban yang tidak boleh masuk cache: penolakan, kosong, atau model hanya mengulang pertanyaan."""
    a = normalize(answer).rstrip("?.! ")
    return not a or is_fallback(answer) or a == normalize(question).rstrip("?.! ")


def build_prompt(question: str, context: str, style: Optional[str] = None) -> str:
    style = style or PROMPT_STYLE
    if style == "strict_refusal":
        return ALPACA_PROMPT.format(STRICT_INSTRUCTION, f"Konteks:\n{context}\n\nPertanyaan pelanggan:\n{question}", "")
    # Pola yang sama dengan format fine-tune Alpaca: pertanyaan sebagai instruksi, konteks sebagai input.
    instruction = (f"{PERSONA} Jawab pertanyaan pelanggan berikut secara singkat dan sopan, "
                   f"hanya berdasarkan informasi pada bagian Input.\n\nPertanyaan: {question}")
    return ALPACA_PROMPT.format(instruction, context, "")


def select_chunks(question: str, q_vec: str) -> List[dict]:
    ranked = rerank(question, retrieve(q_vec))
    return [c for c in ranked if c["rerank_score"] >= RERANK_THRESHOLD][:TOP_K]


def generate(question: str, temperature: float = CHAT_TEMPERATURE):
    """Retrieve + generate. Return (jawaban, vektor query, sumber, llm_dipanggil)."""
    q_vec = embed(question, "query")
    chunks = select_chunks(question, q_vec)
    if not chunks:
        return FALLBACK, q_vec, [], False

    context = "\n\n---\n\n".join(c["chunk_text"] for c in chunks)
    prompt = build_prompt(question, context)
    with llm_lock:
        out = llm(prompt, temperature=temperature, **GEN_KWARGS)
    choice = out["choices"][0]
    answer = clean_answer(choice["text"], question, truncated=choice.get("finish_reason") == "length")
    # Guardrail: jawaban yang memuat angka di luar konteks tidak disampaikan ke pelanggan
    if ungrounded_numbers(answer, context):
        return FALLBACK, q_vec, sorted({c["document_id"] for c in chunks}), True
    sources = sorted({c["document_id"] for c in chunks})
    return answer, q_vec, sources, True


def save_chat(instruction: str, response: str, input_vec: str) -> int:
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO chat_history (instruction, response, input_vector) VALUES (%s, %s, %s::vector) RETURNING id",
            (instruction, response, input_vec),
        )
        return cur.fetchone()[0]


def cache(chat_history_id: int):
    with get_cursor() as cur:
        cur.execute("INSERT INTO cached_chat (chat_history_id) VALUES (%s) ON CONFLICT DO NOTHING", (chat_history_id,))


def uncache(chat_history_id: int):
    with get_cursor() as cur:
        cur.execute("DELETE FROM cached_chat WHERE chat_history_id = %s", (chat_history_id,))


def chat_exists(chat_history_id: int) -> Optional[dict]:
    with get_cursor(dict_rows=True) as cur:
        cur.execute("SELECT id, instruction FROM chat_history WHERE id = %s", (chat_history_id,))
        return cur.fetchone()


def add_reaction(chat_history_id: int, reaction: str):
    with get_cursor() as cur:
        cur.execute("INSERT INTO analytics (type, chat_history_id) VALUES (%s, %s)", (reaction, chat_history_id))


# ---------------------------------------------------------------------------
# Schema request/response
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    instruction: str


class ChatResponse(BaseModel):
    response: str
    chat_history_id: int
    cached: bool
    sources: List[str]


class ReactionRequest(BaseModel):
    chat_history_id: int
    reaction: str


class RegenerateRequest(BaseModel):
    chat_history_id: int


# ---------------------------------------------------------------------------
# Endpoint (def biasa: FastAPI menjalankannya di threadpool, event loop tidak terblokir)
# ---------------------------------------------------------------------------
def require_admin(x_api_key: Optional[str] = Header(None)):
    """Hanya pemegang ADMIN_API_KEY yang boleh mengubah knowledge base."""
    if not ADMIN_API_KEY:
        raise HTTPException(503, "ADMIN_API_KEY belum diset di server.")
    if not x_api_key or not secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(401, "API key tidak valid.")


@app.post("/add_knowledge", dependencies=[Depends(require_admin)])
def add_knowledge(document_id: str = Query(...), file: UploadFile = File(...)):
    if file.content_type not in ("text/plain", "application/pdf"):
        raise HTTPException(400, "Hanya file PDF dan TXT yang didukung.")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Ukuran file maksimal 10 MB.")

    try:
        if file.content_type == "text/plain":
            text = content.decode("utf-8")
        else:
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        raise HTTPException(400, f"Gagal membaca file: {e}")

    chunks = chunk_words(text)
    if not chunks:
        raise HTTPException(400, "Tidak ada teks yang bisa diekstrak (PDF hasil scan perlu OCR).")

    vectors = [embed(c, "doc") for c in chunks]
    try:
        with get_cursor() as cur:
            cur.execute("INSERT INTO knowledge (document_id) VALUES (%s) RETURNING id", (document_id,))
            knowledge_id = cur.fetchone()[0]
            for c, v in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO knowledge_vector (knowledge_id, chunk_text, chunk_vector) VALUES (%s, %s, %s::vector)",
                    (knowledge_id, c, v),
                )
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(409, f"document_id '{document_id}' sudah ada.")
    return {"message": "Knowledge berhasil ditambahkan.", "document_id": document_id, "chunks": len(chunks)}


@app.get("/search")
def search(q: str = Query(...), k: int = Query(TOP_K, ge=1, le=20)):
    """Endpoint debug: skor cosine (pgvector) dan skor reranker, untuk mengkalibrasi RERANK_THRESHOLD."""
    ranked = rerank(q, retrieve(embed(q, "query"), RETRIEVE_K))[:k]
    return {"query": q, "rerank_threshold": RERANK_THRESHOLD,
            "results": [{**r, "similarity": round(float(r["similarity"]), 4),
                         "rerank_score": round(r["rerank_score"], 4)} for r in ranked]}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = normalize(request.instruction)
    with get_cursor(dict_rows=True) as cur:
        cur.execute(
            """
            SELECT ch.id, ch.response FROM chat_history ch
            JOIN cached_chat cc ON cc.chat_history_id = ch.id
            WHERE ch.instruction = %s
            ORDER BY cc.created_at DESC LIMIT 1
            """,
            (question,),
        )
        hit = cur.fetchone()
    if hit:
        return ChatResponse(response=hit["response"], chat_history_id=hit["id"], cached=True, sources=[])

    answer, q_vec, sources, llm_called = generate(question)
    chat_id = save_chat(question, answer, q_vec)
    # Hanya jawaban substantif yang di-cache. Jawaban penolakan (baik dari threshold maupun dari LLM)
    # tidak di-cache, supaya penolakan keliru atau knowledge baru tidak "terkunci" di cache.
    if llm_called and not is_unusable(answer, question):
        cache(chat_id)
    return ChatResponse(response=answer, chat_history_id=chat_id, cached=False, sources=sources)


@app.post("/regenerate", response_model=ChatResponse)
def regenerate(request: RegenerateRequest):
    old = chat_exists(request.chat_history_id)
    if not old:
        raise HTTPException(404, "Chat history tidak ditemukan.")
    answer, q_vec, sources, llm_called = generate(old["instruction"], temperature=REGENERATE_TEMPERATURE)
    new_id = save_chat(old["instruction"], answer, q_vec)
    add_reaction(old["id"], "regenerate")
    uncache(old["id"])
    if llm_called and not is_unusable(answer, old["instruction"]):
        cache(new_id)
    return ChatResponse(response=answer, chat_history_id=new_id, cached=False, sources=sources)


@app.post("/react")
def react(request: ReactionRequest):
    if request.reaction not in ("like", "dislike"):
        raise HTTPException(400, "Reaction harus 'like' atau 'dislike'.")
    if not chat_exists(request.chat_history_id):
        raise HTTPException(404, "Chat history tidak ditemukan.")
    add_reaction(request.chat_history_id, request.reaction)
    if request.reaction == "dislike":
        uncache(request.chat_history_id)  # jawaban yang di-dislike tidak disajikan lagi dari cache
    return {"message": "Reaction tersimpan."}


@app.get("/all-reactions")
def all_reactions(
    reaction_type: str = Query(..., pattern="^(like|dislike|regenerate)$"),
    start_datetime: Optional[str] = Query(None, description="YYYY-MM-DD HH:MM:SS"),
    end_datetime: Optional[str] = Query(None, description="YYYY-MM-DD HH:MM:SS"),
):
    query, params = "SELECT * FROM analytics WHERE type = %s", [reaction_type]
    if start_datetime:
        query += " AND created_at >= %s"
        params.append(start_datetime)
    if end_datetime:
        query += " AND created_at <= %s"
        params.append(end_datetime)
    with get_cursor(dict_rows=True) as cur:
        cur.execute(query + " ORDER BY created_at", params)
        return {"reactions": cur.fetchall()}
