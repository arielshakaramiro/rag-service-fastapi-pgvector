import psycopg2
from psycopg2 import sql
from config import DB_CONFIG, EMBED_DIM

TABLES = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge (
    id SERIAL PRIMARY KEY,
    document_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_vector (
    id SERIAL PRIMARY KEY,
    knowledge_id INT REFERENCES knowledge(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_vector VECTOR({EMBED_DIM}) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS knowledge_vector_hnsw
    ON knowledge_vector USING hnsw (chunk_vector vector_cosine_ops);

CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    instruction TEXT NOT NULL,
    input_data TEXT NOT NULL DEFAULT '',
    response TEXT NOT NULL,
    input_vector VECTOR({EMBED_DIM}),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cached_chat (
    id SERIAL PRIMARY KEY,
    chat_history_id INT UNIQUE REFERENCES chat_history(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics (
    id SERIAL PRIMARY KEY,
    type VARCHAR(20) NOT NULL CHECK (type IN ('like', 'dislike', 'regenerate')),
    chat_history_id INT REFERENCES chat_history(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def setup_database_and_tables():
    admin_cfg = {**DB_CONFIG, "dbname": "postgres"}
    conn = psycopg2.connect(**admin_cfg)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG["dbname"],))
        if cur.fetchone() is None:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_CONFIG["dbname"])))
            print(f"Database '{DB_CONFIG['dbname']}' dibuat.")
        else:
            print(f"Database '{DB_CONFIG['dbname']}' sudah ada.")
    conn.close()

    conn = psycopg2.connect(**DB_CONFIG)
    with conn, conn.cursor() as cur:
        cur.execute(TABLES)
    conn.close()
    print("Semua tabel dan index siap.")


if __name__ == "__main__":
    setup_database_and_tables()
