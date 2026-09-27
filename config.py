import os

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "rag_emerald"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}
EMBED_DIM = 768
