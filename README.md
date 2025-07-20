# ask-a-dietitian-rag-embedding

# Running locally

1. `poetry config virtualenvs.in-project true`
1. From the root, run `docker compose up -d`
1. Connect to the database `psql postgresql://test@127.0.0.1:5432/embedding`:
    - Addr: `localhost:5432`
    - User: `test`
    - Pass (not part of the connection string): `test`
    - DB name: `embedding`
