# ask-a-dietitian-rag-embedding

# Running locally

1. `poetry config virtualenvs.in-project true`
1. From the root, run `docker compose up -d`. When done, `docker compose down` will teardown the DB
1. Connect to the database `psql postgresql://test@127.0.0.1:5432/dev`:
    - Addr: `localhost:5432`
    - User: `test`
    - Pass (not part of the connection string): `test`
    - DB name: `dev`
1. Run `make dev`
