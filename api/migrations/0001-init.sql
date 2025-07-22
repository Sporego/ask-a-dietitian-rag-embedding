CREATE EXTENSION vector;

CREATE TABLE books (
  id bigserial PRIMARY KEY,
  title varchar(128)
);

CREATE TABLE embeddings (
  id        bigserial PRIMARY KEY,
  embedding vector(2000) NOT NULL,
  book_id   bigserial REFERENCES books(id),
  page      integer NOT NULL,
  text      text NOT NULL
);
