import typing
import fastapi
import psycopg
import psycopg.rows
import psycopg_pool
import pydantic

import config
import migrations.migrator as migrator

app = fastapi.FastAPI()

pool = psycopg_pool.ConnectionPool (
        "postgresql://{user}:{password}@{host}:{port}/{name}".format(
            user=config.DB_USER, password=config.DB_PASS,
            host=config.DB_HOST, port=config.DB_PORT,
            name=config.DB_NAME
            ),
        num_workers=3,
    )

# T = typing.TypeVar('T')
# P = typing.ParamSpec('P')
# ConnFunc = typing.Callable[typing.Concatenate[psycopg.Connection, P], T]

# def with_db_connection(func: ConnFunc[P, T]) -> typing.Callable[P, T]:
#     @functools.wraps(func)
#     def wrapper(*args : P.args, **kwargs: P.kwargs) -> T:
#         with pool.connection() as conn:
#             conn.row_factory = psycopg.rows.dict_row
#             return func(conn, *args, **kwargs)
#     return wrapper

class Question(pydantic.BaseModel):
    query: str


@app.middleware("http")
async def db_session_middleware(request: fastapi.Request, call_next):
    request.state.conn_pool = pool
    response = await call_next(request)
    return response

@app.on_event("startup")
async def startup():
    pool.open(wait=True)
    with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        migrator.run_migrations(conn)

@app.on_event("shutdown")
async def shutdown():
    pool.close()
    # cleanup
    pass

@app.get("/")
def read_root(request: fastapi.Request):
    with request.state.conn_pool.connection() as conn:
        print(conn.execute("SELECT * FROM embeddings").fetchall())
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: typing.Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/question/")
def read_question(request: fastapi.Request, question: Question):
    with request.state.conn_pool.connection() as conn:
        # convert user query to vector
        user_vec = stub_embed_user_query(question.query)

        # lookup similar vectors in db
        stub_get_similar_vectors(user_vec)

        # pass off request to chatgpt
        return stub_chatgpt_summorize(question.query, [])

def stub_embed_user_query(query: str):
    pass

def stub_get_similar_vectors(vec: any):
    # res = conn.execute("SELECT * FROM embeddings ORDER BY embedding <=> %s LIMIT 5;", user_vec).fetchall()
    pass

def stub_chatgpt_summorize(user_query: str, contexts: typing.List[typing.OrderedDict[str, str]]):
    pass
