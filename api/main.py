#from config import FILBERT_TOKEN
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import psycopg


class QuestionModel(BaseModel):
    question: str

class EmbeddingDatabase():
    async def create_pool(self):
        await psycopg.create_pool()
        #self.pool = await asyncpg.create_pool(dsn='MYDB_DSN')

        self.pool = await psycopg.pool.SimpleConnectionPool(
            minconn=2,
            maxconn=3,
            user='test',
            password='test',
            host='localhost:5432',
            port='5432',
            database='embedding'
        )


def create_app():

    app = FastAPI()
    db = EmbeddingDatabase()

    @app.middleware("http")
    async def db_session_middleware(request: Request, call_next):
        request.state.pgpool = db.pool
        response = await call_next(request)
        return response

    @app.on_event("startup")
    async def startup():
        await db.create_pool()
     
    @app.on_event("shutdown")
    async def shutdown():
        # cleanup
        pass



    @app.get("/")
    def read_root(request: Request):
        print(request.state.pool)
        return {"Hello": "World"}


    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: Union[str, None] = None):
        return {"item_id": item_id, "q": q}

    @app.post("/question/")
    def read_question(question: QuestionModel):
        #Define our connection string
        conn_string = "host='localhost:5432' dbname='embedding' user='test' password='test'"

        # print the connection string we will use to connect
        print("Connecting to database\n	->" + str(conn_string)) 

        # get a connection, if a connect cannot be made an exception will be raised here
        conn = psycopg.connect(conn_string)

        # conn.cursor will return a cursor object, you can use this cursor to perform queries
        cursor = conn.cursor()
        print("Connected!\n")
        


        return question.question
    
    return app


app = create_app()
