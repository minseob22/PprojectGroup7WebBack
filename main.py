# main.py
from fastapi import FastAPI

app = FastAPI()

# 기본 경로 ("/")로 접속했을 때 실행될 함수
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!", "status": "success"}

# "/items/5" 처럼 데이터를 요청받을 때
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "name": "Test Item"}