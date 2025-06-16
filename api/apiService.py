from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import requests
import logging
import uvicorn
import os
from utils.utils import preprocess_text
from indexing.indexingService import indexing
from dotenv import load_dotenv

load_dotenv("./config/config.env")

# Логируем
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

app = FastAPI()

# Модель входящего запроса
class QueryRequest(BaseModel):
    query: str


def ask_llm(query):
    # Здесь мы используем модель через HF Inference API.
    API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('TOKEN_HH')}"}
    payload = {
        "messages": [{"role": "user", "content": query}],
        "model": os.getenv("LLM_MODEL_NAME"),
    }

    response = requests.post(API_URL, headers=headers, json=payload).json()
    logging.info("Ответ на запрос выдан")
    return response


def build_prompt(query: str, docs: list[Document]) -> str:
    # Формируем промпт из вопроса и релевантных чанков
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
            Ты бот для тестового задания\n
            Отвечай на вопросы только из котнекста\n
            Не используй НИКОГДА информацию вне контекста\n
            Если информации нет в котнексте - скажи об этом прямо\n
            Контекст:\n{context}\n"
            Вопрос:\n{query}
            """
    return prompt


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if os.path.exists(os.getenv("CHROMA_DB_DIR")): 
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL")
            )
            db = Chroma(
                persist_directory=os.getenv("CHROMA_DB_DIR"),
                embedding_function=embedding_model,
            )
            retriever = db.as_retriever()
        except Exception as e:
            logging.error(f"Ошибка при загрузке существующей коллекции: {e}")
            db = None
    else:
        db, retriever = indexing()

    if db is None:
        raise HTTPException(status_code=500, detail="Сервис не инициализирован")

    query = preprocess_text(request.query)
    logging.info(f"Получен запрос: {query}")

    # Получаем релевантные документы
    docs = retriever.invoke(query)
    if not docs:
        return {"answer": "По вашему запросу ничего не найдено."}

    prompt = build_prompt(query, docs)

    # Генерация ответа LLM
    outputs = ask_llm(prompt)
    answer = outputs["choices"][0]["message"]["content"]

    return {"answer": answer}


if __name__ == "__main__":
    # Инициализация при старте сервиса
    logging.info("Запуск API сервера...")
    uvicorn.run(app, host=os.getenv("API_HOST"), port=int(os.getenv("API_PORT")))
