# Loymax AI Q&A System

## 1. Описание проекта и его цели

Цель проекта — разработать систему вопросов и ответов (Q&A), основанную на Retrieval-Augmented Generation (RAG). Система позволяет пользователям задавать произвольные вопросы, а модель находит релевантную информацию в локальной базе и отвечает, используя только эту информацию. Подходит для корпоративных знаний, внутренней документации и т.п.

---

## 2. Архитектурная схема

```
                  ┌────────────────────┐
                  │  Источник данных   │
                  │  (CSV / JSON)      │
                  └────────┬───────────┘
                           ↓
                    ┌──────────────┐
                    │ Preprocessing│
                    └────┬─────────┘
                         ↓
            ┌──────────────────────────────┐
            │ Векторизация (HF Embeddings) │
            └────────┬─────────────────────┘
                     ↓
           ┌───────────────────────────────┐
           │  Хранение в Chroma (DB Dir)   │
           └────────┬──────────────────────┘
                    ↓
         ┌───────────────────────────────┐
         │       FastAPI REST API        │
         │  /query → RAG → HF LLM Output │
         └───────────────────────────────┘
```

---

## 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

---

## 4. Инструкция по запуску

### 4.1 Локальный запуск

```bash
git clone https://github.com/Nonsensens/Loymax_AI_Data_Scientist_LLM_NLP.git
cd your-repo

pip install -r requirements.txt

cp config/config.env.example config/config.env
# Отредактируйте config.env, указав свои переменные

# Запуск индексации (создание векторной БД)
python -m indexing.indexingService

# Запуск API
python -m api.apiService
```

### 4.2 Запуск через Docker

```bash
docker-compose up --build
```

---

## 5. Использованные технологии и модели

- **FastAPI** — современный и быстрый веб-фреймворк для API.
- **HuggingFace Inference API** — для генерации ответов LLM без локального запуска.
- **SentenceTransformers (all-MiniLM-L6-v2)** — для генерации эмбеддингов.
- **LangChain** — упрощает построение пайплайна RAG.
- **ChromaDB** — векторная база данных для хранения эмбеддингов, удобна для MVP.
- **Docker** — контейнеризация и простое развёртывание.
- **Pytest** — тестирование.

---

## 6. API Endpoint

### POST `/query`

**Формат запроса:**

```json
{
  "query": "Что такое НДС?"
}
```

**Формат ответа:**

```json
{
  "answer": "НДС — это налог на добавленную стоимость, который..."
}
```

---

## 7. Анализ данных и проверка качества

- Удаление пустых и дублирующихся записей
- Фильтрация слишком коротких текстов
- Предобработка текста (очистка, нормализация)
- Разбиение на чанки с перекрытием (500 символов с шагом 100)
- Векторизация и сохранение в ChromaDB

---

## 8. Тестирование

### 8.1 Юнит-тесты

Расположены в папке `tests/`. Проверяют препроцессинг, индексацию и построение промпта.

Запуск:

```bash
pytest tests/
```

### 8.2 Интеграционный тест пайплайна

Файл: `tests/test_integration_pipeline.py`

Пройти его можно только после запуска сервисов в Docker
Пример теста, отправляющего запрос к API и проверяющего ответ:

```python
import requests

API_URL = "http://localhost:8000/query"


def test_integration_pipeline():
    test_query = "Что такое язык программирования?"
    response = requests.post(API_URL, json={"query": test_query})

    assert response.status_code == 200, "API не вернул 200 OK"
    json_response = response.json()

    assert "answer" in json_response, "Нет поля 'answer' в ответе"
    assert len(json_response["answer"].strip()) > 0, "Ответ пустой"
```

Запуск:

```bash
pytest tests/test_integration_pipeline.py
```

---

## 9. Масштабирование и улучшения

### Горизонтальное масштабирование

- Запуск нескольких реплик FastAPI-сервиса за балансировщиком нагрузки (nginx, traefik).
- Использование docker-compose с директивой `replicas`.

### Использование более производительной векторной базы

- Заменить ChromaDB на:
  - **FAISS** — быстрая локальная библиотека.
  - **Qdrant** — с поддержкой фильтров и кластеризации.
  - **Milvus** — масштабируемая система.
  - **Weaviate** — поддержка семантического поиска и метаданных.

### Оптимизация LLM

- Запуск модели локально через vLLM или text-generation-inference.
- Включение стриминга ответов.

### Дополнительные улучшения

- Внедрение ранжировщика (reranker) для повышения качества релевантности.
- Кэширование популярных запросов (Redis).
- Мониторинг и логирование (Prometheus + Grafana).
- Автоматизация CI/CD.

---

## 10. Переменные окружения (.env)

```env
DATA_PATH=./data/
CHROMA_DB_DIR=./chroma_db/
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL_NAME=OpenChat/openchat_3.5
TOKEN_HH=your_huggingface_api_token
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MIN_TEXT_LENGTH=100
API_HOST=0.0.0.0
API_PORT=8000
```

---
