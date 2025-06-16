import os
import glob
import hashlib
import logging
import pandas as pd
import io
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from utils.utils import preprocess_text

load_dotenv("./config/config.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    force=True,  # перезаписать существующую конфигурацию
)


def load_data() -> pd.DataFrame:
    """Загрузка из файла/папки JSON/CSV с колонкой 'text'."""
    path = os.getenv("DATA_PATH")
    files = []
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.json")) + glob.glob(
            os.path.join(path, "*.csv")
        )
    else:
        files = [path]

    dfs = []
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".json":
            dfs.append(pd.read_json(file))
        elif ext == ".csv":
            dfs.append(pd.read_csv(file))
    if not dfs:
        raise ValueError("Нет данных для загрузки")
    df = pd.concat(dfs, ignore_index=True)
    if "text" not in df.columns:
        raise ValueError("В данных нет колонки 'text'")
    df["text_length"] = df["text"].apply(len)
    logging.info(f"Загружено {len(df)} документов")
    return df


def EDA_data(df: pd.DataFrame, output_path: str = "indexing/eda_output.md") -> None:
    """
    Выполняет базовый EDA и сохраняет результаты в Markdown-файл.
    Также логирует ключевые метрики.
    """
    buffer = []

    buffer.append("# Exploratory Data Analysis (EDA)\n")

    # Info
    buffer.append("## Dataset Info\n")
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    buffer.extend(
        ["```\n"]
        + [line + "\n" for line in info_buf.getvalue().splitlines()]
        + ["```\n"]
    )
    logging.info("Информация о датафрейме собрана.")

    # Text length
    buffer.append("## Text Length Statistics\n")
    desc = df["text_length"].describe()
    buffer.append(desc.to_markdown())
    logging.info(f"Статистика длины текста:\n{desc}")

    # Nulls
    buffer.append("\n## Null Values\n")
    nulls = df.isnull().sum()
    buffer.append(nulls.to_markdown())
    logging.info(f"Пропущенные значения:\n{nulls}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines([line + "\n" for line in buffer])

    logging.info(f"EDA завершен и сохранен в '{output_path}'")
    return "Анализ зевершен"


def data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверка качества данных:
    - Удаление пустых текстов
    - Удаление дубликатов по 'id' (если есть) и по хешу текста
    - Отсев текстов короче min_length символов
    """
    initial_count = len(df)
    min_length = int(os.getenv("MIN_TEXT_LENGTH"))
    logging.info(f"Начальное количество документов: {initial_count}")

    # 1. Удаляем пустые тексты
    df = df[df["text"].astype(str).str.strip().astype(bool)]
    after_empty_removal = len(df)
    logging.info(
        f"После удаления пустых текстов: {after_empty_removal} ({initial_count - after_empty_removal} удалено)"
    )

    # 2. Удаляем дубликаты по 'id', если есть
    if "id" in df.columns:
        before_dedup_id = len(df)
        df = df.drop_duplicates(subset=["id"])
        after_dedup_id = len(df)
        logging.info(
            f"После удаления дубликатов по id: {after_dedup_id} ({before_dedup_id - after_dedup_id} удалено)"
        )

    # 3. Удаляем дубликаты по хешу текста
    def hash_text(text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    df["text_hash"] = df["text"].apply(hash_text)
    before_dedup_text = len(df)
    df = df.drop_duplicates(subset=["text_hash"])
    after_dedup_text = len(df)
    logging.info(
        f"После удаления дубликатов по тексту: {after_dedup_text} ({before_dedup_text - after_dedup_text} удалено)"
    )
    df = df.drop(columns=["text_hash"])

    # 4. Фильтрация по минимальной длине текста
    df["text"] = df["text"].astype(str)
    before_length_filter = len(df)
    df = df[df["text"].str.len() >= min_length]
    after_length_filter = len(df)
    logging.info(
        f"После фильтрации по длине >= {min_length}: {after_length_filter} ({before_length_filter - after_length_filter} удалено)"
    )

    logging.info(f"Итоговое количество документов после проверки качества: {len(df)}")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Загружает данные, нормализует текст и проводит базовую фильтрацию на пустой текст и дубликаты.
    """
    # нормализация текст
    df["text"] = df["text"].apply(preprocess_text)
    # Убираем пустые
    df = df[df["text"].str.strip().astype(bool)]
    # Удаление дубликатов по тексту
    df = df.drop_duplicates(subset=["text"])
    return df


def vectorize_and_save(df: pd.DataFrame):
    """
    Векторизация данных и сохранение в Chroma
    """
    persist_dir = os.getenv("CHROMA_DB_DIR")
    chunk_size = int(os.getenv("CHUNK_SIZE"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
    documents = [Document(page_content=t) for t in df["text"]]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

    # Попытка загрузить существующую коллекцию, чтобы добавить новые данные
    if os.path.exists(persist_dir):
        try:
            db = Chroma(
                persist_directory=persist_dir, embedding_function=embedding_model
            )
            # Получаем все текущие документы и их хэши
            existing = db.get()
            existing_hashes = set(
                hashlib.md5(doc.encode("utf-8")).hexdigest()
                for doc in existing["documents"]
            )
            # Фильтруем новые чанки
            unique_chunks = []
            for doc in chunks:
                h = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
                if h not in existing_hashes:
                    unique_chunks.append(doc)
            if unique_chunks:
                logging.info(f"Добавляется уникальных документов: {len(unique_chunks)}")
                db.add_documents(unique_chunks)
            else:
                logging.info("Нет новых документов для добавления")
        except Exception as e:
            logging.error(f"Ошибка загрузки Chroma DB: {e}")
            db = None
    else:
        db = Chroma.from_documents(
            chunks, embedding_model, persist_directory=persist_dir
        )
    retriever = db.as_retriever()
    logging.info(f"Коллекция содержит документов: {db._collection.count()}")
    return db, retriever


def indexing():
    """
    Полная индексация данных
    """
    df = load_data()
    EDA_data(df)
    df = prepare_data(df)[:50]
    df = data_quality_checks(df)
    db, retriever = vectorize_and_save(df)
    logging.info("Индексирование завершено")
    return db, retriever


if __name__ == "__main__":
    indexing()
