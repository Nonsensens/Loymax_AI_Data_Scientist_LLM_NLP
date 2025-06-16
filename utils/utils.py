import re


def preprocess_text(text: str) -> str:
    """
    Очистка и нормализация текста:
    - перевод в нижний регистр,
    - удаление лишних пробелов,
    - удаление спецсимволов (кроме точек и запятых, если нужно),
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.]', '', text)
    return text.strip()