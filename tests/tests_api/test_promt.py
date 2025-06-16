from types import SimpleNamespace
from api.apiService import build_prompt

def test_create_prompt():
    question = "Что такое Python?"
    chunks = [
        SimpleNamespace(page_content="Python — язык программирования."),
        SimpleNamespace(page_content="Он популярен.")
    ]
    prompt = build_prompt(question, chunks)
    print(prompt)

    assert "Что такое Python?" in prompt
    assert "Python — язык программирования." in prompt
    assert "Он популярен." in prompt
    assert "Контекст:" in prompt
