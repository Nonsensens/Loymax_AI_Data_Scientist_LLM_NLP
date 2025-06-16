import requests

API_URL = "http://localhost:8000/query"


def test_integration_pipeline():
    test_query = "Что такое язык программирования?"
    response = requests.post(API_URL, json={"query": test_query})

    assert response.status_code == 200, "API не вернул 200 OK"
    json_response = response.json()

    assert "answer" in json_response, "Нет поля 'answer' в ответе"
    assert len(json_response["answer"].strip()) > 0, "Ответ пустой"
