from requests import post


while True:
    query = input('Введи ваш запрос')
    outputs = post("http://0.0.0.0:8000/query", json={"query": query}).json()
    answer = outputs["answer"]
    print(answer)
