from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

BASE_PAYLOAD = {
    "CreditScore": 350,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 55,
    "Tenure": 2,
    "Balance": 150000,
    "EstimatedSalary": 40000
}

def test_churn_with_explainability():
    response = client.post("/previsao", json=BASE_PAYLOAD)
    assert response.status_code == 200

    data = response.json()

    assert data["previsao"] == "Vai cancelar"
    assert data["explicabilidade"] is not None
    assert isinstance(data["explicabilidade"], list)
    assert 1 <= len(data["explicabilidade"]) <= 3


def test_no_churn_without_explainability():
    payload = BASE_PAYLOAD.copy()
    payload["CreditScore"] = 900
    payload["Balance"] = 0

    response = client.post("/previsao", json=payload)
    assert response.status_code == 200

    data = response.json()

    assert data["previsao"] == "Vai continuar"
    assert data["explicabilidade"] is None
