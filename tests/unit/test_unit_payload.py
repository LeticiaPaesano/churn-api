def test_payload_limites_invalidos(client):
    payload_errado = {
        "CreditScore": 600, "Geography": "France", "Gender": "Male",
        "Age": 93, "Tenure": 5, "Balance": 100, "EstimatedSalary": 5000
    }
    r = client.post("/previsao", json=payload_errado)
    assert r.status_code == 422

def test_payload_geography_invalida(client):
    payload_errado = {
        "CreditScore": 600, "Geography": "Brazil", "Gender": "Male",
        "Age": 30, "Tenure": 5, "Balance": 100, "EstimatedSalary": 5000
    }
    r = client.post("/previsao", json=payload_errado)
    assert r.status_code == 422