def test_previsao_churn_com_explicabilidade(client):
    payload = {
        "Surname": "Silva",
        "CreditScore": 400,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 55,
        "Tenure": 2,
        "Balance": 3000,
        "EstimatedSalary": 1500
    }

    response = client.post("/previsao", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert body["previsao"] == "Vai cancelar"
    assert body["nivel_risco"] == "ALTO"
    assert "explicabilidade" in body
    assert len(body["explicabilidade"]) == 3
