def test_previsao_payload_incompleto(client):
    r = client.post("/previsao", json={"CreditScore": 400})
    assert r.status_code == 400
