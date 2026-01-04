def test_previsao_sucesso(client, payload_valido):
    r = client.post("/previsao", json=payload_valido)
    assert r.status_code == 200

    body = r.json()
    assert "previsao" in body
    assert "probabilidade" in body
    assert "nivel_risco" in body
    assert "explicabilidade" in body

    assert isinstance(body["explicabilidade"], list)
    assert len(body["explicabilidade"]) == 3
