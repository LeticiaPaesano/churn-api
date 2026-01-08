def test_previsao_logica_explicabilidade(client, payload_valido):
    r = client.post("/previsao", json=payload_valido)
    assert r.status_code == 200
    res = r.json()
    
    if res["previsao"] == "Vai cancelar":
        assert len(res["explicabilidade"]) == 3
    else:
        assert len(res["explicabilidade"]) == 0