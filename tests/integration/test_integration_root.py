def test_root_status_e_versao(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == "12.1.0"
    assert data["model_loaded"] is True