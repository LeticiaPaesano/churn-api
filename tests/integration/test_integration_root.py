def test_root_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "online"
    assert body["model_loaded"] is True
