def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "UP"
    assert body["model_loaded"] is True
