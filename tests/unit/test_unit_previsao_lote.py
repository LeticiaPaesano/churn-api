import io

def test_previsao_lote_fluxo_completo(client):
    csv = """CreditScore,Geography,Gender,Age,Tenure,Balance,EstimatedSalary
500,Spain,Female,60,1,50000,120000
420,Germany,Male,55,0,150000,90000
"""

    file = io.BytesIO(csv.encode("utf-8"))

    r = client.post(
        "/previsao-lote",
        files={"file": ("test.csv", file, "text/csv")}
    )

    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # polling simples
    for _ in range(10):
        status = client.get(f"/previsao-lote/status/{job_id}").json()
        if status["status"] == "FINALIZADO":
            break

    download = client.get(f"/previsao-lote/download/{job_id}")
    assert download.status_code == 200
    assert download.headers["content-type"] == "text/csv"
