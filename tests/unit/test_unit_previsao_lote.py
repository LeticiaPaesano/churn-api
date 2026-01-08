import io
import time

def test_fluxo_lote_completo(client):
    csv_content = (
        "CreditScore,Geography,Gender,Age,Tenure,Balance,EstimatedSalary\n"
        "600,France,Male,40,5,10000,50000\n"
        "300,Germany,Female,20,2,5000,30000" 
    )
    file = io.BytesIO(csv_content.encode("utf-8"))
    
    r_post = client.post("/previsao-lote", files={"file": ("dados.csv", file, "text/csv")})
    assert r_post.status_code == 200
    job_id = r_post.json()["job_id"]

    for _ in range(10):
        r_status = client.get(f"/previsao-lote/status/{job_id}")
        if r_status.json()["status"] == "FINALIZADO":
            break
        time.sleep(1)
 
    r_down = client.get(f"/previsao-lote/download/{job_id}")
    assert r_down.status_code == 200
    assert "text/csv" in r_down.headers["content-type"]