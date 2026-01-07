from pathlib import Path
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
import uuid
import joblib
import shutil
import tempfile
import threading
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
TMP_DIR = Path(tempfile.gettempdir())

# =========================================================
# APP
# =========================================================
app = FastAPI(title="Churn API", version="2.2.0")
artifacts: Dict = {}

# =========================================================
# STARTUP
# =========================================================
@app.on_event("startup")
def load_artifacts():
    artifacts.update(joblib.load(MODEL_PATH))
    print("✅ Modelo carregado")

# =========================================================
# ENDPOINT ROOT / HEALTH
# =========================================================
@app.get("/")
def root():
    return {"status": "online"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# =========================================================
# PREPARAÇÃO
# =========================================================
def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    for col in artifacts["columns"]:
        if col not in df.columns:
            df[col] = 0
    return df[artifacts["columns"]]

# =========================================================
# EXPLICABILIDADE
# =========================================================
FEATURE_MAP = {
    "CreditScore": "CreditScore",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Balance",
    "EstimatedSalary": "EstimatedSalary",
    "Geography_France": "Geography",
    "Geography_Germany": "Geography",
    "Geography_Spain": "Geography",
    "Gender_Male": "Gender",
}

def gerar_explicabilidade(X, df_original, idx):
    importances = artifacts["model"].feature_importances_
    impactos = importances * np.abs(X[idx])
    mapa = {}

    for f, i in zip(artifacts["columns"], impactos):
        campo = FEATURE_MAP.get(f)
        if campo:
            mapa[campo] = mapa.get(campo, 0) + i

    top = sorted(mapa, key=mapa.get, reverse=True)[:3]

    return ", ".join(
        df_original.iloc[idx][c] if c in ("Geography", "Gender") else c
        for c in top
    )

# =========================================================
# PROCESSAMENTO CSV
# =========================================================
def processar_csv(job_id: str, input_path: Path):
    status_file = TMP_DIR / f"{job_id}.status"
    try:
        df = pd.read_csv(input_path)
        original = df.copy()

        df_proc = preparar_dataframe(df)
        X = artifacts["scaler"].transform(df_proc)

        probs = artifacts["model"].predict_proba(X)[:, 1]
        df["probabilidade"] = probs.round(4)
        df["nivel_risco"] = np.where(
            probs >= artifacts["threshold_cost"], "ALTO", "BAIXO"
        )
        df["previsao"] = np.where(
            df["nivel_risco"] == "ALTO",
            "Vai cancelar",
            "Vai continuar"
        )
        df["explicabilidade"] = [
            gerar_explicabilidade(X, original, i)
            for i in range(len(df))
        ]

        df.to_csv(TMP_DIR / f"{job_id}_resultado.csv", index=False)
        status_file.write_text("FINALIZADO")

    except Exception as e:
        status_file.write_text("ERRO")
        (TMP_DIR / f"{job_id}.error").write_text(str(e))

# =========================================================
# ENDPOINT PREVISAO-LOTE
# =========================================================
@app.post("/previsao-lote")
def previsao_lote(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Arquivo deve ser CSV")

    job_id = str(uuid.uuid4())
    input_path = TMP_DIR / f"{job_id}.csv"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    (TMP_DIR / f"{job_id}.status").write_text("PROCESSANDO")

    thread = threading.Thread(
        target=processar_csv,
        args=(job_id, input_path),
        daemon=False
    )
    thread.start()

    return {"job_id": job_id, "status": "PROCESSANDO"}

# =========================================================
# ENDPOINT PREVISAO-LOTE/STATUS
# =========================================================
@app.get("/previsao-lote/status/{job_id}")
def status(job_id: str):
    status_file = TMP_DIR / f"{job_id}.status"

    if not status_file.exists():
        return {"status": "NAO_ENCONTRADO"}

    return {"status": status_file.read_text()}

# =========================================================
# ENDPOINT PREVISAO-LOTE/DOWNLOAD
# =========================================================
@app.get("/previsao-lote/download/{job_id}")
def download(job_id: str):
    path = TMP_DIR / f"{job_id}_resultado.csv"
    if not path.exists():
        raise HTTPException(404, "Arquivo não disponível")
    return FileResponse(path, filename=path.name, media_type="text/csv")
