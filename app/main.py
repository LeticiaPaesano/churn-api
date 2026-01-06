from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
import uuid
import joblib
import tempfile
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

app = FastAPI(title="Churn API", version="3.1.0")
artifacts: Dict = {}

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def load_artifacts():
    loaded = joblib.load(MODEL_PATH)

    required = {"model", "scaler", "threshold_cost", "columns"}
    if not required.issubset(loaded):
        raise RuntimeError("Artefatos incompletos")

    artifacts.update(loaded)
    print("✅ Modelo carregado")

# =========================================================
# ENDPOINT ROOT / HEALTH
# =========================================================

@app.get("/")
@app.head("/")
def root():
    return {"service": "Churn API", "status": "online"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# =========================================================
# PREPARAÇÃO DE DADOS
# =========================================================

def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    for col in artifacts["columns"]:
        if col not in df.columns:
            df[col] = 0
    return df[artifacts["columns"]]

# =========================================================
# MAPEAMENTO DAS FEATURES
# =========================================================

FEATURE_MAP = {
    "CreditScore": "CreditScore",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Balance",
    "EstimatedSalary": "EstimatedSalary",
    "Age_Tenure": "Age",
    "Balance_Salary": "Balance",
    "Geography_France": "Geography",
    "Geography_Germany": "Geography",
    "Geography_Spain": "Geography",
    "Gender_Male": "Gender",
    "Gender_Female": "Gender",
}

# =========================================================
# EXPLICABILIDADE
# =========================================================

def calcular_explicabilidade_local(X_scaled, payload) -> List[str]:
    model = artifacts["model"]
    features = artifacts["columns"]
    importances = model.feature_importances_

    impactos = importances * abs(X_scaled[0])
    mapa = {}

    for feat, imp in zip(features, impactos):
        campo = FEATURE_MAP.get(feat)
        if campo:
            mapa[campo] = mapa.get(campo, 0) + imp

    ranking = sorted(mapa.items(), key=lambda x: x[1], reverse=True)[:3]

    resultado = []
    for campo, _ in ranking:
        if campo in ("Geography", "Gender"):
            resultado.append(str(payload.get(campo)))
        else:
            resultado.append(campo)

    return resultado

# =========================================================
# ENDPOINT PREVISAO 
# =========================================================

@app.post("/previsao")
def previsao(payload: Dict):
    df = pd.DataFrame([payload])
    df_proc = preparar_dataframe(df)
    X = artifacts["scaler"].transform(df_proc)

    proba = float(artifacts["model"].predict_proba(X)[0, 1])
    risco = "ALTO" if proba >= artifacts["threshold_cost"] else "BAIXO"

    return {
        "previsao": "Vai cancelar" if risco == "ALTO" else "Vai continuar",
        "probabilidade": round(proba, 4),
        "nivel_risco": risco,
        "explicabilidade": calcular_explicabilidade_local(X, payload),
    }

# =========================================================
# ENDPOINT PREVISAO-LOTE
# =========================================================

@app.post("/previsao-lote")
def previsao_lote(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    try:
        df = pd.read_csv(file.file)
        df_proc = preparar_dataframe(df)
        X = artifacts["scaler"].transform(df_proc)

        probs = artifacts["model"].predict_proba(X)[:, 1]
        thr = artifacts["threshold_cost"]

        df["probabilidade"] = probs.round(4)
        df["nivel_risco"] = np.where(probs >= thr, "ALTO", "BAIXO")
        df["previsao"] = np.where(
            df["nivel_risco"] == "ALTO",
            "Vai cancelar",
            "Vai continuar"
        )

        explic = []
        for i in range(len(df)):
            explic.append(
                ", ".join(
                    calcular_explicabilidade_local(X[i:i+1], df.iloc[i].to_dict())
                )
            )

        df["explicabilidade"] = explic

        output = TMP_DIR / f"{job_id}_resultado.csv"
        df.to_csv(output, index=False)

        return {"job_id": job_id, "status": "FINALIZADO"}

    except Exception as e:
        (TMP_DIR / f"{job_id}.error").write_text(str(e))
        return {"job_id": job_id, "status": "ERRO"}

# =========================================================
# ENDPOINT PREVISAO-LOTE/STATUS
# =========================================================

@app.get("/previsao-lote/status/{job_id}")
def status_lote(job_id: str):
    if (TMP_DIR / f"{job_id}.error").exists():
        return {"status": "ERRO"}

    if (TMP_DIR / f"{job_id}_resultado.csv").exists():
        return {"status": "FINALIZADO"}

    return {"status": "NAO_ENCONTRADO"}

# =========================================================
# ENDPOINT PREVISAO-LOTE/DOWNLOAD
# =========================================================

@app.get("/previsao-lote/download/{job_id}")
def download_lote(job_id: str):
    path = TMP_DIR / f"{job_id}_resultado.csv"
    if not path.exists():
        raise HTTPException(404, "Arquivo não disponível")

    return FileResponse(path, filename=path.name, media_type="text/csv")
