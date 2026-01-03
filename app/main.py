
from pathlib import Path
from typing import Dict
from fastapi.responses import Response, FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks

import uuid
import joblib
import shutil
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

app = FastAPI(title="Churn API", version="1.0.0")

artifacts: Dict = {}

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)

    required_keys = {"model", "scaler", "threshold_cost", "columns"}
    missing = required_keys - set(loaded.keys())

    if missing:
        raise RuntimeError(f"Artefatos faltando no model.joblib: {missing}")

    artifacts.update(loaded)
    print("✅ Modelo carregado com sucesso")
    
# =========================================================
# HEADER STATUS
# =========================================================

@app.get("/")
def root():
    return {
        "service": "Churn API",
        "render": "ok",
        "api": "online",
        "modelo_carregado": bool(artifacts),
        "version": "1.0.0"
    }
 
 
 # =========================================================
# SILENCED FAVICON
# =========================================================   
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)   
# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(artifacts)
    }

# =========================================================
# PREPARAÇÃO DE DADOS
# =========================================================

def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-hot encoding básico
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    # Garantir colunas esperadas pelo modelo
    for col in artifacts["columns"]:
        if col not in df.columns:
            df[col] = 0

    return df[artifacts["columns"]]

# =========================================================
# PROCESSAMENTO EM BACKGROUND
# =========================================================

def processar_csv(job_id: str, input_path: Path):
    try:
        df = pd.read_csv(input_path)

        colunas_necessarias = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "EstimatedSalary",
        ]

        faltantes = set(colunas_necessarias) - set(df.columns)
        if faltantes:
            raise ValueError(f"Colunas ausentes: {list(faltantes)}")

        df_proc = preparar_dataframe(df)
        X_scaled = artifacts["scaler"].transform(df_proc)

        probs = artifacts["model"].predict_proba(X_scaled)[:, 1]
        threshold = artifacts["threshold_cost"]

        df["probabilidade"] = probs.round(4)
        df["nivel_risco"] = np.where(probs >= threshold, "ALTO", "BAIXO")
        df["previsao"] = np.where(df["nivel_risco"] == "ALTO", "Vai Sair", "Vai Ficar")

        output_path = TMP_DIR / f"{job_id}_resultado.csv"
        df.to_csv(output_path, index=False)

    except Exception as e:
        error_path = TMP_DIR / f"{job_id}.error"
        error_path.write_text(str(e))
        
# =========================================================
# ENDPOINT DE PREVISÃO
# =========================================================       
@app.post("/previsao")
def previsao(payload: Dict):
    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        df = pd.DataFrame([payload])

        colunas_necessarias = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "EstimatedSalary",
        ]

        faltantes = set(colunas_necessarias) - set(df.columns)
        if faltantes:
            raise HTTPException(
                status_code=400,
                detail=f"Colunas ausentes: {list(faltantes)}"
            )

        df_proc = preparar_dataframe(df)
        X_scaled = artifacts["scaler"].transform(df_proc)

        proba = float(
            artifacts["model"].predict_proba(X_scaled)[0, 1]
        )

        threshold = artifacts["threshold_cost"]

        risco = "ALTO" if proba >= threshold else "BAIXO"
        previsao = "Vai Sair" if risco == "ALTO" else "Vai Ficar"

        return {
            "previsao": previsao,
            "probabilidade": round(proba, 4),
            "nivel_risco": risco
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# ENDPOINT DE PREVISÃO EM LOTE
# =========================================================
@app.post("/previsao-lote")
def previsao_lote(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")

    job_id = str(uuid.uuid4())
    input_path = TMP_DIR / f"{job_id}.csv"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(processar_csv, job_id, input_path)

    return {
        "job_id": job_id,
        "status": "PROCESSANDO"
    }

# =========================================================

@app.get("/previsao-lote/status/{job_id}")
def status(job_id: str):
    result_path = TMP_DIR / f"{job_id}_resultado.csv"
    error_path = TMP_DIR / f"{job_id}.error"

    if error_path.exists():
        return {"status": "ERRO", "detail": error_path.read_text()}

    if result_path.exists():
        return {"status": "FINALIZADO"}

    return {"status": "PROCESSANDO"}

# =========================================================

@app.get("/previsao-lote/download/{job_id}")
def download(job_id: str):
    result_path = TMP_DIR / f"{job_id}_resultado.csv"

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo ainda não disponível")

    return FileResponse(
        path=result_path,
        filename=result_path.name,
        media_type="text/csv"
    )
