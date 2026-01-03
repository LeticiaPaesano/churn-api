from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, Response

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

app = FastAPI(title="Churn API", version="2.0.0")

artifacts: Dict = {}

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)

    required = {"model", "scaler", "threshold_cost", "columns"}
    missing = required - set(loaded.keys())

    if missing:
        raise RuntimeError(f"Artefatos faltando no model.joblib: {missing}")

    artifacts.update(loaded)
    print("✅ Modelo carregado com sucesso")

# =========================================================
# ROOT / HEALTH / HEAD
# =========================================================

@app.get("/")
@app.head("/")
def root():
    return {
        "service": "Churn API",
        "status": "online",
        "model_loaded": bool(artifacts),
        "version": "2.0.0",
        "environment": "render"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(artifacts)
    }

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
# MAPA DE FEATURES DO MODELO -> CONTRATO DA API
# =========================================================
FEATURE_MAP = {
    "CreditScore": "CreditScore",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Balance",
    "EstimatedSalary": "EstimatedSalary",

    # Features derivadas
    "Age_Tenure": "Age",
    "Balance_Salary": "Balance",

    # Enums (One-Hot)
    "Geography_France": "Geography",
    "Geography_Spain": "Geography",
    "Geography_Germany": "Geography",
    "Gender_Male": "Gender",
    "Gender_Female": "Gender",
}


# =========================================================
# EXPLICABILIDADE LOCAL (TOP 3)
# =========================================================

def calcular_explicabilidade_local(
    X_scaled: np.ndarray,
    payload: Dict
) -> list[str]:
    model = artifacts["model"]
    features = artifacts["columns"]
    importances = model.feature_importances_

    impactos = importances * np.abs(X_scaled[0])

    impacto_por_contrato = {}

    for feature, impacto in zip(features, impactos):
        campo = FEATURE_MAP.get(feature)

        if not campo:
            continue

        impacto_por_contrato[campo] = (
            impacto_por_contrato.get(campo, 0) + impacto
        )

    ranking = sorted(
        impacto_por_contrato.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    explicabilidade = []

    for campo, _ in ranking:
        if campo in ("Geography", "Gender"):
            explicabilidade.append(payload[campo])
        else:
            explicabilidade.append(campo)

    return explicabilidade

# =========================================================
# ENDPOINT /previsao
# =========================================================

@app.post("/previsao")
def previsao(payload: Dict):
    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    colunas = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "EstimatedSalary",
    ]

    faltantes = set(colunas) - set(payload.keys())
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"Colunas ausentes: {list(faltantes)}"
        )

    df = pd.DataFrame([payload])
    df_proc = preparar_dataframe(df)
    X_scaled = artifacts["scaler"].transform(df_proc)

    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    threshold = artifacts["threshold_cost"]

    risco = "ALTO" if proba >= threshold else "BAIXO"
    previsao = "Vai cancelar" if risco == "ALTO" else "Vai permanecer"

    explicabilidade = calcular_explicabilidade_local(X_scaled, payload)

    return {
        "previsao": previsao,
        "probabilidade": round(proba, 4),
        "nivel_risco": risco,
        "recomendacao": (
            "Ação imediata recomendada: contato ativo e oferta personalizada"
            if risco == "ALTO"
            else "Cliente estável"
        ),
        "explicabilidade": explicabilidade
    }

# =========================================================
# PROCESSAMENTO EM BACKGROUND
# =========================================================

def processar_csv(job_id: str, input_path: Path):
    try:
        df = pd.read_csv(input_path)

        df_proc = preparar_dataframe(df)
        X_scaled = artifacts["scaler"].transform(df_proc)

        probs = artifacts["model"].predict_proba(X_scaled)[:, 1]
        threshold = artifacts["threshold_cost"]

        df["probabilidade"] = probs.round(4)
        df["nivel_risco"] = np.where(probs >= threshold, "ALTO", "BAIXO")
        df["previsao"] = np.where(
            df["nivel_risco"] == "ALTO",
            "Vai cancelar",
            "Vai permanecer"
        )

        output = TMP_DIR / f"{job_id}_resultado.csv"
        df.to_csv(output, index=False)

    except Exception as e:
        (TMP_DIR / f"{job_id}.error").write_text(str(e))

# =========================================================
# ENDPOINT PREVISÃO EM LOTE
# =========================================================

@app.post("/previsao-lote")
def previsao_lote(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
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

@app.get("/previsao-lote/status/{job_id}")
def status_lote(job_id: str):
    if (TMP_DIR / f"{job_id}.error").exists():
        return {"status": "ERRO"}

    if (TMP_DIR / f"{job_id}_resultado.csv").exists():
        return {"status": "FINALIZADO"}

    return {"status": "PROCESSANDO"}

@app.get("/previsao-lote/download/{job_id}")
def download(job_id: str):
    path = TMP_DIR / f"{job_id}_resultado.csv"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não disponível")

    return FileResponse(path, filename=path.name, media_type="text/csv")
