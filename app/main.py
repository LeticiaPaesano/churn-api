from pathlib import Path
from typing import Dict, List, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
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
# SCHEMA DE VALIDAÇÃO
# =========================================================
class ChurnPayload(BaseModel):
    CreditScore: int = Field(..., ge=350, le=850)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=92)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0, le=500000)
    EstimatedSalary: float

# =========================================================
# APP
# =========================================================
app = FastAPI(title="Churn API", version="12.1.0")
artifacts: Dict = {}
model_loaded = False

# =========================================================
# STARTUP
# =========================================================
@app.on_event("startup")
def load_artifacts():
    global model_loaded
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)

    if "threshold" in loaded:
        loaded["threshold_cost"] = loaded["threshold"]

    required = {
        "model", 
        "scaler", 
        "columns", 
        "threshold_cost", 
        "balance_median",
        "salary_median"
    }

    missing = required - set(loaded.keys())
    if missing:
        raise RuntimeError(f"Erro: Faltam chaves no model.joblib: {missing}")

    artifacts.update(loaded)
    model_loaded = True
    
    print("✅ Artefatos carregados com sucesso!")

# =========================================================
# ENDPOINT / e /HEAD
# =========================================================
@app.get("/")
@app.head("/")
def root():
    return {
        "service": "Churn API",
        "status": "online",
        "model_loaded": model_loaded,
        "version": "12.1.0",
        "environment": "render"
    }

# =========================================================
# ENDPOINT /HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

# =========================================================
# ENDPOINT /favicon.ico
# =========================================================
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# =========================================================
# PREPARAÇÃO DE DADOS
# =========================================================
def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    for col in artifacts["columns"]:
        if col not in df_proc.columns:
            df_proc[col] = 0
    return df_proc[artifacts["columns"]]

# =========================================================
# MAPA DE FEATURES
# =========================================================
FEATURE_MAP = {
    "CreditScore": "CreditScore", "Age": "Age", "Tenure": "Tenure",
    "Balance": "Balance", "EstimatedSalary": "EstimatedSalary",
    "Age_Tenure": "Age", "Balance_Salary": "Balance",
    "Geography_France": "Geography", "Geography_Spain": "Geography",
    "Geography_Germany": "Geography", "Gender_Male": "Gender",
    "Gender_Female": "Gender",
}

# =========================================================
# EXPLICABILIDADE LOCAL
# =========================================================
def calcular_explicabilidade_local(X_scaled: np.ndarray, payload_dict: Dict) -> list[str]:
    model = artifacts["model"]
    features = artifacts["columns"]
    importances = model.feature_importances_
    impactos = importances * np.abs(X_scaled[0])
    
    impacto_por_contrato = {}
    for feature, impacto in zip(features, impactos):
        campo = FEATURE_MAP.get(feature)
        if campo:
            impacto_por_contrato[campo] = impacto_por_contrato.get(campo, 0) + impacto
    
    ranking = sorted(impacto_por_contrato.items(), key=lambda x: x[1], reverse=True)
    
    explicabilidade = []
    seen = set()
    for campo, _ in ranking:
        if campo not in seen:
            seen.add(campo)
            val = str(payload_dict[campo]) if campo in ("Geography", "Gender") else campo
            explicabilidade.append(val)
        if len(explicabilidade) == 3: break
    return explicabilidade

# =========================================================
# ENDPOINT /PREVISAO
# =========================================================
@app.post("/previsao")
def previsao(payload: ChurnPayload):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    data = payload.dict()
    df = pd.DataFrame([data])
    df_proc = preparar_dataframe(df)
    X_scaled = artifacts["scaler"].transform(df_proc)
    
    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    risco = "ALTO" if proba >= artifacts["threshold_cost"] else "BAIXO"
    previsao_txt = "Vai cancelar" if risco == "ALTO" else "Vai continuar"
    
    explicabilidade = calcular_explicabilidade_local(X_scaled, data) if risco == "ALTO" else []
    
    return {
        "previsao": previsao_txt,
        "probabilidade": round(proba, 4),
        "nivel_risco": risco,
        "explicabilidade": explicabilidade
    }

# =========================================================
# PROCESSAMENTO EM BACKGROUND
# =========================================================
def obter_explicabilidade_lote(X_scaled: np.ndarray, chunk_df: pd.DataFrame, mask_cancelar: np.ndarray) -> List[str]:
    model = artifacts["model"]
    features = artifacts["columns"]
    importances = model.feature_importances_
    impactos_matriz = np.abs(X_scaled) * importances
    nomes_campos = np.array([FEATURE_MAP.get(f, f) for f in features])
    
    resultados = []
    for i in range(impactos_matriz.shape[0]):
        if not mask_cancelar[i]:
            resultados.append("")
            continue
        indices = np.argsort(impactos_matriz[i])[::-1]
        final_names, seen = [], set()
        for idx in indices:
            nome = nomes_campos[idx]
            if nome not in seen:
                seen.add(nome)
                final_names.append(str(chunk_df.iloc[i][nome]) if nome in ["Geography", "Gender"] else nome)
            if len(final_names) == 3: break
        resultados.append(", ".join(final_names))
    return resultados

def processar_csv(job_id: str, input_path: Path):
    try:
        output_path = TMP_DIR / f"{job_id}_resultado.csv"
        is_first = True
        for chunk in pd.read_csv(input_path, chunksize=5000):
            mask_valid = (
                (chunk["CreditScore"].between(350, 850)) &
                (chunk["Geography"].isin(["France", "Germany", "Spain"])) &
                (chunk["Age"].between(18, 92)) &
                (chunk["Tenure"].between(0, 10)) &
                (chunk["Balance"].between(0, 500000))
            )
            chunk_valid = chunk[mask_valid].copy()
            if chunk_valid.empty: continue

            df_proc = preparar_dataframe(chunk_valid)
            X_scaled = artifacts["scaler"].transform(df_proc)
            probs = artifacts["model"].predict_proba(X_scaled)[:, 1]
            
            chunk_valid["probabilidade"] = probs.round(4)
            mask_alto = probs >= artifacts["threshold_cost"]
            chunk_valid["nivel_risco"] = np.where(mask_alto, "ALTO", "BAIXO")
            chunk_valid["previsao"] = np.where(mask_alto, "Vai cancelar", "Vai continuar")
            chunk_valid["explicabilidade"] = obter_explicabilidade_lote(X_scaled, chunk_valid, mask_alto)

            chunk_valid.to_csv(output_path, mode='a', index=False, header=is_first)
            is_first = False
        if input_path.exists(): input_path.unlink()
    except Exception as e:
        (TMP_DIR / f"{job_id}.error").write_text(str(e))

# =========================================================
# ENDPOINT /PREVISAO-LOTE
# =========================================================
@app.post("/previsao-lote")
def previsao_lote(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")
    job_id = str(uuid.uuid4())
    input_path = TMP_DIR / f"{job_id}.csv"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    background_tasks.add_task(processar_csv, job_id, input_path)
    return {"job_id": job_id, "status": "PROCESSANDO"}

# =========================================================
# ENDPOINT /STATUS
# =========================================================
@app.get("/previsao-lote/status/{job_id}")
def status_lote(job_id: str):
    if (TMP_DIR / f"{job_id}.error").exists(): return {"status": "ERRO"}
    if (TMP_DIR / f"{job_id}_resultado.csv").exists(): return {"status": "FINALIZADO"}
    return {"status": "PROCESSANDO"}


# =========================================================
# ENDPOINT /DOWNLOAD
# =========================================================
@app.get("/previsao-lote/download/{job_id}")
def download(job_id: str):
    path = TMP_DIR / f"{job_id}_resultado.csv"
    if not path.exists(): raise HTTPException(status_code=404)
    return FileResponse(path, filename="resultado.csv")