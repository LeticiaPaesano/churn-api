from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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
    print(f"📌 Threshold: {artifacts['threshold_cost']} | Colunas: {len(artifacts['columns'])}")

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
        "version": "2.0.0",
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
# MAPA DE FEATURES DO MODELO -> CONTRATO DA API
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
    "Geography_Spain": "Geography",
    "Geography_Germany": "Geography",
    "Gender_Male": "Gender",
    "Gender_Female": "Gender",
}

# =========================================================
# EXPLICABILIDADE LOCAL (TOP 3)
# =========================================================
def calcular_explicabilidade_local(X_scaled: np.ndarray, payload: Dict) -> list[str]:
    model = artifacts["model"]
    features = artifacts["columns"]
    importances = model.feature_importances_
    impactos = importances * np.abs(X_scaled[0])
    
    impacto_por_contrato = {}
    for feature, impacto in zip(features, impactos):
        campo = FEATURE_MAP.get(feature)
        if not campo: continue
        impacto_por_contrato[campo] = impacto_por_contrato.get(campo, 0) + impacto
    
    ranking = sorted(impacto_por_contrato.items(), key=lambda x: x[1], reverse=True)
    
    explicabilidade = []
    seen = set()
    for campo, _ in ranking:
        if campo not in seen:
            seen.add(campo)
            if campo in ("Geography", "Gender"):
                explicabilidade.append(str(payload[campo]))
            else:
                explicabilidade.append(campo)
        if len(explicabilidade) == 3:
            break
            
    return explicabilidade

# =========================================================
# ENDPOINT /PREVISAO
# =========================================================
@app.post("/previsao")
def previsao(payload: Dict):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    colunas_obrigatorias = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "EstimatedSalary"]
    faltantes = set(colunas_obrigatorias) - set(payload.keys())
    if faltantes:
        raise HTTPException(status_code=400, detail=f"Colunas ausentes: {list(faltantes)}")
    
    df = pd.DataFrame([payload])
    df_proc = preparar_dataframe(df)
    X_scaled = artifacts["scaler"].transform(df_proc)
    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    
    risco = "ALTO" if proba >= artifacts["threshold_cost"] else "BAIXO"
    previsao_txt = "Vai cancelar" if risco == "ALTO" else "Vai continuar"
    
    explicabilidade = []
    if previsao_txt == "Vai cancelar":
        explicabilidade = calcular_explicabilidade_local(X_scaled, payload)
    
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

        indices_decrescentes = np.argsort(impactos_matriz[i])[::-1]
        final_row_names = []
        seen_features = set()
        
        for idx in indices_decrescentes:
            nome_amigavel = nomes_campos[idx]
            if nome_amigavel not in seen_features:
                seen_features.add(nome_amigavel)
                if nome_amigavel in ["Geography", "Gender"]:
                    val = str(chunk_df.iloc[i][nome_amigavel])
                    final_row_names.append(val)
                else:
                    final_row_names.append(nome_amigavel)
            
            if len(final_row_names) == 3:
                break
        
        resultados.append(", ".join(final_row_names))
    return resultados

def processar_csv(job_id: str, input_path: Path):
    try:
        output_path = TMP_DIR / f"{job_id}_resultado.csv"
        chunk_size = 5000 
        is_first = True

        if not input_path.exists():
            raise FileNotFoundError("Arquivo de entrada não encontrado")

        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
            df_proc = preparar_dataframe(chunk)
            X_scaled = artifacts["scaler"].transform(df_proc)
            
            probs = artifacts["model"].predict_proba(X_scaled)[:, 1]
            threshold = artifacts["threshold_cost"]

            chunk["probabilidade"] = probs.round(4)
            mask_alto = probs >= threshold
            chunk["nivel_risco"] = np.where(mask_alto, "ALTO", "BAIXO")
            chunk["previsao"] = np.where(mask_alto, "Vai cancelar", "Vai continuar")
            
            chunk["explicabilidade"] = obter_explicabilidade_lote(X_scaled, chunk, mask_alto)

            chunk.to_csv(output_path, mode='a', index=False, header=is_first)
            is_first = False

        if input_path.exists():
            input_path.unlink()

    except Exception as e:
        error_file = TMP_DIR / f"{job_id}.error"
        error_file.write_text(str(e))

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
# ENDPOINT /PREVISAO-LOTE/STATUS
# =========================================================
@app.get("/previsao-lote/status/{job_id}")
def status_lote(job_id: str):
    if (TMP_DIR / f"{job_id}.error").exists():
        erro = (TMP_DIR / f"{job_id}.error").read_text()
        return {"status": "ERRO", "detalhe": erro}
    
    if (TMP_DIR / f"{job_id}_resultado.csv").exists():
        return {"status": "FINALIZADO"}
    
    return {"status": "PROCESSANDO"}

# =========================================================
# ENDPOINT /PREVISAO-LOTE/DOWNLOAD
# =========================================================
@app.get("/previsao-lote/download/{job_id}")
def download(job_id: str):
    path = TMP_DIR / f"{job_id}_resultado.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não disponível")
    return FileResponse(path, filename=path.name, media_type="text/csv")