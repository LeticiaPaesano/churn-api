import os
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional, List

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =========================================================
# LOGGING
# =========================================================

logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")

artifacts: dict = {}

# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)

    artifacts["model"] = loaded["model"]
    artifacts["scaler"] = loaded["scaler"]
    artifacts["columns"] = loaded["columns"]
    artifacts["threshold"] = loaded.get("threshold", 0.35)
    artifacts["balance_median"] = loaded.get("balance_median", 0.0)
    artifacts["salary_median"] = loaded.get("salary_median", 0.0)

    print("✅ Modelo carregado com sucesso")
    yield

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="ChurnInsight API",
    version="1.1.0",
    lifespan=lifespan,
)

# =========================================================
# UTILS
# =========================================================

def classificar_faixa_score(score: int) -> str:
    if score >= 701: return "Excelente"
    if score >= 501: return "Bom"
    if score >= 301: return "Regular"
    return "Baixo"

def gerar_recomendacao(nivel_risco: str) -> str:
    if nivel_risco == "ALTO":
        return "Ação imediata recomendada: contato ativo e oferta personalizada"
    if nivel_risco == "MÉDIO":
        return "Monitoramento recomendado e campanhas de retenção"
    return "Cliente estável - manutenção padrão"

def calcular_explicabilidade(model, X: np.ndarray, feature_names: List[str]) -> List[str]:
    if hasattr(model, "coef_"):
        impactos = np.abs(model.coef_[0] * X[0])
    elif hasattr(model, "feature_importances_"):
        impactos = model.feature_importances_
    else:
        return []

    top_idx = np.argsort(impactos)[-3:][::-1]
    return [feature_names[i] for i in top_idx]

# =========================================================
# SCHEMAS
# =========================================================

class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=0, le=1000)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=92)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    EstimatedSalary: float = Field(..., ge=0)

class PredictionOutput(BaseModel):
    previsao: str
    probabilidade: float
    nivel_risco: str
    recomendacao: str
    explicabilidade: Optional[List[str]] = None

# =========================================================
# ENDPOINT
# =========================================================

@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):

    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]
    df["High_Value_Customer"] = (
        (df["Balance"] > artifacts["balance_median"]) &
        (df["EstimatedSalary"] > artifacts["salary_median"])
    ).astype(int)

    for col in ["Geography_Germany", "Geography_Spain", "Gender_Male"]:
        df[col] = 0

    if data.Geography == "Germany":
        df["Geography_Germany"] = 1
    elif data.Geography == "Spain":
        df["Geography_Spain"] = 1

    if data.Gender == "Male":
        df["Gender_Male"] = 1

    df_final = df[artifacts["columns"]]
    X = artifacts["scaler"].transform(df_final)

    proba = float(artifacts["model"].predict_proba(X)[0, 1])
    threshold = artifacts["threshold"]

    previsao = "Vai cancelar" if proba >= threshold else "Vai continuar"

    if proba >= threshold:
        nivel_risco = "ALTO"
    elif proba >= threshold * 0.7:
        nivel_risco = "MÉDIO"
    else:
        nivel_risco = "BAIXO"

    explicabilidade = None
    if previsao == "Vai cancelar":
        explicabilidade = calcular_explicabilidade(
            artifacts["model"], X, artifacts["columns"]
        )

    return PredictionOutput(
        previsao=previsao,
        probabilidade=round(proba, 4),
        nivel_risco=nivel_risco,
        recomendacao=gerar_recomendacao(nivel_risco),
        explicabilidade=explicabilidade,
    )


# =========================================================
# ENDPOINT DE DEBUG 
# =========================================================

@app.get("/debug/model")
def debug_model():
    return {
        "status_carregamento": "OK" if "model" in artifacts else "ERRO",
        "model_type": str(type(artifacts.get("model"))),
        "scaler_presente": "scaler" in artifacts,
        "colunas_esperadas": artifacts.get("columns", []),
        "threshold_atual": artifacts.get("threshold"),
        "medianas": {
            "balance": artifacts.get("balance_median"),
            "salary": artifacts.get("salary_median"),
        },
    }
    
# =========================================================
# ENDPOINT DE HEALTH CHECK 
# =========================================================   
    
@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "model_loaded": "model" in artifacts,
        "scaler_loaded": "scaler" in artifacts,
        "columns_loaded": bool(artifacts.get("columns")),
    }
