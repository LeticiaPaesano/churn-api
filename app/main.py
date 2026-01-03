import os
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional, List, Dict, Any

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

    artifacts.update(loaded)
    print("✅ Pipeline carregado com sucesso")
    yield

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="ChurnInsight API", 
    version="1.2.1", 
    lifespan=lifespan
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
    recomendas = {
        "ALTO": "Ação imediata recomendada: contato ativo e oferta personalizada",
        "MÉDIO": "Monitoramento recomendado e campanhas de retenção",
        "BAIXO": "Cliente estável - manutenção padrão"
    }
    return recomendas.get(nivel_risco, "Manutenção padrão")

def calcular_explicabilidade_local(
    model,
    X: np.ndarray,
    feature_names: List[str],
    baseline_proba: float, 
    input_data: dict 
) -> List[str]:
    """
    Retorna lista com as 3 principais features do contrato original
    """

    mapeamento_para_contrato = {
        "CreditScore": "CreditScore",
        "Age": "Age",
        "Tenure": "Tenure",
        "Balance": "Balance",
        "EstimatedSalary": "EstimatedSalary",
        "Geography_Germany": "Geography",
        "Geography_Spain": "Geography",
        "Gender_Male": "Gender",
        "Balance_Salary_Ratio": "Balance",
        "Age_Tenure": "Age",
        "High_Value_Customer": "Balance"
    }

    impactos = []
    for i, feature in enumerate(feature_names):
        X_mod = X.copy()
        X_mod[0, i] = 0 
        proba_mod = model.predict_proba(X_mod)[0, 1]
        impacto = abs(baseline_proba - proba_mod)
        impactos.append((feature, impacto))

    impactos_ordenados = sorted(impactos, key=lambda x: x[1], reverse=True)

    features_contrato = []
    for feat_interna, _ in impactos_ordenados:
        feature_contrato = mapeamento_para_contrato.get(feat_interna)
        
        if feature_contrato:
            if feature_contrato == "Geography":
                valor = input_data.get("Geography")
                if valor and valor not in features_contrato:
                    features_contrato.append(valor)
            elif feature_contrato == "Gender":
                valor = input_data.get("Gender")
                if valor and valor not in features_contrato:
                    features_contrato.append(valor)
            elif feature_contrato not in features_contrato:
                features_contrato.append(feature_contrato)
        
        if len(features_contrato) >= 3:
            break
    
    return features_contrato

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
    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
    df["Geography_Germany"] = 1 if data.Geography == "Germany" else 0
    df["Geography_Spain"] = 1 if data.Geography == "Spain" else 0
    df["Gender_Male"] = 1 if data.Gender == "Male" else 0
    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]

    balance_median = artifacts.get("balance_median", 0)
    salary_median = artifacts.get("salary_median", 0)
    
    df["High_Value_Customer"] = (
        (df["Balance"] > balance_median) &
        (df["EstimatedSalary"] > salary_median)
    ).astype(int)

    required_columns = artifacts.get("columns", [])

    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    df_final = df[required_columns]

    X_scaled = artifacts["scaler"].transform(df_final)

    proba_raw = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    proba_percentual = proba_raw * 100.0
    proba = round(proba_percentual, 1)
    
    threshold = artifacts.get("threshold", 0.5) * 100.0

    if proba >= 40.0:
        risco = "ALTO"
    elif proba >= 20.0:
        risco = "MÉDIO"
    else:
        risco = "BAIXO"

    previsao = "Vai cancelar" if proba >= threshold else "Vai continuar"

    explicabilidade_output = None
    if previsao == "Vai cancelar" or risco == "ALTO":
        explicabilidade_output = calcular_explicabilidade_local(
            artifacts["model"], X_scaled, required_columns, proba_raw, input_dict
        )

    return PredictionOutput(
        previsao=previsao,
        probabilidade=proba,
        nivel_risco=risco,
        recomendacao=gerar_recomendacao(risco),
        explicabilidade=explicabilidade_output
    )

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "model_loaded": "model" in artifacts,
        "scaler_loaded": "scaler" in artifacts,
        "columns_loaded": "columns" in artifacts,
        "threshold_loaded": "threshold" in artifacts,
        "threshold_percentual": artifacts.get("threshold", 0.5) * 100.0 if "threshold" in artifacts else None
    }

# =========================================================
# TEST CASES
# =========================================================
@app.post("/test-case-1")
def test_case_1():
    data = CustomerInput(
        CreditScore=500,
        Geography="Germany",
        Gender="Female",
        Age=45,
        Tenure=2,
        Balance=125000.0,
        EstimatedSalary=180000.0
    )
    return predict_churn(data)

@app.post("/test-case-2")
def test_case_2():
    data = CustomerInput(
        CreditScore=350,
        Geography="Germany",
        Gender="Female",
        Age=55,
        Tenure=8,
        Balance=0.0,
        EstimatedSalary=15000.0
    )
    return predict_churn(data)

@app.post("/test-case-baixo")
def test_case_baixo():
    data = CustomerInput(
        CreditScore=850,
        Geography="France",
        Gender="Male",
        Age=30,
        Tenure=10,
        Balance=10000.0,
        EstimatedSalary=50000.0
    )
    return predict_churn(data)

@app.post("/test-case-medio")
def test_case_medio():
    data = CustomerInput(
        CreditScore=600,
        Geography="Spain",
        Gender="Female",
        Age=40,
        Tenure=3,
        Balance=50000.0,
        EstimatedSalary=60000.0
    )
    return predict_churn(data)

@app.get("/")
def root():
    return {
        "message": "ChurnInsight API",
        "version": "1.2.1",
        "escala_probabilidade": "0.0 a 99.9 (percentual)",
        "faixas_risco": {
            "ALTO": "≥ 40.0%",
            "MÉDIO": "20.0% a 39.9%",
            "BAIXO": "0.0% a 19.9%"
        },
        "endpoints": {
            "POST /previsao": "Fazer previsão de churn",
            "GET /health": "Verificar saúde da API",
            "POST /test-case-1": "Cliente alto risco (exemplo 1)",
            "POST /test-case-2": "Cliente alto risco (exemplo 2)",
            "POST /test-case-medio": "Cliente risco médio",
            "POST /test-case-baixo": "Cliente baixo risco"
        }
    }