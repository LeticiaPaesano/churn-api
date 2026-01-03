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
# PATHS / ARTIFACTS
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

    required_keys = {
        "model",
        "scaler",
        "columns",
        "threshold_cost",
        "balance_median",
        "salary_median",
        "model_version"
    }

    missing = required_keys - set(loaded.keys())
    if missing:
        raise RuntimeError(f"Artefato inválido. Chaves ausentes: {missing}")

    artifacts.update(loaded)
    yield

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(
    title="ChurnInsight API",
    version="3.1.0",
    lifespan=lifespan
)

# =========================================================
# UTILS
# =========================================================
def gerar_recomendacao(nivel_risco: str) -> str:
    mapa = {
        "ALTO": "Ação imediata recomendada: contato ativo e oferta personalizada",
        "MÉDIO": "Monitoramento recomendado e campanhas de retenção",
        "BAIXO": "Cliente estável - manutenção padrão"
    }
    return mapa.get(nivel_risco, "Manutenção padrão")


def calcular_explicabilidade_local(
    model,
    X: np.ndarray,
    feature_names: List[str],
    baseline_proba: float,
    input_data: dict
) -> List[str]:

    mapeamento = {
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
        impactos.append((feature, abs(baseline_proba - proba_mod)))

    impactos = sorted(impactos, key=lambda x: x[1], reverse=True)

    features_saida = []
    for feat, _ in impactos:
        contrato = mapeamento.get(feat)
        if not contrato:
            continue

        if contrato in ("Geography", "Gender"):
            valor = input_data.get(contrato)
            if valor and valor not in features_saida:
                features_saida.append(valor)
        else:
            if contrato not in features_saida:
                features_saida.append(contrato)

        if len(features_saida) == 3:
            break

    return features_saida


# =========================================================
# SCHEMAS
# =========================================================
class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=350, le=900)
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
# ENDPOINT PREVISAO
# =========================================================
@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):

    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    df["Geography_Germany"] = int(data.Geography == "Germany")
    df["Geography_Spain"] = int(data.Geography == "Spain")
    df["Gender_Male"] = int(data.Gender == "Male")

    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]

    df["High_Value_Customer"] = (
        (df["Balance"] > artifacts["balance_median"]) &
        (df["EstimatedSalary"] > artifacts["salary_median"])
    ).astype(int)

    for col in artifacts["columns"]:
        if col not in df:
            df[col] = 0

    df = df[artifacts["columns"]]
    X_scaled = artifacts["scaler"].transform(df)

    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    threshold = artifacts["threshold_cost"]

    if proba >= threshold:
        risco = "ALTO"
    elif proba >= 0.6 * threshold:
        risco = "MÉDIO"
    else:
        risco = "BAIXO"

    if proba >= threshold:
        previsao = "vai cancelar"
    else:
        previsao = "vai continuar"

    explicabilidade = None
    if risco in ("ALTO"):
        explicabilidade = calcular_explicabilidade_local(
            artifacts["model"],
            X_scaled,
            artifacts["columns"],
            proba,
            input_dict
        )

    return PredictionOutput(
        previsao=previsao,
        probabilidade = float(f"{proba:.2f}"),
        nivel_risco=risco,
        recomendacao=gerar_recomendacao(risco),
        explicabilidade=explicabilidade
    )

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "UP",
        "model_version": artifacts.get("model_version"),
        "threshold_cost": artifacts.get("threshold_cost"),
        "features_count": artifacts.get("features_count"),
        "trained_at": artifacts.get("trained_at"),
        "sklearn_version": artifacts.get("sklearn_version")
    }
