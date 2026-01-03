from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tempfile
import os

# =========================
# Configuração global
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"

artifacts = {}

# =========================
# Lifespan - carga do modelo
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)
    artifacts.update(loaded)

    print("✅ Modelo carregado com sucesso")
    yield

app = FastAPI(
    title="Churn Prediction API",
    version="1.0",
    lifespan=lifespan
)

# =========================
# Schemas
# =========================

class CustomerInput(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    EstimatedSalary: float

class PredictionOutput(BaseModel):
    previsao: str
    probabilidade: float
    nivel_risco: str
    recomendacao: str
    explicabilidade: list | None

# =========================
# Utilidades
# =========================

def gerar_recomendacao(risco: str) -> str:
    if risco == "ALTO":
        return "Ação imediata recomendada: contato ativo e oferta personalizada"
    if risco == "MÉDIO":
        return "Monitorar cliente e oferecer incentivos"
    return "Cliente estável, sem ação necessária"

def calcular_explicabilidade_local(model, X, columns, proba):
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "feature": columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df_imp.head(3)["feature"].tolist()

def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)
    df["Geography_Spain"] = (df["Geography"] == "Spain").astype(int)
    df["Gender_Male"] = (df["Gender"] == "Male").astype(int)

    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]

    df["High_Value_Customer"] = (
        (df["Balance"] > artifacts["balance_median"]) &
        (df["EstimatedSalary"] > artifacts["salary_median"])
    ).astype(int)

    for col in artifacts["columns"]:
        if col not in df:
            df[col] = 0

    return df[artifacts["columns"]]

# =========================
# Health check
# =========================

@app.get("/health")
def health():
    return {
        "status": "UP",
        "model_loaded": bool(artifacts)
    }

# =========================
# Previsão unitária
# =========================

@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):

    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    df = pd.DataFrame([data.model_dump()])
    df = preparar_dataframe(df)

    X_scaled = artifacts["scaler"].transform(df)
    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])

    threshold = artifacts["threshold_cost"]

    risco = "ALTO" if proba >= threshold else "BAIXO"
    previsao = "Vai Sair" if risco == "ALTO" else "Vai Ficar"

    explicabilidade = None
    if risco == "ALTO":
        explicabilidade = calcular_explicabilidade_local(
            artifacts["model"],
            X_scaled,
            artifacts["columns"],
            proba
        )

    return PredictionOutput(
        previsao=previsao,
        probabilidade=float(f"{proba:.2f}"),
        nivel_risco=risco,
        recomendacao=gerar_recomendacao(risco),
        explicabilidade=explicabilidade
    )

# =========================
# Previsão em lote (CSV)
# =========================

@app.post("/previsao-lote")
def previsao_lote(file: UploadFile = File(...)):

    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")

    df = pd.read_csv(file.file)

    colunas_necessarias = artifacts["raw_columns"]
    faltantes = set(colunas_necessarias) - set(df.columns)
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"Colunas ausentes: {list(faltantes)}"
        )

    df_proc = preparar_dataframe(df.copy())
    X_scaled = artifacts["scaler"].transform(df_proc)
    probs = artifacts["model"].predict_proba(X_scaled)[:, 1]

    threshold = artifacts["threshold_cost"]

    df["probabilidade"] = [float(f"{p:.2f}") for p in probs]
    df["nivel_risco"] = np.where(probs >= threshold, "ALTO", "BAIXO")
    df["previsao"] = np.where(df["nivel_risco"] == "ALTO", "Vai Sair", "Vai Ficar")

    explicabilidades = []
    for i, p in enumerate(probs):
        if p >= threshold:
            explicabilidades.append(
                "|".join(
                    calcular_explicabilidade_local(
                        artifacts["model"],
                        X_scaled[i:i+1],
                        artifacts["columns"],
                        p
                    )
                )
            )
        else:
            explicabilidades.append(None)

    df["explicabilidade"] = explicabilidades

    output_path = Path(tempfile.gettempdir()) / f"{Path(file.filename).stem}_previsionado.csv"
    df.to_csv(output_path, index=False)

    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type="text/csv"
    )
