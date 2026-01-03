from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4
import pandas as pd
import numpy as np
import joblib
import tempfile
import io

app = FastAPI(title="Churn Prediction API")

artifacts = {}
jobs = {}

# =========================
# MODELS
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
# STARTUP
# =========================

@app.on_event("startup")
def load_artifacts():
    global artifacts

    path = Path("model/model.joblib")
    if not path.exists():
        raise RuntimeError("Artefato do modelo não encontrado")

    artifacts = joblib.load(path)

    required_keys = {
        "model",
        "scaler",
        "columns",
        "threshold_cost",
        "balance_median",
        "salary_median"
    }

    if not required_keys.issubset(artifacts.keys()):
        raise RuntimeError("Artefatos incompletos")


# =========================
# UTILS
# =========================

def gerar_recomendacao(risco: str) -> str:
    if risco == "ALTO":
        return "Ação imediata recomendada: contato ativo e oferta personalizada"
    if risco == "MÉDIO":
        return "Monitorar cliente e avaliar oferta preventiva"
    return "Nenhuma ação necessária no momento"


def calcular_explicabilidade_local(model, X, columns, proba, input_dict):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[1]
    impact = np.abs(shap_values[0])
    top_idx = np.argsort(impact)[-3:]
    return [columns[i] for i in top_idx]


# =========================
# ENDPOINT PREVISAO
# =========================

@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):

    df = pd.DataFrame([data.model_dump()])

    df["Geography_Germany"] = int(data.Geography == "Germany")
    df["Geography_Spain"] = int(data.Geography == "Spain")
    df["Gender_Male"] = int(data.Gender == "Male")

    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]

    df["High_Value_Customer"] = int(
        df["Balance"].iloc[0] > artifacts["balance_median"] and
        df["EstimatedSalary"].iloc[0] > artifacts["salary_median"]
    )

    for col in artifacts["columns"]:
        if col not in df:
            df[col] = 0

    X = df[artifacts["columns"]]
    X_scaled = artifacts["scaler"].transform(X)

    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    threshold = artifacts["threshold_cost"]

    if proba >= threshold:
        risco = "ALTO"
        previsao = "Vai Sair"
    elif proba >= 0.20:
        risco = "MÉDIO"
        previsao = "Vai Sair"
    else:
        risco = "BAIXO"
        previsao = "Vai Ficar"

    explicabilidade = None
    if risco == "ALTO":
        explicabilidade = calcular_explicabilidade_local(
            artifacts["model"],
            X_scaled,
            artifacts["columns"],
            proba,
            data.model_dump()
        )

    return PredictionOutput(
        previsao=previsao,
        probabilidade=float(f"{proba:.2f}"),
        nivel_risco=risco,
        recomendacao=gerar_recomendacao(risco),
        explicabilidade=explicabilidade
    )


# =========================
# PROCESSAMENTO EM BACKGROUND
# =========================

def processar_csv_lote(contents: bytes, filename: str, job_id: str):

    df = pd.read_csv(io.BytesIO(contents))

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

    X = df[artifacts["columns"]]
    X_scaled = artifacts["scaler"].transform(X)

    probas = artifacts["model"].predict_proba(X_scaled)[:, 1]
    threshold = artifacts["threshold_cost"]

    df["probabilidade"] = np.round(probas, 2)
    df["nivel_risco"] = np.where(
        probas >= threshold, "ALTO",
        np.where(probas >= 0.20, "MÉDIO", "BAIXO")
    )
    df["previsao"] = np.where(probas >= threshold, "Vai Sair", "Vai Ficar")

    explicabilidades = []
    for i, risco in enumerate(df["nivel_risco"]):
        if risco == "ALTO":
            explicabilidades.append(
                "|".join(
                    calcular_explicabilidade_local(
                        artifacts["model"],
                        X_scaled[i:i+1],
                        artifacts["columns"],
                        probas[i],
                        df.iloc[i].to_dict()
                    )
                )
            )
        else:
            explicabilidades.append(None)

    df["explicabilidade"] = explicabilidades

    output = Path(tempfile.gettempdir()) / filename.replace(".csv", "_previsionado.csv")
    df.to_csv(output, index=False)

    jobs[job_id] = {
        "status": "FINALIZADO",
        "progress": 100,
        "output": str(output)
    }


# =========================
# ENDPOINT PREVISAO LOTE 
# =========================

@app.post("/previsao-lote")
async def previsao_lote(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):

    job_id = str(uuid4())

    jobs[job_id] = {
        "status": "PROCESSANDO",
        "progress": 0,
        "output": None
    }

    contents = await file.read()

    background_tasks.add_task(
        processar_csv_lote,
        contents,
        file.filename,
        job_id
    )

    return {
        "job_id": job_id,
        "status": "PROCESSANDO"
    }


# =========================
# ENDPOINT STATUS
# =========================

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado")
    return jobs[job_id]


# =========================
# ENDPOINT DOWNLOAD
# =========================

@app.get("/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)

    if not job or job["status"] != "FINALIZADO":
        raise HTTPException(404, "Arquivo não disponível")

    return FileResponse(
        job["output"],
        media_type="text/csv",
        filename=Path(job["output"]).name
    )

# =========================
# ENDPOINT HEALTH CHECK
# =========================
@app.get("/health")
def health_check():
    try:
        if not artifacts:
            return {
                "status": "DOWN",
                "model_loaded": False
            }

        dummy = pd.DataFrame(
            [[0] * len(artifacts["columns"])],
            columns=artifacts["columns"]
        )

        artifacts["scaler"].transform(dummy)
        artifacts["model"].predict_proba(dummy)

        return {
            "status": "UP",
            "model_loaded": True
        }

    except Exception as e:
        return {
            "status": "DOWN",
            "model_loaded": False,
            "error": str(e)
        }
