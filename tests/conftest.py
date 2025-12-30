import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app, artifacts


class DummyModel:
    coef_ = np.array([[0.4, 0.3, 0.2, 0.1, 0.05, 0.01]])

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


class DummyScaler:
    def transform(self, X):
        return X.values


@pytest.fixture(scope="session")
def client():
    artifacts["model"] = DummyModel()
    artifacts["scaler"] = DummyScaler()
    artifacts["columns"] = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "EstimatedSalary",
        "Balance_Salary_Ratio",
    ]
    artifacts["threshold"] = 0.5
    artifacts["balance_median"] = 1000
    artifacts["salary_median"] = 1000

    return TestClient(app)
