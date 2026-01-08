import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def payload_valido():
    return {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 12000.50,
        "EstimatedSalary": 50000.0
    }