import numpy as np
from app.main import calcular_explicabilidade_local

def test_explicabilidade_retorna_campos_contrato(payload_valido):
    X_fake = np.ones((1, 10))  # tamanho não importa aqui

    resultado = calcular_explicabilidade_local(X_fake, payload_valido)

    for item in resultado:
        assert item in (
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "EstimatedSalary",
            payload_valido["Geography"],
            payload_valido["Gender"]
        )
