import numpy as np
from app.main import calcular_explicabilidade


class FakeModel:
    coef_ = np.array([[0.8, 0.1, 0.05]])


def test_explicabilidade_retorna_top_3():
    model = FakeModel()
    X = np.array([[10, 1, 1]])
    features = ["Age", "Tenure", "Balance"]

    resultado = calcular_explicabilidade(model, X, features)

    assert isinstance(resultado, list)
    assert len(resultado) == 3
    assert resultado[0] == "Age"
