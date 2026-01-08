import numpy as np
from app.main import calcular_explicabilidade_local

def test_explicabilidade_sem_repeticao(payload_valido):

    from app.main import artifacts
    X_fake = np.random.rand(1, len(artifacts["columns"]))

    resultado = calcular_explicabilidade_local(X_fake, payload_valido)

    assert len(resultado) == 3
    assert len(set(resultado)) == 3
    
    for item in resultado:
        assert isinstance(item, str)