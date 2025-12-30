from app.main import classificar_faixa_score, gerar_recomendacao


def test_classificacao_score_excelente():
    assert classificar_faixa_score(750) == "Excelente"


def test_classificacao_score_baixo():
    assert classificar_faixa_score(200) == "Baixo"


def test_recomendacao_alto_risco():
    msg = gerar_recomendacao("ALTO")
    assert "Ação imediata" in msg
