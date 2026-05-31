"""Testes unitários para questao3.py (Precificação Minimax)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import questao3
from questao3 import (
    CENARIOS,
    R_ALVO,
    receita_cenario,
    receitas,
    desvios,
    objetivo_minimax,
    run_methods,
)


# --------------------------------------------------------------------------- #
# Funções base
# --------------------------------------------------------------------------- #
def test_receita_cenario():
    """R = a1/p1 + a2/p2."""
    val = receita_cenario((10.0, 20.0), 100000.0, 80000.0)
    assert np.isclose(val, 100000 / 10 + 80000 / 20)


def test_receitas_vetor():
    """receitas(p) coincide com a definição por cenário e tem shape (4,)."""
    p = (15.0, 25.0)
    r = receitas(p)
    esperado = CENARIOS[:, 0] / p[0] + CENARIOS[:, 1] / p[1]
    assert r.shape == (4,)
    assert np.allclose(r, esperado)


def test_desvios_e_objetivo_consistentes():
    """f_obj(p) == max_i |R_i - R_alvo|."""
    p = (18.0, 22.0)
    d = desvios(p)
    assert np.allclose(d, np.abs(receitas(p) - R_ALVO))
    assert np.isclose(objetivo_minimax(p), d.max())


def test_guarda_positividade():
    """Fora do ortante positivo o objetivo retorna +inf."""
    assert objetivo_minimax((-1.0, 10.0)) == np.inf
    assert objetivo_minimax((10.0, 0.0)) == np.inf
    assert np.isfinite(objetivo_minimax((10.0, 10.0)))


# --------------------------------------------------------------------------- #
# Estrutura analítica dos cenários (§1.2 do plano)
# --------------------------------------------------------------------------- #
def test_ordenacao_R2_R1_R3():
    """Proporção a1:a2 = 1.25 em 1,2,3 => R2 < R1 < R3 para todo p > 0."""
    for p in [(10.0, 10.0), (12.0, 30.0), (40.0, 8.0)]:
        r = receitas(p)
        assert r[1] < r[0] < r[2]


# --------------------------------------------------------------------------- #
# Otimização: concordância dos métodos + propriedade minimax
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def resultados():
    """Resultados dos 3 métodos a partir de um ponto inicial padrão."""
    res, _ = run_methods((20.0, 20.0))
    return res


def test_metodos_concordam_no_valor_otimo(resultados):
    """
    Os três métodos atingem o mesmo valor ótimo f* = 3000.

    OBS.: o minimizador NÃO é único. Ao longo da reta 10/p1 + 8/p2 = 1 valem
    R_2 = 7000 e R_3 = 13000 (desvio 3000) — um segmento contínuo de soluções.
    Por isso comparamos f* (invariante), não p* (que difere entre métodos).
    """
    f_vals = [r["fun"] for r in resultados.values()]
    f_ref = min(f_vals)
    for f in f_vals:
        assert abs(f - f_ref) <= 1e-2 * max(f_ref, 1.0) + 1e-3
    assert f_ref == pytest.approx(3000.0, rel=1e-4)


def test_estrutura_do_otimo(resultados):
    """
    Em cada p* os cenários 2 e 3 ficam ativos com R_2 ≈ 7000 e R_3 ≈ 13000,
    e o ponto satisfaz a relação analítica 10/p1 + 8/p2 = 1.
    """
    for r in resultados.values():
        p = r["p"]
        rev = receitas(p)
        assert rev[1] == pytest.approx(7000.0, abs=1.0)    # cenário 2
        assert rev[2] == pytest.approx(13000.0, abs=1.0)   # cenário 3
        assert (10.0 / p[0] + 8.0 / p[1]) == pytest.approx(1.0, abs=1e-3)


def test_minimax_dois_cenarios_ativos(resultados):
    """No ótimo minimax, >= 2 cenários empatam no desvio máximo."""
    p_star = resultados["SLSQP (epígrafo)"]["p"]
    d = desvios(p_star)
    ativos = np.sum(d >= d.max() - 1e-2 * max(d.max(), 1.0))
    assert ativos >= 2


def test_cenario1_nunca_e_pior_no_otimo(resultados):
    """Previsão §1.2: cenário 1 não é o pior caso no ótimo (ativos em {2,3,4})."""
    p_star = resultados["SLSQP (epígrafo)"]["p"]
    d = desvios(p_star)
    assert np.argmax(d) != 0  # índice 0 == cenário 1


def test_slsqp_factivel(resultados):
    """O p* do SLSQP é estritamente positivo (dentro do domínio prático)."""
    p_star = resultados["SLSQP (epígrafo)"]["p"]
    assert np.all(p_star > 0)


def test_otimo_independe_do_ponto_inicial():
    """f* é robusto à variação do ponto inicial (mesma bacia)."""
    f_finais = []
    for p0 in [(10.0, 10.0), (40.0, 40.0), (15.0, 35.0)]:
        res, _ = run_methods(p0)
        f_finais.append(res["SLSQP (epígrafo)"]["fun"])
    f_finais = np.array(f_finais)
    assert f_finais.std() <= 1e-3 * max(f_finais.mean(), 1.0) + 1e-4


# --------------------------------------------------------------------------- #
def test_smoke_main_runs(monkeypatch):
    """main() roda sem erro em modo headless (plt desativado)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(questao3, "plot_contorno", lambda *a, **k: None)
    monkeypatch.setattr(questao3, "plot_convergencia", lambda *a, **k: None)
    questao3.main()


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.sys.exit(_pytest.main([__file__, "-v"]))
