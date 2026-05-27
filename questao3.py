"""
Problema 3 - Precificação Minimax

Minimizar max_i |R_i(p1, p2) - R_alvo|
onde R_i(p1, p2) = a_{1,i}/p1 + a_{2,i}/p2.
"""

import numpy as np
from scipy.optimize import minimize


# Parâmetros dos cenários (a_{1,i}, a_{2,i})
CENARIOS = np.array([
    [100000,  80000],   # Mercado Normal
    [ 70000,  56000],   # Recessão Econômica
    [130000, 104000],   # Alta Demanda
    [ 90000,  90000],   # Pressão Competitiva
], dtype=float)

R_ALVO = 10000.0


def receita_cenario(p, a1, a2):
    """R_i(p1, p2) = a1/p1 + a2/p2"""
    # TODO: implementar
    pass


def objetivo_minimax(p):
    """max_i |R_i(p1, p2) - R_alvo|"""
    # TODO: implementar (calcular desvios em todos os cenários e retornar o máximo)
    pass


def main():
    # TODO: ponto inicial (preços positivos)
    p0 = None

    # TODO: escolher método adequado para função não-diferenciável (ex: Nelder-Mead)
    # resultado = minimize(objetivo_minimax, p0, method="Nelder-Mead")

    # TODO: imprimir p1*, p2*, valor da função objetivo, desvios em cada cenário
    pass


if __name__ == "__main__":
    main()
