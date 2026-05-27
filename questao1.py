"""
Problema 1 - Frequência Natural Fundamental (Quociente de Rayleigh)

Minimizar R(x) = (x^T K x) / (x^T M x)
para encontrar omega_1^2 do sistema massa-mola com 3 graus de liberdade.
"""

import numpy as np
from scipy.optimize import minimize


# Matrizes do sistema
M = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=float)

K = np.array([
    [ 2, -1,  0],
    [-1,  2, -1],
    [ 0, -1,  1],
], dtype=float)


def rayleigh_quotient(x):
    """Quociente de Rayleigh R(x) = (x^T K x) / (x^T M x)."""
    # TODO: implementar
    pass


def gradient(x):
    """Gradiente analítico (opcional)."""
    # TODO: implementar (opcional)
    pass


def main():
    # TODO: definir ponto inicial
    x0 = None

    # TODO: chamar o método de otimização escolhido (scipy.optimize.minimize)
    # resultado = minimize(rayleigh_quotient, x0, method=...)

    # TODO: imprimir resultados
    #   - x*
    #   - R(x*) = omega_1^2
    #   - omega_1
    #   - número de iterações
    #   - número de avaliações da função
    pass


if __name__ == "__main__":
    main()
