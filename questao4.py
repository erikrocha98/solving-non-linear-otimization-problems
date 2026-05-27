"""
Problema 4 - Energia Potencial do Sistema de Carrinhos

Minimizar U(x1, x2) = (1/2) k2 x1^2 + (1/2) k3 (x2 - x1)^2 + (1/2) k1 x2^2 - P x2
"""

import numpy as np
from scipy.optimize import minimize


# Parâmetros físicos
K1 = 1000.0   # N/m
K2 = 1500.0   # N/m
K3 = 2000.0   # N/m
P  = 100.0    # N


def energia_potencial(x):
    """U(x1, x2)"""
    # TODO: implementar
    pass


def gradiente(x):
    """Gradiente analítico de U (opcional)."""
    # TODO: implementar (opcional)
    pass


def hessiana(x):
    """Hessiana de U (opcional - é constante)."""
    # TODO: implementar (opcional)
    pass


def main():
    # TODO: ponto inicial
    x0 = None

    # TODO: chamar otimizador (problema quadrático convexo -> métodos como Newton/BFGS)
    # resultado = minimize(energia_potencial, x0, method=..., jac=gradiente)

    # TODO: imprimir x1*, x2*, U(x*), número de iterações/avaliações
    pass


if __name__ == "__main__":
    main()
