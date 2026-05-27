"""
Problema 2 - Restauração de Imagem (Total Variation)

Minimizar f(x) = ||A x - b||_2^2 + lambda * TV(x)
para uma imagem 32x32 contida em imagem.txt.
"""

import numpy as np
from scipy.optimize import minimize


# Parâmetros do problema
N, M_DIM = 32, 32         # dimensões da imagem
EPS = 1e-6                # epsilon do termo TV
LAMBDA = 0.1              # parâmetro de regularização (ajustar conforme necessário)


def carregar_imagem(caminho="imagem.txt"):
    """Lê a imagem borrada/ruidosa do arquivo (vetor b)."""
    # TODO: ler arquivo (np.loadtxt) e retornar como vetor achatado
    pass


def construir_matriz_blur(n_linhas, n_colunas):
    """Constrói a matriz A de convolução (blur)."""
    # TODO: implementar (ou usar exemplo fornecido em aula)
    pass


def termo_fidelidade(x, A, b):
    """||A x - b||_2^2"""
    # TODO: implementar
    pass


def termo_tv(x, n_linhas, n_colunas, eps=EPS):
    """Total Variation: soma sqrt((X_{i+1,j} - X_{i,j})^2 + (X_{i,j+1} - X_{i,j})^2 + eps)."""
    # TODO: implementar (reshape x para 2D e calcular diferenças)
    pass


def funcao_objetivo(x, A, b, lam, n_linhas, n_colunas):
    """f(x) = ||A x - b||^2 + lambda * TV(x)"""
    # TODO: implementar
    pass


def main():
    # TODO: carregar imagem b
    b = None

    # TODO: construir matriz A
    A = None

    # TODO: definir ponto inicial (ex: x0 = b)
    x0 = None

    # TODO: configurar e chamar otimizador
    # Dica: começar com maxiter baixo (5-10) e aumentar
    # resultado = minimize(funcao_objetivo, x0, args=(A, b, LAMBDA, N, M_DIM),
    #                      method=..., options={"maxiter": 10})

    # TODO: visualizar/salvar imagem recuperada e imprimir métricas
    pass


if __name__ == "__main__":
    main()
