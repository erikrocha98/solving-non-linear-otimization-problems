"""Testes unitários — Problema 1 (Quociente de Rayleigh)."""

import sys
import os

import numpy as np
import pytest
from scipy.optimize import approx_fprime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from questao1 import K, M, rayleigh_quotient, gradient, run_methods


# ---------------------------------------------------------------------------
# rayleigh_quotient
# ---------------------------------------------------------------------------

def test_rayleigh_unit_vectors():
    """R(e_i) reproduz a diagonal de K: R(e1)=2, R(e2)=2, R(e3)=1."""
    e1, e2, e3 = np.eye(3)
    assert rayleigh_quotient(e1) == pytest.approx(K[0, 0], rel=1e-12)  # 2.0
    assert rayleigh_quotient(e2) == pytest.approx(K[1, 1], rel=1e-12)  # 2.0
    assert rayleigh_quotient(e3) == pytest.approx(K[2, 2], rel=1e-12)  # 1.0


def test_rayleigh_scale_invariance():
    """R(α·x) == R(x) para qualquer α ≠ 0 (invariância de escala)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(3)
    R_x = rayleigh_quotient(x)
    for alpha in [0.5, 2.0, -1.0, 100.0, -0.001, 1e-8]:
        assert rayleigh_quotient(alpha * x) == pytest.approx(R_x, rel=1e-10)


def test_rayleigh_between_eigenvalues():
    """R(x) deve estar sempre no intervalo [λ_min, λ_max] (Courant-Fischer)."""
    lambdas = np.linalg.eigh(K)[0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(3)
        R = rayleigh_quotient(x)
        assert R >= lambdas[0] - 1e-12
        assert R <= lambdas[-1] + 1e-12


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------

def test_gradient_vs_finite_differences():
    """Gradiente analítico coincide com diferenças finitas em pontos aleatórios."""
    rng = np.random.default_rng(7)
    for _ in range(8):
        x = rng.standard_normal(3)
        grad_analytic = gradient(x)
        grad_fd = approx_fprime(x, rayleigh_quotient, 1e-7)
        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-5, atol=1e-8)


def test_gradient_zero_at_eigenvectors():
    """Nos autovetores de K o gradiente deve ser numericamente nulo."""
    _, eigenvectors = np.linalg.eigh(K)
    for i in range(3):
        v = eigenvectors[:, i]
        assert np.linalg.norm(gradient(v)) == pytest.approx(0.0, abs=1e-10)


def test_gradient_homogeneity():
    """∇R(α·x) = (1/α)·∇R(x) — consequência da invariância de escala."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(3)
    alpha = 3.0
    # R(αx)=R(x)  ⟹  ∇R(αx)·d = ∇R(x)·(d/α)  ⟹  ∇R(αx) = (1/α)∇R(x)
    np.testing.assert_allclose(gradient(alpha * x), gradient(x) / alpha, rtol=1e-10)


# ---------------------------------------------------------------------------
# Otimização — resultado final
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name", ["BFGS", "Nelder-Mead", "Newton-CG"])
def test_optimal_value(method_name):
    """R(x*) deve aproximar o menor autovalor de K com tolerância 1e-6."""
    x0 = np.array([1.0, 1.0, 1.0])
    eigenvalues, _ = np.linalg.eigh(K)
    omega1_sq_ref = eigenvalues[0]

    results, _ = run_methods(x0)
    res = results[method_name]
    assert res.fun == pytest.approx(omega1_sq_ref, rel=1e-6), (
        f"{method_name}: R(x*)={res.fun:.12f}, esperado ≈ {omega1_sq_ref:.12f}"
    )


def test_optimal_vector_aligns_with_eigenvector():
    """x* normalizado deve ser paralelo ao autovetor do menor autovalor."""
    x0 = np.array([1.0, 1.0, 1.0])
    _, eigenvectors = np.linalg.eigh(K)
    v1 = eigenvectors[:, 0]

    results, _ = run_methods(x0)
    for name, res in results.items():
        xn = res.x / np.linalg.norm(res.x)
        # |cos θ| = |xn · v1| deve ser ≈ 1 (vetores paralelos ou antiparalelos)
        cosine = abs(np.dot(xn, v1))
        assert cosine == pytest.approx(1.0, abs=1e-5), (
            f"{name}: alinhamento com v1 = {cosine:.8f}"
        )
