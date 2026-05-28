"""
Problema 1 - Frequência Natural Fundamental (Quociente de Rayleigh)

Minimizar R(x) = (x^T K x) / (x^T M x)
para encontrar omega_1^2 do sistema massa-mola com 3 graus de liberdade.
"""

import numpy as np
import matplotlib.pyplot as plt
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
    return (x @ K @ x) / (x @ M @ x)


def gradient(x):
    """
    Gradiente analítico de R(x).

    Derivando R = f/g com f = x^T K x, g = x^T M x:
        ∇R(x) = (2 / g) * (K x − R(x) · M x)

    Nulo exatamente nos autovetores de K (pontos críticos de R).
    """
    g = x @ M @ x
    R = (x @ K @ x) / g
    return (2.0 / g) * (K @ x - R * (M @ x))


def hessian_vp(x, p):
    """
    Produto Hessiana-vetor H_R(x) · p para Newton-CG.

    Derivando ∇R(x) = (2/g)(Kx − R·Mx) por x:
        H_R = (2/g)(K − R·M) − (2/g)(∇R·(Mx)^T + Mx·(∇R)^T)

    No produto direcional:
        H_R p = (2/g)[(K − R·M)p − (Mx·p)·∇R − (∇R·p)·Mx]
    """
    g = x @ M @ x
    R = (x @ K @ x) / g
    grad = gradient(x)
    Mx = M @ x
    return (2.0 / g) * ((K - R * M) @ p - np.dot(Mx, p) * grad - np.dot(grad, p) * Mx)


def _make_callback(history):
    """Callback que armazena R(x) a cada passo do otimizador."""
    def cb(intermediate):
        # scipy >= 1.7: Nelder-Mead passa OptimizeResult; demais passam vetor x
        x = intermediate.x if hasattr(intermediate, "x") else intermediate
        history.append(rayleigh_quotient(x))
    return cb


def run_methods(x0):
    """
    Executa BFGS, Nelder-Mead e Newton-CG a partir de x0.

    Retorna:
        results  : dict {nome -> OptimizeResult}
        histories: dict {nome -> lista de R(x) por passo}
    """
    configs = [
        (
            "BFGS",
            dict(
                method="BFGS",
                jac=gradient,
                tol=1e-12,
                options={"maxiter": 1000},
            ),
        ),
        (
            "Nelder-Mead",
            dict(
                method="Nelder-Mead",
                tol=1e-12,
                options={"maxiter": 10_000},
            ),
        ),
        (
            "Newton-CG",
            dict(
                method="Newton-CG",
                jac=gradient,
                hessp=hessian_vp,
                tol=1e-12,
                options={"maxiter": 1000},
            ),
        ),
    ]

    results = {}
    histories = {}
    for name, kwargs in configs:
        hist = [rayleigh_quotient(x0)]
        res = minimize(rayleigh_quotient, x0.copy(), callback=_make_callback(hist), **kwargs)
        results[name] = res
        histories[name] = hist

    return results, histories


def print_results(results, eigenvalue_ref):
    """Imprime tabela comparativa e vetores ótimos normalizados."""
    print(f"\n{'Método':<12} {'R(x*)':<18} {'ω₁ (rad/s)':<18} {'Erro rel.':<12} {'nit':>5} {'nfev':>6}  Sucesso")
    print("-" * 80)
    for name, res in results.items():
        R_opt = res.fun
        omega1 = np.sqrt(max(R_opt, 0.0))
        err = abs(R_opt - eigenvalue_ref) / eigenvalue_ref
        print(
            f"{name:<12} {R_opt:<18.12f} {omega1:<18.12f} "
            f"{err:<12.2e} {res.nit:>5} {res.nfev:>6}  {res.success}"
        )

    print("\nVetores ótimos x* (normalizados para comparação com autovetor):")
    for name, res in results.items():
        xn = res.x / np.linalg.norm(res.x)
        # garante sinal canônico: primeira componente positiva
        if xn[0] < 0:
            xn = -xn
        print(f"  {name:<12}: [{xn[0]:+.8f}  {xn[1]:+.8f}  {xn[2]:+.8f}]")


def plot_convergence(histories, eigenvalue_ref, save_path="convergencia_q1.png"):
    """
    Dois painéis de convergência para o relatório:
      Esquerdo : R(x) absoluto por passo.
      Direito  : |R(x) − ω₁²| em escala logarítmica (revela ordem de convergência).
    """
    palette = {"BFGS": "#2274A5", "Nelder-Mead": "#E57C23", "Newton-CG": "#2CA02C"}
    styles  = {"BFGS": "-",       "Nelder-Mead": "--",       "Newton-CG": "-."}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Convergência dos métodos — Quociente de Rayleigh (Problema 1)",
        fontsize=13, fontweight="bold",
    )

    # --- Painel esquerdo: R(x) por passo ---
    for name, hist in histories.items():
        ax0.plot(
            np.arange(len(hist)), hist,
            label=name, color=palette[name], ls=styles[name], lw=2,
        )
    ax0.axhline(eigenvalue_ref, color="crimson", ls=":", lw=1.5, label=r"$\omega_1^2$ referência")
    ax0.set_xlabel("Passo (callback)", fontsize=11)
    ax0.set_ylabel(r"$R(\mathbf{x})$", fontsize=11)
    ax0.set_title("Valor do quociente de Rayleigh por iteração", fontsize=11)
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.35)

    # --- Painel direito: erro absoluto em log ---
    for name, hist in histories.items():
        errs = np.maximum(np.abs(np.array(hist) - eigenvalue_ref), 1e-16)
        ax1.semilogy(
            np.arange(len(errs)), errs,
            label=name, color=palette[name], ls=styles[name], lw=2,
        )
    ax1.set_xlabel("Passo (callback)", fontsize=11)
    ax1.set_ylabel(r"$|R(\mathbf{x}) - \omega_1^2|$", fontsize=11)
    ax1.set_title("Erro absoluto em escala logarítmica", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", format="png")
    print(f"\nGráfico salvo em: {save_path}")
    plt.show()


def plot_sensitivity(
    eigenvalue_ref,
    n_trials=50,
    seed=42,
    save_path="sensibilidade_q1.png",
):
    """
    Sorteia n_trials pontos iniciais aleatórios, roda BFGS em cada um e plota:
      Esquerdo : histograma de R(x*) — deve concentrar em ω₁² (bacia global).
      Direito  : histograma de nit   — variabilidade do custo conforme x₀.
    """
    rng = np.random.default_rng(seed)

    # Pontos com direções e normas variadas (normal + escala aleatória)
    directions = rng.standard_normal((n_trials, 3))
    scales = rng.uniform(0.1, 20.0, size=n_trials)
    initial_points = directions * scales[:, None]

    r_stars = []
    n_iters = []
    n_failed = 0

    for x0 in initial_points:
        res = minimize(
            rayleigh_quotient, x0,
            method="BFGS",
            jac=gradient,
            tol=1e-10,
            options={"maxiter": 1000},
        )
        if res.success or res.fun < eigenvalue_ref + 1e-4:
            r_stars.append(res.fun)
            n_iters.append(res.nit)
        else:
            n_failed += 1

    r_stars = np.array(r_stars)
    n_iters = np.array(n_iters)
    n_ok = len(r_stars)

    print(f"\nSensibilidade ao ponto inicial ({n_trials} ensaios, BFGS):")
    print(f"  Convergidos : {n_ok}/{n_trials}")
    print(f"  R(x*) — min: {r_stars.min():.10f}  max: {r_stars.max():.10f}  std: {r_stars.std():.2e}")
    print(f"  nit   — min: {n_iters.min()}  max: {n_iters.max()}  média: {n_iters.mean():.1f}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Sensibilidade ao ponto inicial — BFGS ({n_trials} ensaios aleatórios)",
        fontsize=13, fontweight="bold",
    )

    # --- Painel esquerdo: histograma de R(x*) ---
    # Usa bins bem estreitos para evidenciar a concentração
    spread = max(r_stars.max() - r_stars.min(), 1e-8)
    bins_r = np.linspace(r_stars.min() - spread * 0.5, r_stars.max() + spread * 0.5, 30)
    ax0.hist(r_stars, bins=bins_r, color="#2274A5", edgecolor="white", linewidth=0.6)
    ax0.axvline(eigenvalue_ref, color="crimson", ls="--", lw=2, label=rf"$\omega_1^2 = {eigenvalue_ref:.6f}$")
    ax0.set_xlabel(r"$R(\mathbf{x}^*)$", fontsize=11)
    ax0.set_ylabel("Frequência", fontsize=11)
    ax0.set_title("Valor ótimo obtido por ensaio", fontsize=11)
    ax0.legend(fontsize=10)
    ax0.grid(True, axis="y", alpha=0.35)

    # Anotação de robustez
    ax0.text(
        0.05, 0.92,
        f"Todos {n_ok} ensaios → mesmo mínimo",
        transform=ax0.transAxes,
        fontsize=9, color="#1a5276",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#d6eaf8", "ec": "#2274A5", "alpha": 0.8},
    )

    # --- Painel direito: histograma de nit ---
    bins_nit = np.arange(n_iters.min(), n_iters.max() + 2) - 0.5
    ax1.hist(n_iters, bins=bins_nit, color="#2CA02C", edgecolor="white", linewidth=0.6)
    ax1.axvline(n_iters.mean(), color="crimson", ls="--", lw=2,
                label=f"média = {n_iters.mean():.1f} it.")
    ax1.set_xlabel("Número de iterações (nit)", fontsize=11)
    ax1.set_ylabel("Frequência", fontsize=11)
    ax1.set_title("Custo de convergência por ensaio", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", format="png")
    print(f"Gráfico salvo em: {save_path}")
    plt.show()


def main():
    # Ponto inicial genérico — não ortogonal a nenhum autovetor de K
    x0 = np.array([1.0, 1.0, 1.0])

    # Ground truth via decomposição espectral
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    omega1_sq_ref = eigenvalues[0]
    omega1_ref    = np.sqrt(omega1_sq_ref)

    print("=" * 65)
    print("Problema 1 — Frequência Natural Fundamental (Quociente de Rayleigh)")
    print("=" * 65)
    print(f"\nAutovalores de K (np.linalg.eigh): {eigenvalues}")
    print(f"ω₁²  = {omega1_sq_ref:.12f}")
    print(f"ω₁   = {omega1_ref:.12f}  rad/s")
    print(f"v₁   = [{eigenvectors[0,0]:+.8f}  {eigenvectors[1,0]:+.8f}  {eigenvectors[2,0]:+.8f}]")
    print(f"\nPonto inicial: x0 = {x0}")

    results, histories = run_methods(x0)

    print_results(results, omega1_sq_ref)
    plot_convergence(histories, omega1_sq_ref)
    plot_sensitivity(omega1_sq_ref)


if __name__ == "__main__":
    main()
