"""
Problema 3 - Precificação Minimax

Minimizar o pior desvio absoluto da receita em relação a uma meta, entre 4 cenários:

    min_{p1,p2}  max_i | R_i(p1, p2) - R_alvo |,    i = 1, ..., 4

onde
    R_i(p1, p2) = a_{1,i}/p1 + a_{2,i}/p2,   R_alvo = 10000.

A função-objetivo é NÃO DIFERENCIÁVEL (o envelope max|·| introduz quinas no ótimo,
onde >= 2 cenários ficam ativos). Por isso comparamos:

  1. Nelder-Mead  - simplex sem derivadas (candidato principal).
  2. Powell       - direções conjugadas sem derivadas.
  3. SLSQP        - aplicado à REFORMULAÇÃO EPIGRÁFICA suave equivalente
                    (min t  s.a.  -t <= R_i - R_alvo <= t), que recupera a
                    diferenciabilidade e serve de referência de alta precisão.

Resultado: f* = 3000. O MINIMIZADOR NÃO É ÚNICO — ao longo da reta
10/p1 + 8/p2 = 1 tem-se R_2 = 7000 e R_3 = 13000 (ambos com desvio 3000),
um segmento contínuo de ótimos. Os cenários 2 e 3 governam o pior caso;
o cenário 1 (espremido entre 2 e 3) nunca é o pior. Por isso os três métodos
concordam em f* mas podem retornar p* distintos sobre o segmento.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Parâmetros dos cenários (a_{1,i}, a_{2,i})
CENARIOS = np.array([
    [100000,  80000],   # 1 - Mercado Normal
    [ 70000,  56000],   # 2 - Recessão Econômica
    [130000, 104000],   # 3 - Alta Demanda
    [ 90000,  90000],   # 4 - Pressão Competitiva
], dtype=float)

NOMES_CENARIOS = [
    "Mercado Normal",
    "Recessão Econômica",
    "Alta Demanda",
    "Pressão Competitiva",
]

R_ALVO = 10000.0


# --------------------------------------------------------------------------- #
# Funções base
# --------------------------------------------------------------------------- #
def receita_cenario(p, a1, a2):
    """R_i(p1, p2) = a1/p1 + a2/p2."""
    return a1 / p[0] + a2 / p[1]


def receitas(p):
    """Vetor [R_1, ..., R_4] para um dado p = (p1, p2)."""
    return CENARIOS[:, 0] / p[0] + CENARIOS[:, 1] / p[1]


def desvios(p):
    """Vetor de desvios absolutos |R_i(p) - R_alvo|, i = 1..4."""
    return np.abs(receitas(p) - R_ALVO)


def objetivo_minimax(p):
    """
    max_i |R_i(p1, p2) - R_alvo|.

    Guarda de positividade: como a função explode (e troca de sinal) quando
    p -> 0, retornamos +inf fora do ortante positivo para manter a busca
    derivada-livre presa em p1, p2 > 0.
    """
    if p[0] <= 0.0 or p[1] <= 0.0:
        return np.inf
    return float(np.max(desvios(p)))


# --------------------------------------------------------------------------- #
# Reformulação epigráfica (para o SLSQP)
#   variáveis z = [p1, p2, t];  minimiza t
#   s.a.  -t <= R_i - R_alvo <= t   <=>   t - (R_i - R_alvo) >= 0
#                                          t + (R_i - R_alvo) >= 0
# --------------------------------------------------------------------------- #
def _epi_obj(z):
    return z[2]


def _epi_grad(z):
    return np.array([0.0, 0.0, 1.0])


def _epi_constraints():
    """Lista de restrições de desigualdade (g(z) >= 0) para o SLSQP."""
    cons = []
    for i in range(len(CENARIOS)):
        a1, a2 = CENARIOS[i]

        def r_menos_alvo(z, a1=a1, a2=a2):
            return a1 / z[0] + a2 / z[1] - R_ALVO

        def jac_r(z, a1=a1, a2=a2):
            # d/dz de (a1/p1 + a2/p2 - R_alvo); componente t = 0
            return np.array([-a1 / z[0] ** 2, -a2 / z[1] ** 2, 0.0])

        # t - (R_i - R_alvo) >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda z, f=r_menos_alvo: z[2] - f(z),
            "jac": lambda z, jf=jac_r: np.array([0.0, 0.0, 1.0]) - jf(z),
        })
        # t + (R_i - R_alvo) >= 0
        cons.append({
            "type": "ineq",
            "fun": lambda z, f=r_menos_alvo: z[2] + f(z),
            "jac": lambda z, jf=jac_r: np.array([0.0, 0.0, 1.0]) + jf(z),
        })
    return cons


# --------------------------------------------------------------------------- #
# Execução dos métodos
# --------------------------------------------------------------------------- #
def _make_callback(history):
    """Callback que registra f_obj(p) a cada passo (Nelder-Mead / Powell)."""
    def cb(intermediate):
        x = intermediate.x if hasattr(intermediate, "x") else intermediate
        history.append(objetivo_minimax(x[:2]))
    return cb


def _make_callback_epi(history):
    """Callback do SLSQP: z = [p1, p2, t] -> registra f_obj nos preços."""
    def cb(intermediate):
        z = intermediate.x if hasattr(intermediate, "x") else intermediate
        history.append(objetivo_minimax(z[:2]))
    return cb


def run_methods(p0):
    """
    Roda Nelder-Mead, Powell e SLSQP (reformulação epigráfica) a partir de p0.

    Retorna:
        results   : dict {nome -> dict com p, fun, nit, nfev, success, tempo}
        histories : dict {nome -> lista de f_obj por passo}
    """
    results = {}
    histories = {}

    # --- 1. Nelder-Mead (direto sobre f_obj) ---
    hist = [objetivo_minimax(p0)]
    t0 = time.perf_counter()
    res = minimize(
        objetivo_minimax, np.array(p0, float),
        method="Nelder-Mead",
        callback=_make_callback(hist),
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10_000},
    )
    dt = time.perf_counter() - t0
    results["Nelder-Mead"] = _pack(res.x[:2], res.fun, res.nit, res.nfev, res.success, dt)
    histories["Nelder-Mead"] = hist

    # --- 2. Powell (direto sobre f_obj) ---
    hist = [objetivo_minimax(p0)]
    t0 = time.perf_counter()
    res = minimize(
        objetivo_minimax, np.array(p0, float),
        method="Powell",
        callback=_make_callback(hist),
        options={"xtol": 1e-8, "ftol": 1e-8, "maxiter": 10_000},
    )
    dt = time.perf_counter() - t0
    nit = getattr(res, "nit", len(hist))
    results["Powell"] = _pack(res.x[:2], res.fun, nit, res.nfev, res.success, dt)
    histories["Powell"] = hist

    # --- 3. SLSQP sobre a reformulação epigráfica ---
    z0 = np.array([p0[0], p0[1], objetivo_minimax(p0)], float)
    hist = [objetivo_minimax(z0[:2])]
    t0 = time.perf_counter()
    res = minimize(
        _epi_obj, z0,
        method="SLSQP",
        jac=_epi_grad,
        constraints=_epi_constraints(),
        bounds=[(1e-6, None), (1e-6, None), (0.0, None)],
        callback=_make_callback_epi(hist),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    dt = time.perf_counter() - t0
    # f* reportado é o objetivo minimax real nos preços ótimos (não o t)
    f_opt = objetivo_minimax(res.x[:2])
    results["SLSQP (epígrafo)"] = _pack(res.x[:2], f_opt, res.nit, res.nfev, res.success, dt)
    histories["SLSQP (epígrafo)"] = hist

    return results, histories


def _pack(p, fun, nit, nfev, success, tempo):
    return {
        "p": np.asarray(p, float),
        "fun": float(fun),
        "nit": int(nit),
        "nfev": int(nfev),
        "success": bool(success),
        "tempo": float(tempo),
    }


# --------------------------------------------------------------------------- #
# Saídas em texto
# --------------------------------------------------------------------------- #
def print_comparativo(results):
    print(f"\n{'Método':<18} {'p1*':>10} {'p2*':>10} {'f*':>14} "
          f"{'nit':>6} {'nfev':>7} {'t (ms)':>9}  Sucesso")
    print("-" * 88)
    for nome, r in results.items():
        print(f"{nome:<18} {r['p'][0]:>10.5f} {r['p'][1]:>10.5f} {r['fun']:>14.6e} "
              f"{r['nit']:>6} {r['nfev']:>7} {r['tempo']*1e3:>9.2f}  {r['success']}")


def print_desvios(results):
    """Desvios por cenário no ótimo de cada método (evidencia cenários ativos)."""
    print("\nDesvios por cenário no ótimo  |R_i - R_alvo|  (* = cenário ativo):")
    print(f"\n{'Método':<18}", end="")
    for i in range(len(CENARIOS)):
        print(f"{'cen.'+str(i+1):>14}", end="")
    print()
    print("-" * (18 + 14 * len(CENARIOS)))
    for nome, r in results.items():
        d = desvios(r["p"])
        ativo = d >= d.max() - 1e-3 * max(d.max(), 1.0)
        print(f"{nome:<18}", end="")
        for i in range(len(CENARIOS)):
            marca = "*" if ativo[i] else " "
            print(f"{d[i]:>13.4f}{marca}", end="")
        print()

    print("\nLegenda dos cenários:")
    for i, nome in enumerate(NOMES_CENARIOS):
        print(f"  cen.{i+1}: {nome:<22} a1={CENARIOS[i,0]:>7.0f}  a2={CENARIOS[i,1]:>7.0f}")


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_contorno(results, save_path="contorno_q3.png"):
    """
    Mapa de contorno de f(p1, p2) = max_i |R_i - R_alvo| com os pontos finais
    dos métodos sobrepostos. As quinas do envelope max aparecem como vincos.
    """
    p1 = np.linspace(5, 45, 400)
    p2 = np.linspace(5, 45, 400)
    P1, P2 = np.meshgrid(p1, p2)

    # f vetorizado sobre a grade (todos os pontos têm p > 0)
    R = (CENARIOS[:, 0][:, None, None] / P1[None, :, :]
         + CENARIOS[:, 1][:, None, None] / P2[None, :, :])
    Z = np.max(np.abs(R - R_ALVO), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    levels = np.linspace(0, np.percentile(Z, 95), 35)
    cf = ax.contourf(P1, P2, Z, levels=levels, cmap="viridis", extend="max")
    ax.contour(P1, P2, Z, levels=levels[::4], colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(cf, ax=ax, label=r"$f(p_1,p_2)=\max_i\,|R_i-R_{alvo}|$")

    markers = {"Nelder-Mead": "o", "Powell": "s", "SLSQP (epígrafo)": "^"}
    cores   = {"Nelder-Mead": "#E57C23", "Powell": "#D7263D", "SLSQP (epígrafo)": "#F7F7F7"}
    for nome, r in results.items():
        ax.plot(r["p"][0], r["p"][1], markers.get(nome, "x"),
                color=cores.get(nome, "red"), markersize=11, markeredgecolor="black",
                markeredgewidth=1.0, label=f"{nome}: f*={r['fun']:.3g}")

    ax.set_xlabel(r"$p_1$", fontsize=12)
    ax.set_ylabel(r"$p_2$", fontsize=12)
    ax.set_title("Curvas de nível de f e pontos ótimos (Problema 3)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", format="png")
    print(f"\nGráfico salvo em: {save_path}")
    plt.show()


def plot_convergencia(histories, save_path="convergencia_q3.png"):
    """Curva de convergência: f_obj vs. iteração para cada método."""
    palette = {"Nelder-Mead": "#E57C23", "Powell": "#D7263D", "SLSQP (epígrafo)": "#2274A5"}
    styles  = {"Nelder-Mead": "-",       "Powell": "--",       "SLSQP (epígrafo)": "-."}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Convergência dos métodos — Precificação Minimax (Problema 3)",
                 fontsize=13, fontweight="bold")

    for nome, hist in histories.items():
        ax0.plot(np.arange(len(hist)), hist, label=nome,
                 color=palette.get(nome), ls=styles.get(nome, "-"), lw=2)
    ax0.set_xlabel("Passo (callback)", fontsize=11)
    ax0.set_ylabel(r"$f(p)=\max_i|R_i-R_{alvo}|$", fontsize=11)
    ax0.set_title("Objetivo por iteração", fontsize=11)
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.35)

    # erro relativo ao melhor f* encontrado (escala log)
    f_best = min(min(h) for h in histories.values())
    for nome, hist in histories.items():
        errs = np.maximum(np.abs(np.array(hist) - f_best), 1e-12)
        ax1.semilogy(np.arange(len(errs)), errs, label=nome,
                     color=palette.get(nome), ls=styles.get(nome, "-"), lw=2)
    ax1.set_xlabel("Passo (callback)", fontsize=11)
    ax1.set_ylabel(r"$|f - f^\ast|$", fontsize=11)
    ax1.set_title("Erro absoluto em escala logarítmica", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", format="png")
    print(f"Gráfico salvo em: {save_path}")
    plt.show()


# --------------------------------------------------------------------------- #
# Sensibilidade ao ponto inicial
# --------------------------------------------------------------------------- #
def estudo_pontos_iniciais(grade=(10.0, 20.0, 30.0, 40.0)):
    """
    Roda os 3 métodos a partir de uma grade de pontos iniciais e resume a
    dispersão de (p1*, p2*, f*) por método -> robustez ao x0.
    """
    p0_list = [(a, b) for a in grade for b in grade]
    print(f"\n{'='*65}")
    print(f"Sensibilidade ao ponto inicial ({len(p0_list)} pontos da grade {grade})")
    print(f"{'='*65}")

    agreg = {nome: {"f": [], "p1": [], "p2": []}
             for nome in ["Nelder-Mead", "Powell", "SLSQP (epígrafo)"]}

    for p0 in p0_list:
        results, _ = run_methods(p0)
        for nome, r in results.items():
            agreg[nome]["f"].append(r["fun"])
            agreg[nome]["p1"].append(r["p"][0])
            agreg[nome]["p2"].append(r["p"][1])

    print(f"\n{'Método':<18} {'f* (mín)':>14} {'f* (máx)':>14} {'f* (std)':>12} "
          f"{'p1* médio':>11} {'p2* médio':>11}")
    print("-" * 84)
    for nome, d in agreg.items():
        f = np.array(d["f"])
        print(f"{nome:<18} {f.min():>14.6e} {f.max():>14.6e} {f.std():>12.2e} "
              f"{np.mean(d['p1']):>11.4f} {np.mean(d['p2']):>11.4f}")


# --------------------------------------------------------------------------- #
def main():
    print("=" * 65)
    print("Problema 3 — Precificação Minimax")
    print("=" * 65)

    p0 = (20.0, 20.0)
    print(f"\nR_alvo = {R_ALVO:.1f}")
    print(f"Ponto inicial: p0 = {p0}")

    results, histories = run_methods(p0)

    print_comparativo(results)
    print_desvios(results)

    print("\nNota analítica: o ótimo NÃO é único.")
    print("  Ao longo da reta  10/p1 + 8/p2 = 1  valem  R_2 = 7000  e  R_3 = 13000")
    print("  (ambos com desvio 3000 = f*). Cenários 2 e 3 governam o pior caso;")
    print("  por isso os métodos concordam em f* mas retornam p* distintos no segmento.")

    estudo_pontos_iniciais()

    plot_contorno(results)
    plot_convergencia(histories)


if __name__ == "__main__":
    main()
