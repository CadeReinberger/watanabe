#!/usr/bin/env python3
"""Solve a 1-parameter third-order ODE with shooting and make requested plots.

ODE:
    alpha(alpha+1) f''' + f * f'' * |f''|^(1-alpha) = 0
BCs:
    f(0)=0, f'(0)=1, f'(L) ~ 0 (domain-truncated infinity)

Outputs:
    - new_thinning_asymptotic.png  (alpha=0.3, log-log vs asymptotic A*x^p)
    - new_thickening_deriv.png     (alpha=1.4, f'''' vs x)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import warnings


def ode_system(alpha: float):
    c = alpha * (alpha + 1.0)

    def rhs(_x: float, y: np.ndarray) -> np.ndarray:
        f, fp, fpp = y
        # f'' * |f''|^(1-alpha) = sign(f'') * |f''|^(2-alpha)
        nonlinear = np.sign(fpp) * (np.abs(fpp) ** (2.0 - alpha))
        fppp = -(f * nonlinear) / c
        return np.array([fp, fpp, fppp], dtype=float)

    return rhs


def integrate_profile(
    alpha: float,
    fpp0: float,
    L: float,
    npts: int = 3000,
    method: str = "LSODA",
    rtol: float = 1e-7,
    atol: float = 1e-9,
):
    rhs = ode_system(alpha)
    xs = np.linspace(0.0, L, npts)
    y0 = np.array([0.0, 1.0, fpp0], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="lsoda:.*")
        sol = solve_ivp(rhs, (0.0, L), y0, method=method, t_eval=xs, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(
            f"solve_ivp failed for alpha={alpha}, f''(0)={fpp0}, method={method}: {sol.message}"
        )
    return sol


def find_shooting_fpp0(alpha: float, L: float) -> float:
    """Find f''(0) so that f'(L) ~ 0."""

    def residual(fpp0: float) -> float:
        sol = integrate_profile(
            alpha,
            fpp0,
            L,
            npts=1200,
            method="DOP853",
            rtol=1e-7,
            atol=1e-9,
        )
        return sol.y[1, -1]

    # Scan negative values (physically expected for monotone-decaying f').
    scan = -np.linspace(0, 2, 200) # -np.logspace(-4, 2, 50)
    vals = []
    for s in scan:
        try:
            vals.append((s, residual(s)))
        except Exception:
            continue

    bracket = None
    for (s0, r0), (s1, r1) in zip(vals[:-1], vals[1:]):
        if np.isfinite(r0) and np.isfinite(r1) and (r0 == 0.0 or r1 == 0.0 or r0 * r1 < 0.0):
            bracket = (s0, s1)
            break

    if bracket is None:
        msg = "\n".join([f"  fpp0={s:.4e}, residual=f'(L)={r:.4e}" for s, r in vals[:10]])
        raise RuntimeError(f"Could not bracket shooting root for alpha={alpha}. Sample residuals:\n{msg}")

    root = root_scalar(
        residual, bracket=bracket, method="brentq", xtol=1e-10, rtol=1e-8, maxiter=120
    )
    if not root.converged:
        raise RuntimeError(f"Shooting solve did not converge for alpha={alpha}")
    return float(root.root)


def asymptotic_prefactor(alpha: float) -> tuple[float, float]:
    p = (1.0 - 2.0 * alpha) / (2.0 - alpha)
    A = (
        alpha
        * (alpha + 1.0)
        * (2.0 - p)
        / ((p * (1.0 - p)) ** (1.0 - alpha))
    ) ** (1.0 / (2.0 - alpha))
    return p, A


def compute_f4(alpha: float, f: np.ndarray, fp: np.ndarray, fpp: np.ndarray, fppp: np.ndarray) -> np.ndarray:
    """Compute f'''' from differentiated ODE in closed form.

    From C f''' + f g = 0, with g = f''|f''|^(1-alpha), C=alpha(alpha+1):
        C f'''' + f' g + f g' = 0
        g' = (2-alpha)|f''|^(1-alpha) f'''
    so
        f'''' = -(f' g + f (2-alpha)|f''|^(1-alpha) f''') / C
    """
    c = alpha * (alpha + 1.0)
    g = fpp * (np.abs(fpp) ** (1.0 - alpha))
    gp = (2.0 - alpha) * (np.abs(fpp) ** (1.0 - alpha)) * fppp
    return -(fp * g + f * gp) / c


def plot_alpha_03():
    alpha = 0.1
    L = 4000
    s = find_shooting_fpp0(alpha, L)
    sol = integrate_profile(alpha, s, 100, npts=6000, rtol=1e-12, atol=1e-14)

    x = sol.t
    f = sol.y[0]
    p, A = asymptotic_prefactor(alpha)
    fasym = A * (x ** p)

    mask = (x > 1e-6) & (f > 0.0) & np.isfinite(fasym) & (fasym > 0.0)

    plt.figure(figsize=(8, 5.5))
    plt.loglog(x[mask], f[mask], lw=3.2, label=r"Numerical Power Law Solution, Shear-Thinning, alpha < 1/2") #alpha={alpha}")
    plt.loglog(x[mask], fasym[mask], "--", lw=3.0, label=r"Asymptotic $A \eta^p$")
    plt.xlabel(r"\eta")
    plt.ylabel(r"f(\eta)")
    plt.title("New Solutiuon Asymptotic For Shear-Thinning Blasius")
    plt.legend()
    plt.tight_layout()
    plt.savefig("new_thinning_asymptotic.png", dpi=200)
    plt.close()

    print(f"alpha={alpha}: shooting f''(0)={s:.10e}; saved new_thinning_asymptotic.png")


def plot_alpha_14_f4():
    alpha = 1.5
    L = 30.0
    s = find_shooting_fpp0(alpha, L)
    # sol = integrate_profile(alpha, s, .5 * L, npts=10000, rtol=.5*1e-7, atol=1e-9)
    sol = integrate_profile(alpha, s, .6 * L, npts=1000, rtol=2.5e-14, atol=1e-16)

    x = sol.t
    f, fp, fpp = sol.y
    # Reconstruct f''' directly from ODE for consistency.
    c = alpha * (alpha + 1.0)
    fppp = -(f * (fpp * (np.abs(fpp) ** (1.0 - alpha)))) / c
    f4 = compute_f4(alpha, f, fp, fpp, fppp)

    plt.figure(figsize=(8, 5.5))
    plt.plot(x, f4, lw=3, label=r"$f''''(x)$")
    plt.xlabel(r"\eta")
    plt.ylabel(r"$f''''(\eta)$")
    plt.title("Singular Solution in Shear-Thickening Case for Power-Law Sakiadis")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("new_thickening_deriv.png", dpi=200)
    plt.close()

    print(f"alpha={alpha}: shooting f''(0)={s:.10e}; saved new_thickening_deriv.png")


def main():
    plot_alpha_03()
    plot_alpha_14_f4()


if __name__ == "__main__":
    main()
