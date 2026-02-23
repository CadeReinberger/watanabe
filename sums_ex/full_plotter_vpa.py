
# full_plotter_vpa.py
# High-precision (vpa-like) rewrite of full_plotter.py using mpmath (mp.dps=100 by default).
#
# Notes:
# - mpmath is the typical "vpa" analogue in Python. Set mp.mp.dps to control digits.
# - This version keeps the same math but evaluates everything in arbitrary precision
#   to avoid under/overflow and catastrophic cancellation in coefficient generation.

import mpmath as mp
from matplotlib import pyplot as plt
from tqdm import tqdm

# ====== Precision control ======
MP_DPS = 100   # "vpa 100 digits"
mp.mp.dps = MP_DPS

def mp_linspace(a, b, n):
    """mpmath-friendly linspace returning a Python list of mp.mpf."""
    if n <= 1:
        return [mp.mpf(a)]
    a = mp.mpf(a); b = mp.mpf(b)
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def compute_g(N):
    # First, let's make the h in closed form (high precision)
    h = [mp.mpf('0')] * (N + 1)
    h[0] = mp.mpf('1')
    pi2 = mp.pi ** 2
    for n in range(1, N + 1):
        s = mp.mpf('0')
        for k in range(n):
            s += mp.mpf('1') / ((2 * k + 1) * (2 * n - 2 * k - 1))
        h[n] = (mp.mpf('16') / pi2) * s

    # Now, we make the g via series inversion
    g = [mp.mpf('0')] * (N + 1)
    g[0] = mp.mpf('1') / h[0]
    for n in range(1, N + 1):
        s = mp.mpf('0')
        for k in range(1, n + 1):
            s += h[k] * g[n - k]
        g[n] = -(mp.mpf('1') / h[0]) * s

    return g

def compute_alpha(N):
    # Compute the I first (high precision)
    I = [mp.mpf('0')] * (N + 1)
    I0 = mp.mpf('0.5') * mp.atan(2)
    I[0] = I0
    m025 = mp.mpf('-0.25')
    p025 = mp.mpf('0.25')
    for k in range(1, N + 1):
        first_term = I0 * (m025 ** k)
        second_sum = mp.mpf('0')
        for j in range(k):
            second_sum += (m025 ** (k - j - 1)) / (2 * j + 1)
        I[k] = first_term + p025 * second_sum

    # Next, compute the a coefficients using loggamma in high precision
    a = [[mp.mpf('0')] * (N + 1) for _ in range(N + 1)]
    log4 = mp.log(4)
    for n in range(N + 1):
        for l in range(n + 1):
            lg = (mp.loggamma(2 * (n + l) + 1)
                  - n * log4
                  - mp.loggamma(2 * l + 1)
                  - mp.loggamma(n - l + 1)
                  - mp.loggamma(n + l + 1))
            val = mp.e ** lg
            if (n + l) % 2 == 1:
                val = -val
            a[n][l] = val

    # Finally, compute alpha
    alpha = [mp.mpf('0')] * (N + 1)
    for n in range(N + 1):
        s = mp.mpf('0')
        for l in range(n + 1):
            s += a[n][l] * I[l]
        alpha[n] = (4 * n + 1) * s

    return alpha

def compute_lambda(N):
    lam = [mp.mpf('0')] * (N + 1)
    lam[0] = mp.mpf('1')
    for n in range(1, N + 1):
        lam[n] = -lam[n - 1] / n
    print(lam)
    return lam

def make_errors_one(N, M=1000, zmax=20):
    z_ev = mp_linspace(0, zmax, M)
    f_ex = []
    for z in z_ev:
        ez = mp.e ** (-z)
        f_ex.append(mp.mpf('1') / (16 * ez**2 - 16 * ez + 5))

    g = compute_g(N)
    u = []
    half_pi = mp.mpf('0.5') * mp.pi
    for z in z_ev:
        ez = mp.e ** (-z)
        u.append(mp.tanh(half_pi * (2 * ez - 1)))

    f_cur = [mp.mpf('0')] * M
    ns = list(range(N + 1))
    errs = [mp.mpf('0')] * (N + 1)

    for n, gn in enumerate(tqdm(g, desc="Strip transform", leave=False)):
        for i in range(M):
            f_cur[i] += gn * (u[i] ** (2 * n))
        maxerr = mp.mpf('0')
        for i in range(M):
            e = abs(f_ex[i] - f_cur[i])
            if e > maxerr:
                maxerr = e
        errs[n] = maxerr

    return ns, errs

def make_errors_two(N, M=1000, zmax=20):
    z_ev = mp_linspace(0, zmax, M)
    f_ex = []
    for z in z_ev:
        ez = mp.e ** (-z)
        f_ex.append(mp.mpf('1') / (16 * ez**2 - 16 * ez + 5))

    alpha = compute_alpha(N)
    u = []
    for z in z_ev:
        ez = mp.e ** (-z)
        u.append(2 * ez - 1)

    f_cur = [mp.mpf('0')] * M
    ns = list(range(N + 1))
    errs = [mp.mpf('0')] * (N + 1)

    for n, an in enumerate(tqdm(alpha, desc="Legendre expansion", leave=False)):
        deg = 2 * n
        for i in range(M):
            f_cur[i] += an * mp.legendre(deg, u[i])
        maxerr = mp.mpf('0')
        for i in range(M):
            e = abs(f_ex[i] - f_cur[i])
            if e > maxerr:
                maxerr = e
        errs[n] = maxerr

    return ns, errs

def make_errors_three(N, M=1000, zmax=20):
    z_ev = mp_linspace(0, zmax, M)
    f_ex = []
    u = []
    for z in z_ev:
        ez = mp.e ** (-z)
        denom = 16 * ez**2 - 16 * ez + 5
        f_ex.append(mp.mpf('1') / denom)
        u.append(mp.log(denom))

    lam = compute_lambda(N)

    f_cur = [mp.mpf('0')] * M
    ns = list(range(N + 1))
    errs = [mp.mpf('0')] * (N + 1)

    for n, ln in enumerate(tqdm(lam, desc="Log expansion", leave=False)):
        for i in range(M):
            f_cur[i] += ln * (u[i] ** (n))
        maxerr = mp.mpf('0')
        for i in range(M):
            e = abs(f_ex[i] - f_cur[i])
            if e > maxerr:
                maxerr = e
        errs[n] = maxerr

    return ns, errs

def fit_prefactor_fixed_rate(ns, errs, rate, tail_count=5):
    """Least-squares fit for C in errs[N] ~ C * rate**N over the last tail_count points."""
    use_ns = ns[-tail_count:]
    use_errs = errs[-tail_count:]
    num = mp.mpf('0')
    den = mp.mpf('0')
    for n, e in zip(use_ns, use_errs):
        rn = rate ** n
        num += e * rn
        den += rn * rn
    return num / den if den != 0 else mp.mpf('0')

def make_combined_error_plot(N=100, M=1000, zmax=20):
    one_ns, one_errs = make_errors_one(N, M=M, zmax=zmax)
    two_ns, two_errs = make_errors_two(N, M=M, zmax=zmax)
    three_ns, three_errs = make_errors_three(N, M=M, zmax=zmax)

    r_one = mp.tanh(mp.pi / 2)
    phi = (1 + mp.sqrt(5)) / 2
    r_two = 1 / phi
    A = fit_prefactor_fixed_rate(one_ns, one_errs, r_one, tail_count=5)
    B = fit_prefactor_fixed_rate(two_ns, two_errs, r_two, tail_count=5)
    one_fit_errs = [A * (r_one ** n) for n in one_ns]
    two_fit_errs = [B * (r_two ** n) for n in two_ns]

    # Convert mp -> float for plotting (matplotlib expects float)
    one_errs_f = [float(e) for e in one_errs]
    two_errs_f = [float(e) for e in two_errs]
    three_errs_f = [float(e) for e in three_errs]
    one_fit_errs_f = [float(e) for e in one_fit_errs]
    two_fit_errs_f = [float(e) for e in two_fit_errs]

    plt.figure(dpi=300)
    plt.semilogy(one_ns, one_errs_f, 'b-', label='Strip Transform (mp)')
    plt.semilogy(two_ns, two_errs_f, 'g-', label='Legendre Expansion (mp)')
    plt.semilogy(three_ns, three_errs_f, 'r-', label='Log Expansion (mp)')
    plt.semilogy(one_ns, one_fit_errs_f, 'b--',
                 label='Fit: A*(tanh(pi/2))^N')
    plt.semilogy(two_ns, two_fit_errs_f, 'g--',
                 label='Fit: B*(1/phi)^N')

    plt.ylim(bottom=1e-16)
    plt.xlabel('Expansion N')
    plt.ylabel('L_infty Error')
    plt.title(f'Max Error for All Expansions (mp.dps={MP_DPS})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    make_combined_error_plot(N=40, M=1000, zmax=20)
