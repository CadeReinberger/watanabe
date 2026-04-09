from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


@dataclass
class FilmParams:
    # mu and gamma should be from the other sims, chloroform gamma, and mu from viscometer data
    mu: float = .537          # viscosity, poise = g/(cm·s). # 100 * viscosity of chloroform
    gamma: float = 27.1         # surface tension, dyn/cm = g/s² # chloroform
    h0: float = 0.01              # initial film thickness, cm. # (50 microns-100 microns)
    v_web: float = 0.0105          # web velocity, cm/s # (105 microns/s)
    beta: float = field(default_factory=lambda: np.radians(45))  # radians
    L: float = .1               # domain length, cm (maybe a few mm). 
    v_evap: float = field(init=False)

    def __post_init__(self):
        self.v_evap = self.h0 * np.sin(self.beta) * self.v_web / (2 * self.L)


def ode_system(x, y, params):
    p = params
    coeff = 3 * p.mu / p.gamma
    y0, y1, y2, y3 = y
    h_safe = np.where(np.abs(y0) > 1e-8, y0, 1e-8)
    dy3 = (-y1 * y3 - coeff * (p.v_evap + p.v_web * np.sin(p.beta) * y1)) / h_safe
    return np.vstack([y1, y2, y3, dy3])


def boundary_conditions(ya, yb, params):
    p = params
    bc3 = -p.h0 / (2 * p.L)
    bc4 = -3 * p.mu * p.v_web * np.sin(p.beta) / (2 * p.gamma * p.h0**2) 
    return np.array([
        ya[0] - p.h0,   # h(0) = h0
        yb[0],           # h(L) = 0
        yb[1] - bc3,     # h'(L) = -h0/(2L)
        ya[3] - bc4,     # h'''(0) = bc4
    ])


def solve_bvp_problem(params):
    p = params
    eps = 1e-3 * p.L
    x_mesh = np.linspace(0, p.L - eps, 500)

    # Initial guess: quadratic profile satisfying BCs 1, 2, 3
    y_guess = np.zeros((4, x_mesh.size))
    y_guess[0] = p.h0 * (x_mesh**2 / (2 * p.L**2) - 3 * x_mesh / (2 * p.L) + 1)
    y_guess[1] = p.h0 * (x_mesh / p.L**2 - 3 / (2 * p.L))
    y_guess[2] = p.h0 / p.L**2
    y_guess[3] = 0.0

    ode_fn = lambda x, y: ode_system(x, y, params)
    bc_fn = lambda ya, yb: boundary_conditions(ya, yb, params)

    # Step 1: coarse solve to get a good initial guess
    sol = solve_bvp(ode_fn, bc_fn, x_mesh, y_guess, tol=1e-3, max_nodes=1_000_000)
    if not sol.success:
        print(f"WARNING: coarse solve_bvp did not converge: {sol.message}")

    # Step 2: refine from coarse solution
    # Note: h''''∝1/h diverges as h→0 at x=L, so tight tolerances require
    # exponentially many nodes there. tol=1e-3 is adequate for plotting.
    sol2 = solve_bvp(ode_fn, bc_fn, sol.x, sol.y, tol=1e-5, max_nodes=2_000_000)
    if sol2.success:
        sol = sol2
        print("Refined solve succeeded.")
    else:
        print(f"Note: refined solve did not converge (h→0 singularity at x=L); "
              f"using coarse solution (tol=1e-3, {sol.x.size} nodes).")

    x_fine = np.linspace(0, p.L - eps, 2000)
    y_fine = sol.sol(x_fine)
    h_vals = y_fine[0]
    h_prime = y_fine[1]
    h_pprime = y_fine[2]
    h_tprime = y_fine[3]

    return x_fine, h_vals, h_prime, h_pprime, h_tprime, sol

def compute_curvature(sol, params, x):
    """
    Compute path curvature from the solve_bvp solution.

    Parameters
    ----------
    sol : scipy.integrate.solve_bvp result
    params : dataclass with mu, gamma, v_web, beta
    x : 1-d array of evaluation points
    """
    p = params

    # Evaluate the BVP state vector at the requested x
    y = sol.sol(x)          # shape (4, N)
    h    = y[0]
    hp   = y[1]             # h'
    hpp  = y[2]             # h''
    hppp = y[3]             # h'''

    # h'''' straight from the ODE RHS (4th component)
    h_safe = np.where(np.abs(h) > 1e-8, h, 1e-8)
    hpppp = (-hp * hppp
             - (3 * p.mu / p.gamma)
               * (p.v_evap + p.v_web * np.sin(p.beta) * hp)) / h_safe

    # Capillary contribution and its x-derivative
    coeff = p.gamma / (2 * p.mu)
    phi  = coeff * hppp * h**2
    dphi = coeff * (hpppp * h**2 + 2 * hppp * hp * h)

    # Velocity components
    u = p.v_web * np.sin(p.beta) + phi
    v = -p.v_web * np.cos(p.beta)

    # Curvature: |v * u * u'| / (u^2 + v^2)^(3/2)
    kappa = np.abs(v * u * dphi) / (u**2 + v**2)**1.5

    return kappa

def theta_L(theta0, sol, params):
    """
    Compute theta_L for an array of theta_0 values.

    Parameters
    ----------
    theta0  : 1-d array of initial angles
    sol     : scipy.integrate.solve_bvp result
    params  : dataclass with mu, gamma, v_web, beta, L

    Returns
    -------
    thetaL  : 1-d array, same shape as theta0
    """
    p = params
    coeff = p.gamma / (3 * p.mu) # 3 is for average, 2 is for top

    # Evaluate u(x) = v_web*sin(beta) + (gamma/2mu)*h'''*h^2 at endpoints
    y0 = sol.sol(.01*p.L)
    yL = sol.sol(.99*p.L)

    u0 = p.v_web * np.sin(p.beta) + coeff * y0[3] * y0[0]**2
    uL = p.v_web * np.sin(p.beta) + coeff * yL[3] * yL[0]**2

    I = np.log(np.abs(uL / u0))
    
    # Some prints to check some things
    #print(f'u0: {u0}')
    #print(f'uL: {uL}')
    
    # theta_L for each theta_0
    T = np.arctanh(np.cos(2 * theta0)) + I
    thetaL = 0.5 * np.arccos(np.tanh(T))

    return thetaL

def plot_curvature_top(sol, params, x):
    kappa = compute_curvature(sol, params, x)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, kappa, lw=2, color='tab:purple')
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("κ (cm⁻¹)")
    ax.set_title("Streamline Curvature at Top Surface")
    fig.savefig("plot7_curvature_top.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved plot7_curvature_top.png")


def plot_film_thickness(x, h, params):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, h, lw=2)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("h (cm)")
    ax.set_title("Film Thickness Profile")
    fig.savefig("plot1_film_thickness.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved plot1_film_thickness.png")


def plot_pressure(x, h_tprime, params):
    P = -params.gamma * h_tprime
    mask = x <= 0.99 * params.L
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x[mask], P[mask], lw=2, color='tab:orange')
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("P (dyn/cm²)")
    ax.set_title("Pressure Distribution")
    fig.savefig("plot2_pressure.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved plot2_pressure.png")


def plot_x_velocity(x, u_x_bar, params, top=False):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, u_x_bar, lw=2, color='tab:green')
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("ū_x (cm/s)")
    ax.set_title("Depth-Averaged x-Velocity" if not top else "x-Velocity at Top")
    fig_name = "plot3_x_velocity.png" if not top else "plot6_top_x_velocity.png"
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(fig_name)


def plot_streamplot(x_fine, u_x_bar_1d, params, top=False):
    p = params
    eps = 1e-8 * p.L
    nx, ny = 200, 200
    x_grid = np.linspace(0, p.L - eps, nx)
    y_grid = np.linspace(-2 * p.L, 2 * p.L, ny)
    X, Y = np.meshgrid(x_grid, y_grid)  # shape (ny, nx)

    # Interpolate u_x_bar onto the grid (no y-dependence)
    u_x_1d = np.interp(x_grid, x_fine, u_x_bar_1d)
    U_x = np.tile(u_x_1d, (ny, 1))  # broadcast over y rows

    U_y = np.full_like(U_x, -p.v_web * np.cos(p.beta))
    speed = np.sqrt(U_x**2 + U_y**2)

    fig, ax = plt.subplots(figsize=(6, 8))
    pcm = ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto')

    # --- constant web-velocity field [v_web*sin(beta), -v_web*cos(beta)] ---
    d_x = p.v_web * np.sin(p.beta)
    d_y = -p.v_web * np.cos(p.beta)
    mag = np.hypot(d_x, d_y)
    ux_w, uy_w = d_x / mag, d_y / mag   # unit direction
    perp_x, perp_y = -uy_w, ux_w        # perpendicular (for line spacing)

    x_min, x_max = 0, p.L
    y_min, y_max = -2 * p.L, 2 * p.L
    diag = np.hypot(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    t_line = np.linspace(-2 * diag, 2 * diag, 2000)
    arrow_len = 0.12 * p.L

    for offset in np.linspace(-diag, diag, 15):
        x0 = cx + offset * perp_x
        y0 = cy + offset * perp_y
        xs = x0 + t_line * ux_w
        ys = y0 + t_line * uy_w
        in_box = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
        if not in_box.any():
            continue
        xs_c, ys_c = xs[in_box], ys[in_box]
        ax.plot(xs_c, ys_c, color='white', linestyle='--', linewidth=1.0,
                alpha=0.7, zorder=1)
        mid = len(xs_c) // 2
        ax.annotate('',
                    xy=(xs_c[mid] + ux_w * arrow_len,
                        ys_c[mid] + uy_w * arrow_len),
                    xytext=(xs_c[mid], ys_c[mid]),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
                    zorder=1)

    ax.streamplot(x_grid, y_grid, U_x, U_y, color='white', linewidth=0.8,
                  arrowsize=1.0, density=1.5)
    fig.colorbar(pcm, ax=ax, label="Speed (cm/s)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Velocity Field Streamlines")
    ax.set_aspect('equal')

    fig_name = "plot4_streamplot.png" if not top else "plot5_topstreamplot.png"

    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(fig_name)

def plot_theta_L(sol, params):
    theta0 = np.linspace(1e-3, np.pi / 2, 500)
    thetaL = theta_L(theta0, sol, params)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(theta0, thetaL, lw=2, color='tab:blue')
    ax.set_xlabel(r"$\theta_0$ (rad)")
    ax.set_ylabel(r"$\theta_L$ (rad)")
    ax.set_title(r"Contact Angle Evolution: $\theta_L$ vs $\theta_0$")
    ax.set_xlim(0, np.pi / 2)
    fig.savefig("plot8_thetaL.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved plot8_thetaL.png")


if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')

    params = FilmParams()
    print(f"v_evap = {params.v_evap:.6e} cm/s")
    print(f"beta   = {params.beta:.6f} rad ({np.degrees(params.beta):.1f} deg)")

    x_fine, h_vals, h_prime, h_pprime, h_tprime, sol = solve_bvp_problem(params)
    print(f"solve_bvp success: {sol.success}")

    coeff = params.gamma / (3 * params.mu)
    u_x_bar = params.v_web * np.sin(params.beta) + coeff * h_vals**2 * h_tprime

    plot_film_thickness(x_fine, h_vals, params)
    plot_pressure(x_fine, h_tprime, params)
    plot_x_velocity(x_fine, u_x_bar, params)
    plot_streamplot(x_fine, u_x_bar, params)

    # Now plot the streamplot at the top
    coeff = params.gamma / (2 * params.mu)
    u_x_top = params.v_web * np.sin(params.beta) + coeff * h_vals**2 * h_tprime
    plot_streamplot(x_fine, u_x_top, params, top=True)
    plot_x_velocity(x_fine, u_x_top, params, top=True)

    x_pressure = x_fine[x_fine <= 0.99 * params.L]
    plot_curvature_top(sol, params, x_pressure)

    plot_theta_L(sol, params)

    print("Done.")
