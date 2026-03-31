# Design Document: Thin-Film Lubrication BVP Solver

## Overview

Build a single Python script (`thin_film_solver.py`) that:

1. Defines physical parameters in a dataclass (CGS units)
2. Solves a 4th-order BVP for the film thickness h(x)
3. Produces four publication-quality matplotlib plots

Use `numpy`, `scipy`, and `matplotlib` only (all available in standard scientific Python).

---

## 1. Parameters Dataclass

Define a `@dataclass` called `FilmParams` with the following float fields, all in **CGS units** (centimeters, grams, seconds).

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| `mu`      | μ      | `0.00386`  | Viscosity of heptane at 25 °C, in poise = g/(cm·s) |
| `gamma`   | γ      | `20.14`    | Surface tension of heptane at 25 °C, in dyn/cm = g/s² |
| `h0`      | h₀     | `0.4`      | 4 mm → 0.4 cm |
| `v_web`   | v_web  | `0.01`     | 100 μm/s → 1×10⁻² cm/s |
| `beta`    | β      | `np.radians(45)` ≈ `0.7854` | 45 degrees, converted to radians for storage |
| `v_evap`  | v_evap | *computed*  | See formula below |
| `L`       | L      | `2.0`      | Domain length, 2 cm |

**Derived parameter:**

```
v_evap = h0 * sin(beta) * v_web / (2 * L)
```

With the values above: `v_evap = 0.4 * sin(π/4) * 0.01 / (2 * 2.0) = 0.4 * 0.7071 * 0.01 / 4 ≈ 7.071×10⁻⁴`.

Store `L` in the dataclass as well for convenience, even though it was given separately.

---

## 2. BVP Formulation and Solution

### The ODE (4th order, on x ∈ [0, L])

```
d/dx [ -γ/(3μ) · h(x) · h‴(x) ] = v_evap + v_web · sin(β) · h′(x)
```

### Expansion into explicit form

Expand the left side:

```
-γ/(3μ) · [ h′ · h‴ + h · h⁗ ] = v_evap + v_web · sin(β) · h′
```

Solve for h⁗:

```
h⁗ = [ -h′ · h‴  -  (3μ/γ) · (v_evap + v_web · sin(β) · h′) ] / h
```

### First-order system

Let `y = [y0, y1, y2, y3] = [h, h′, h″, h‴]`. Then:

```
y0′ = y1
y1′ = y2
y2′ = y3
y3′ = [ -y1 · y3  -  (3μ/γ) · (v_evap + v_web · sin(β) · y1) ] / y0
```

### Boundary conditions (4 total)

| # | Location | Condition | In terms of y |
|---|----------|-----------|---------------|
| 1 | x = 0    | h(0) = h₀ | y0(0) = h₀    |
| 2 | x = L    | h(L) = 0  | y0(L) = 0     |
| 3 | x = L    | h′(L) = −h₀/(2L) | y1(L) = −h₀/(2L) |
| 4 | x = 0    | h‴(0) = −3μ·v_web·sin(β) / (2γ·h₀³) | y3(0) = −3μ·v_web·sin(β) / (2γ·h₀³) |

Pre-computed BC #3 value: `−0.4 / (2 · 2.0) = −0.1`

Pre-computed BC #4 value: `−3 · 0.00386 · 0.01 · sin(π/4) / (2 · 20.14 · 0.4³) ≈ −3.18×10⁻⁵` (verify numerically).

### Handling the singularity at x = L

Since h(L) = 0, the explicit ODE (which divides by h) has a singularity at x = L. To handle this:

- **Do NOT end the domain at x = L.** Instead, use `x ∈ [0, L − ε]` where `ε = 1e-8 * L` (i.e., `2e-8 cm`).
- Apply BCs #2 and #3 at x = L − ε instead of x = L.
- Near x = L, h ≈ h′(L)·(x − L) to leading order, so h(L − ε) ≈ h₀ε/(2L), which is tiny but nonzero.
- The h′(L) BC should keep the solution well-behaved as it approaches L; the offset is just to prevent numerical division by zero.

### Initial guess

Use the quadratic profile that satisfies BCs #1, #2, #3:

```
h_guess(x) = h0 · [ x²/(2L²) − 3x/(2L) + 1 ]
```

Verify: h(0) = h₀ ✓, h(L) = h₀·[1/2 − 3/2 + 1] = 0 ✓, h′(x) = h₀·[x/L² − 3/(2L)], h′(L) = h₀·[1/L − 3/(2L)] = −h₀/(2L) ✓.

This is monotonically decreasing on [0, L] (the vertex is at x = 3L/2, outside the domain), going from h₀ down to 0. Physically sensible for a coating film.

For the initial guess of y = [y0, y1, y2, y3]:
```
y0 = h0 · [ x²/(2L²) − 3x/(2L) + 1 ]
y1 = h0 · [ x/L² − 3/(2L) ]
y2 = h0 / L²
y3 = 0
```

Use a mesh of ~200 points on [0, L − ε].

### Solver

Use `scipy.integrate.solve_bvp`. Increase `max_nodes` to at least 5000 if needed. Check that `sol.success` is True and print a warning if not. Set `tol=1e-8` or adjust as needed for convergence.

### Store the solution

After solving, create a **fine evaluation grid** of ~2000 points on [0, L − ε]:

```python
x_fine = np.linspace(0, L - eps, 2000)
h_vals = sol.sol(x_fine)[0]
h_prime = sol.sol(x_fine)[1]
h_pprime = sol.sol(x_fine)[2]
h_tprime = sol.sol(x_fine)[3]
```

These arrays are used in all subsequent plots.

---

## 3. Plots

Produce **four** separate figures, saved as PNGs (300 dpi). Use a clean, readable style (e.g., `plt.style.use('seaborn-v0_8-whitegrid')` or similar). Label all axes with units. Use linear scales unless noted.

### Plot 1: Film thickness h(x)

- Plot `h_vals` vs `x_fine` on [0, L].
- x-axis: "x (cm)"
- y-axis: "h (cm)"
- Title: "Film Thickness Profile"
- The profile should be monotonically decreasing from h₀ to 0. Make sure the aspect ratio looks good — the y-range will be much smaller than the x-range, so do NOT use `set_aspect('equal')`. Just let matplotlib auto-scale.

### Plot 2: Pressure P(x)

Compute pressure as:

```
P(x) = −γ · h‴(x)
```

i.e., `P = -gamma * h_tprime`.

- Plot `P` vs `x_fine` on [0, L].
- x-axis: "x (cm)"
- y-axis: "P (dyn/cm²)"
- Title: "Pressure Distribution"

### Plot 3: Depth-averaged x-velocity ū_x(x)

Compute:

```
u_x_bar(x) = v_web · sin(β) + γ/(3μ) · h(x)² · h‴(x)
```

**IMPORTANT**: The coefficient is `γ / (3·μ)`, NOT `(γ/3) · μ`.

- Plot `u_x_bar` vs `x_fine` on [0, L].
- x-axis: "x (cm)"
- y-axis: "ū_x (cm/s)"
- Title: "Depth-Averaged x-Velocity"

Note: `u_x_bar` is already the depth-averaged (z-averaged) velocity and is a function of x. No further spatial averaging is needed — just plot it.

### Plot 4: Streamplot with velocity magnitude colormap

Define the 2D velocity field on the rectangle `[0, L] × [−2L, 2L]` (width W = 4L):

```
u_x_bar(x, y) = v_web · sin(β) + γ/(3μ) · h(x)² · h‴(x)     [no y-dependence]
u_y_bar(x, y) = −v_web · cos(β)                                  [constant everywhere]
```

Steps:

1. Create a 2D meshgrid: `x_grid` with ~200 points on [0, L − ε], `y_grid` with ~200 points on [−2L, 2L].
2. Evaluate `u_x_bar` on the grid by interpolating h(x) and h‴(x) from the BVP solution (use `sol.sol` or `np.interp`).
3. `u_y_bar` is constant: fill the grid with `−v_web · cos(β)`.
4. Compute velocity magnitude: `speed = sqrt(u_x_bar² + u_y_bar²)`.
5. Plot using:
   - `ax.pcolormesh(X, Y, speed, cmap='viridis', shading='auto')` for the color background.
   - `ax.streamplot(X, Y, U_x, U_y, color='white', linewidth=0.8, arrowsize=1.0, density=1.5)` for streamlines on top. Ensure arrows are visible (white on viridis works well).
6. Add a colorbar labeled "Speed (cm/s)".
7. x-axis: "x (cm)", y-axis: "y (cm)"
8. Title: "Velocity Field Streamlines"
9. Use `set_aspect('equal')` so the rectangle isn't distorted.

**Note on meshgrid orientation for `streamplot`**: `matplotlib.streamplot` expects `x` as a 1D array (columns), `y` as a 1D array (rows), and `U`, `V` as 2D arrays with shape `(len(y), len(x))`. Make sure the arrays are oriented correctly — `U_x[j, i]` corresponds to position `(x[i], y[j])`.

---

## 4. Output

- Save all four plots as:
  - `plot1_film_thickness.png`
  - `plot2_pressure.png`
  - `plot3_x_velocity.png`
  - `plot4_streamplot.png`
- All at 300 dpi, with `bbox_inches='tight'`.

---

## 5. Code Structure

```
thin_film_solver.py
│
├── @dataclass FilmParams
│     h0, beta, v_web, v_evap, mu, gamma, L
│     (initialized with CGS values as above)
│
├── def ode_system(x, y, params) → dy/dx array
│
├── def boundary_conditions(ya, yb, params) → residuals array (4 elements)
│
├── def solve_bvp_problem(params) → (x_fine, h, h', h'', h''', sol object)
│
├── def plot_film_thickness(x, h, params)
├── def plot_pressure(x, h_tprime, params)
├── def plot_x_velocity(x, u_x_bar, params)
├── def plot_streamplot(x_fine, u_x_bar_1d, params)
│
└── if __name__ == "__main__":
      params = FilmParams()
      solve → extract arrays → compute derived quantities → plot all four
```

---

## 6. Potential Pitfalls

1. **Singularity at x = L**: The division by h(x) blows up where h(L) = 0. Offset the domain end to L − ε where ε = 1e-8 · L.
2. **solve_bvp convergence**: If it doesn't converge, try (a) a finer initial mesh (500+ points), (b) relaxed tolerance first then tightened, (c) `max_nodes=10000`.
3. **streamplot array orientation**: `streamplot` is fussy about shapes. Double-check that U and V have shape `(ny, nx)`.
4. **γ/(3μ) vs (γ/3)·μ**: The coefficient in the velocity and ODE is always γ divided by the product 3μ. This is a very common transcription error — be careful.
5. **β = 45 degrees**: Store as `np.radians(45)` ≈ 0.7854 radians. Do NOT use 45 directly in `np.sin()`.
