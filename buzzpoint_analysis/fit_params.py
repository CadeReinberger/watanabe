"""
fit_params.py — MAP parameter fitting for the quiz bowl buzz model.

Parameters
----------
r[q]    : information rate for question q  (0, ∞)
lam[i]  : buzz rate for player i           (0, ∞)
eta[i]  : neg probability for player i     (0, 1)

Hyperparameters (empirical Bayes, jointly optimised)
----------------------------------------------------
sigma_r           : population prior spread for log(r[q])     ~ N(0,       sigma_r^2)
mu_lam, sigma_lam : population prior for log(lam[i])          ~ N(mu_lam,  sigma_lam^2)
mu_eta, sigma_eta : population prior for logit(eta[i])        ~ N(mu_eta,  sigma_eta^2)

Note: mu_r is fixed to 0 to anchor the scale between r and lambda.
mu_lam absorbs the overall scale.

Total params: 2 * player_world_size + question_world_size + 5

See quizbowl_mle_spec.md for the full probability model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# JAX configuration — must be set before the first jax import.
# Requires Python ≤ 3.12 (jaxlib 0.9.x wheel is cp312).
# Run with:  conda run -n base python fit_params.py
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import jax
    jax.config.update("jax_enable_x64", True)   # keep float64 to match scipy precision
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError as _jax_err:
    raise ImportError(
        "JAX is required but could not be imported. "
        "Install it with:  pip install jax\n"
        "If you are on Python 3.13, JAX wheels are not yet available; "
        "run with Python 3.12, e.g.:  conda run -n base python fit_params.py\n"
        f"Original error: {_jax_err}"
    ) from _jax_err

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid, used only for final output conversion


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Buzz:
    player: int       # player index in {0, ..., player_world_size - 1}
    celerity: float   # fraction of question read at buzz time, in (0, 1]
    correct: bool     # whether the player answered correctly


@dataclass
class PlayedQuestion:
    question_number: int          # question index in {0, ..., question_world_size - 1}
    buzzes: list[Buzz] = field(default_factory=list)  # 0, 1, or 2 elements


@dataclass
class GameRoom:
    team_a: set[int]                          # player indices on team A
    team_b: set[int]                          # player indices on team B
    played_questions: list[PlayedQuestion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preprocessing: gamerooms → batched numpy arrays grouped by case type
# ---------------------------------------------------------------------------

def _make_subset_masks(ty_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all 2^(ty_size-1) subsets of {0,...,ty_size-1} that include index 0.

    In the neg-then-convert case, p2 is placed at index 0 of the team_y array,
    so these are exactly the subsets containing p2.

    Returns
    -------
    masks     : float64 (n_subsets, ty_size)  — 1.0 if member j is in subset s
    log_sizes : float64 (n_subsets,)          — log of |subset s|
    """
    k_others = ty_size - 1
    n_subsets = 2 ** k_others
    masks = np.zeros((n_subsets, ty_size), dtype=np.float64)
    masks[:, 0] = 1.0   # p2 always included
    for b in range(k_others):
        for s in range(n_subsets):
            if s & (1 << b):
                masks[s, b + 1] = 1.0
    log_sizes = np.log(masks.sum(axis=1))
    return masks, log_sizes


def _preprocess_gamerooms(gamerooms: list[GameRoom]) -> dict:
    """
    Convert a list of GameRooms into fixed-shape numpy arrays grouped by the
    four likelihood cases:

      1. No buzz
      2. One correct buzz
      3. Neg, no conversion by opposing team
      4. Neg then correct conversion

    Variable-length fields (room player lists, team_y) are padded to a fixed
    maximum width; padding positions are indicated by boolean mask arrays.

    Returns a dict containing all arrays needed by ``_build_jax_nll``.
    """
    num_rooms = len(gamerooms)

    # --- Determine padding widths from data ---
    max_room_size = max(len(r.team_a) + len(r.team_b) for r in gamerooms)

    max_ty_size = 0
    for room in gamerooms:
        for pq in room.played_questions:
            if pq.buzzes and not pq.buzzes[0].correct:
                p1 = pq.buzzes[0].player
                team_x = room.team_a if p1 in room.team_a else room.team_b
                ty = len((room.team_a | room.team_b) - team_x)
                if ty > max_ty_size:
                    max_ty_size = ty
    max_ty_size = max(max_ty_size, 1)   # at least 1 to avoid degenerate shapes

    # --- Room-level arrays: all player indices per room (padded) ---
    room_players  = np.zeros((num_rooms, max_room_size), dtype=np.int32)
    room_ply_mask = np.zeros((num_rooms, max_room_size), dtype=bool)

    for r_idx, room in enumerate(gamerooms):
        all_p = sorted(room.team_a | room.team_b)
        n = len(all_p)
        room_players[r_idx, :n] = all_p
        room_ply_mask[r_idx, :n] = True

    # --- Observation-level accumulators per case ---
    c1 = dict(q=[], room=[])
    c2 = dict(q=[], room=[], p=[], c=[])
    c3 = dict(q=[], room=[], p1=[], c1=[], ty=[], ty_mask=[])
    c4 = dict(q=[], room=[], p1=[], c1=[], ty=[], ty_mask=[])   # ty[0] = p2

    for r_idx, room in enumerate(gamerooms):
        for pq in room.played_questions:
            q      = pq.question_number
            buzzes = pq.buzzes

            if len(buzzes) == 0:
                c1["q"].append(q);   c1["room"].append(r_idx)

            elif len(buzzes) == 1 and buzzes[0].correct:
                c2["q"].append(q);   c2["room"].append(r_idx)
                c2["p"].append(buzzes[0].player)
                c2["c"].append(buzzes[0].celerity)

            elif len(buzzes) == 1 and not buzzes[0].correct:
                p1 = buzzes[0].player
                team_x = room.team_a if p1 in room.team_a else room.team_b
                team_y = list((room.team_a | room.team_b) - team_x)
                sz     = len(team_y)
                c3["q"].append(q);   c3["room"].append(r_idx)
                c3["p1"].append(p1); c3["c1"].append(buzzes[0].celerity)
                c3["ty"].append(team_y + [0] * (max_ty_size - sz))
                c3["ty_mask"].append([True] * sz + [False] * (max_ty_size - sz))

            elif len(buzzes) == 2 and not buzzes[0].correct and buzzes[1].correct:
                p1 = buzzes[0].player
                p2 = buzzes[1].player
                team_x = room.team_a if p1 in room.team_a else room.team_b
                team_y = list((room.team_a | room.team_b) - team_x)
                sz     = len(team_y)
                # Place p2 at index 0 for the subset-mask enumeration
                ty_reordered = [p2] + [j for j in team_y if j != p2]
                c4["q"].append(q);   c4["room"].append(r_idx)
                c4["p1"].append(p1); c4["c1"].append(buzzes[0].celerity)
                c4["ty"].append(ty_reordered + [0] * (max_ty_size - sz))
                c4["ty_mask"].append([True] * sz + [False] * (max_ty_size - sz))

    def _arr1(lst, dtype=np.int32):
        return np.array(lst, dtype=dtype) if lst else np.zeros(0, dtype=dtype)

    def _arr2(lst, dtype=np.int32):
        return np.array(lst, dtype=dtype) if lst else np.zeros((0, max_ty_size), dtype=dtype)

    return {
        "max_ty_size":   max_ty_size,
        "room_players":  room_players,
        "room_ply_mask": room_ply_mask,
        # case 1
        "c1_q":    _arr1(c1["q"]),    "c1_room": _arr1(c1["room"]),
        # case 2
        "c2_q":    _arr1(c2["q"]),    "c2_room": _arr1(c2["room"]),
        "c2_p":    _arr1(c2["p"]),    "c2_c":    _arr1(c2["c"],  np.float64),
        # case 3
        "c3_q":    _arr1(c3["q"]),    "c3_room": _arr1(c3["room"]),
        "c3_p1":   _arr1(c3["p1"]),   "c3_c1":   _arr1(c3["c1"], np.float64),
        "c3_ty":   _arr2(c3["ty"]),   "c3_ty_m": _arr2(c3["ty_mask"], bool),
        # case 4
        "c4_q":    _arr1(c4["q"]),    "c4_room": _arr1(c4["room"]),
        "c4_p1":   _arr1(c4["p1"]),   "c4_c1":   _arr1(c4["c1"], np.float64),
        "c4_ty":   _arr2(c4["ty"]),   "c4_ty_m": _arr2(c4["ty_mask"], bool),
    }


# ---------------------------------------------------------------------------
# JAX NLL + gradient
# ---------------------------------------------------------------------------

def _build_jax_nll(data: dict, Q: int, P: int):
    """
    Build a JIT-compiled ``value_and_grad`` function for the negative log-posterior.

    The preprocessing arrays are closed over as compile-time constants so that
    JAX does not need to retrace them on every optimizer step.

    Parameters
    ----------
    data : dict
        Output of ``_preprocess_gamerooms``.
    Q, P : int
        question_world_size, player_world_size.

    Returns
    -------
    value_and_grad_np : callable
        ``value_and_grad_np(x: np.ndarray) -> (float, np.ndarray)``
        Returns the NLL and its gradient, both as plain numpy scalars/arrays.
    """
    # Convert all index / mask arrays to JAX constants.
    room_players  = jnp.array(data["room_players"])    # (R, max_room)
    room_ply_mask = jnp.array(data["room_ply_mask"])   # (R, max_room) bool

    c1_q    = jnp.array(data["c1_q"]);    c1_room = jnp.array(data["c1_room"])
    c2_q    = jnp.array(data["c2_q"]);    c2_room = jnp.array(data["c2_room"])
    c2_p    = jnp.array(data["c2_p"]);    c2_c    = jnp.array(data["c2_c"])
    c3_q    = jnp.array(data["c3_q"]);    c3_room = jnp.array(data["c3_room"])
    c3_p1   = jnp.array(data["c3_p1"]);   c3_c1   = jnp.array(data["c3_c1"])
    c3_ty   = jnp.array(data["c3_ty"]);   c3_ty_m = jnp.array(data["c3_ty_m"])
    c4_q    = jnp.array(data["c4_q"]);    c4_room = jnp.array(data["c4_room"])
    c4_p1   = jnp.array(data["c4_p1"]);   c4_c1   = jnp.array(data["c4_c1"])
    c4_ty   = jnp.array(data["c4_ty"]);   c4_ty_m = jnp.array(data["c4_ty_m"])

    # Subset enumeration for case 4.
    # p2 is at team_y index 0; enumerate all 2^(max_ty-1) subsets containing 0.
    max_ty = data["max_ty_size"]
    smasks_np, slsizes_np = _make_subset_masks(max_ty)
    smasks    = jnp.array(smasks_np)           # (n_subsets, max_ty)
    not_smasks = 1.0 - smasks                  # (n_subsets, max_ty)
    slsizes   = jnp.array(slsizes_np)          # (n_subsets,)  log of subset sizes

    def _nll_raw(params: jnp.ndarray) -> jnp.ndarray:
        """Vectorised negative log-posterior (no Python loops over observations)."""
        log_r      = params[:Q]
        log_lam    = params[Q : Q + P]
        logit_eta  = params[Q + P : Q + 2 * P]
        log_sig_r  = params[Q + 2 * P]
        mu_lam     = params[Q + 2 * P + 1]
        log_sig_lm = params[Q + 2 * P + 2]
        mu_eta     = params[Q + 2 * P + 3]
        log_sig_et = params[Q + 2 * P + 4]

        sig_r  = jnp.exp(log_sig_r)
        sig_lm = jnp.exp(log_sig_lm)
        sig_et = jnp.exp(log_sig_et)

        # Constrained parameters
        r   = jnp.exp(jnp.clip(log_r,   -20.0, 20.0))
        lam = jnp.exp(jnp.clip(log_lam, -20.0, 20.0))
        # log(eta) and log(1-eta) via numerically-stable log-sigmoid
        log_eta     = jax.nn.log_sigmoid(logit_eta)    # log p(neg)
        log_1m_eta  = jax.nn.log_sigmoid(-logit_eta)   # log p(correct)

        # MAP penalties: -log N(x|mu, sigma^2) summed over all individual params
        nll = (
            jnp.sum(log_r ** 2)                       / (2.0 * sig_r  ** 2) + Q * log_sig_r
            + jnp.sum((log_lam  - mu_lam) ** 2)       / (2.0 * sig_lm ** 2) + P * log_sig_lm
            + jnp.sum((logit_eta - mu_eta) ** 2)      / (2.0 * sig_et ** 2) + P * log_sig_et
        )

        # Per-room Lambda = sum of buzz rates over all players in the room
        room_lam = jnp.where(room_ply_mask, lam[room_players], 0.0)  # (R, max_room)
        Lambda   = room_lam.sum(axis=1)                               # (R,)

        # ------------------------------------------------------------------
        # Case 1: no buzz — log_L = -Lambda * r[q]
        # ------------------------------------------------------------------
        nll = nll + jnp.sum(Lambda[c1_room] * r[c1_q])

        # ------------------------------------------------------------------
        # Case 2: one correct buzz
        # log_L = log(r[q]) + log(lam[p]) - Lambda*r[q]*c + log(1-eta[p])
        # ------------------------------------------------------------------
        rq2   = r[c2_q]
        lr2   = jnp.clip(log_r[c2_q],   -20.0, 20.0)
        llm2  = jnp.clip(log_lam[c2_p], -20.0, 20.0)
        nll   = nll - jnp.sum(lr2 + llm2 - Lambda[c2_room] * rq2 * c2_c + log_1m_eta[c2_p])

        # ------------------------------------------------------------------
        # Case 3: neg, no conversion
        # log_L = log(r[q]) + log(lam[p1]) - Lambda*r[q]*c1
        #         + log(eta[p1]) + sum_j log(1 - k[j])
        # where k[j] = (1-eta[j]) * (1 - exp(-lam[j]*r[q]))
        # ------------------------------------------------------------------
        rq3  = r[c3_q]                                       # (N3,)
        lam_ty3 = lam[c3_ty]                                 # (N3, max_ty)
        eta_ty3 = jax.nn.sigmoid(logit_eta[c3_ty])          # (N3, max_ty)
        k3      = (1.0 - eta_ty3) * (-jnp.expm1(            # (N3, max_ty)
                      -jnp.clip(lam_ty3 * rq3[:, None], 0.0, 500.0)))
        k3      = jnp.where(c3_ty_m, k3, 0.0)               # zero out padding
        # sum_j log(1-k[j])  — padding positions contribute 0 since k=0 there
        log_1mk3 = jnp.where(c3_ty_m,
                              jnp.log(jnp.clip(1.0 - k3, 1e-300, None)),
                              0.0)
        nll = nll - jnp.sum(
            jnp.clip(log_r[c3_q], -20.0, 20.0)
            + jnp.clip(log_lam[c3_p1], -20.0, 20.0)
            - Lambda[c3_room] * rq3 * c3_c1
            + log_eta[c3_p1]
            + log_1mk3.sum(axis=1)
        )

        # ------------------------------------------------------------------
        # Case 4: neg then correct conversion
        # log_L = log(r[q]) + log(lam[p1]) - Lambda*r[q]*c1
        #         + log(eta[p1]) + log_p2_chosen(p2, team_y, k)
        #
        # log_p2_chosen is vectorised via pre-enumerated subset masks:
        #   log_prob_S[n,s] = Σ_i mask[s,i]*log_k[n,i]
        #                   + Σ_i (1-mask[s,i])*log_1mk[n,i]
        #                   - log|S[s]|
        #   log_p2[n] = logsumexp_s log_prob_S[n,s]
        #
        # Subsets containing a padded position (k→0) contribute −∞ via
        # log_k=-1e38 and are numerically zeroed by logsumexp.
        # ------------------------------------------------------------------
        rq4     = r[c4_q]                                    # (N4,)
        lam_ty4 = lam[c4_ty]                                 # (N4, max_ty)
        eta_ty4 = jax.nn.sigmoid(logit_eta[c4_ty])          # (N4, max_ty)
        k4      = (1.0 - eta_ty4) * (-jnp.expm1(
                      -jnp.clip(lam_ty4 * rq4[:, None], 0.0, 500.0)))
        k4      = jnp.where(c4_ty_m, k4, 0.0)               # zero out padding

        log_k4   = jnp.where(c4_ty_m,
                              jnp.log(jnp.clip(k4, 1e-300, None)),
                              -1e38)                         # (N4, max_ty)
        log_1mk4 = jnp.where(c4_ty_m,
                              jnp.log(jnp.clip(1.0 - k4, 1e-300, None)),
                              0.0)                           # (N4, max_ty)

        # (N4, n_subsets): log P(S) for each observation and each subset
        log_prob_S = (
            jnp.einsum("si,ni->ns", smasks,     log_k4)
            + jnp.einsum("si,ni->ns", not_smasks, log_1mk4)
            - slsizes[None, :]
        )
        log_p2 = jax.scipy.special.logsumexp(log_prob_S, axis=1)  # (N4,)

        nll = nll - jnp.sum(
            jnp.clip(log_r[c4_q], -20.0, 20.0)
            + jnp.clip(log_lam[c4_p1], -20.0, 20.0)
            - Lambda[c4_room] * rq4 * c4_c1
            + log_eta[c4_p1]
            + log_p2
        )

        return nll

    # Compile a combined value+gradient function once.
    _vg = jax.jit(jax.value_and_grad(_nll_raw))

    def value_and_grad_np(x: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = _vg(jnp.array(x))
        return float(val), np.array(grad, dtype=np.float64)

    return value_and_grad_np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_params(
    question_world_size: int,
    player_world_size: int,
    gamerooms: list[GameRoom],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Fit quiz bowl model parameters via joint MAP with empirical Bayes.

    Parameters
    ----------
    question_world_size : int
        Total number of distinct questions (indices 0..question_world_size-1).
        Questions never played in any room have unidentified r[q]; they are
        returned at their initial value (r[q] = 1.0).
    player_world_size : int
        Total number of distinct players (indices 0..player_world_size-1).
    gamerooms : list[GameRoom]
        Observed game data. Player parameters are shared across all rooms.

    Returns
    -------
    r   : ndarray shape (question_world_size,)  — fitted information rates
    lam : ndarray shape (player_world_size,)    — fitted buzz rates
    eta : ndarray shape (player_world_size,)    — fitted neg probabilities
    hyperparams : dict with keys
        'mu_r', 'sigma_r'     — fitted prior for log(r[q])
        'mu_lam', 'sigma_lam' — fitted prior for log(lam[i])
        'mu_eta', 'sigma_eta' — fitted prior for logit(eta[i])
    """
    Q, P = question_world_size, player_world_size
    n_params = Q + 2 * P + 5

    # Initial values in unconstrained space:
    #   log_r = 0    → r   = 1.0
    #   log_lam = 0  → lam = 1.0
    #   logit_eta=-2 → eta ≈ 0.12
    #   mu_r fixed at 0, mu_lam=0, mu_eta=-2  (match individual param inits)
    #   log_sigma_r=0, log_sigma_lam=0, log_sigma_eta=0  → sigma = 1.0
    x0 = np.zeros(n_params)
    x0[Q + P : Q + 2 * P] = -2.0          # logit_eta initialised to -2
    x0[Q + 2 * P + 3] = -2.0              # mu_eta = -2

    # --- Preprocess gamerooms into batched arrays (Python, one-time cost) ---
    print("  Preprocessing gamerooms ...", flush=True)
    data = _preprocess_gamerooms(gamerooms)
    obs_counts = {
        "no_buzz":    data["c1_q"].shape[0],
        "correct":    data["c2_q"].shape[0],
        "neg_nc":     data["c3_q"].shape[0],
        "neg_conv":   data["c4_q"].shape[0],
    }
    print(f"  Observations by case: {obs_counts}", flush=True)

    # --- Build JIT-compiled NLL + gradient ---
    print("  Building JAX NLL (first call triggers JIT compilation) ...", flush=True)
    value_and_grad_np = _build_jax_nll(data, Q, P)

    # --- Progress tracking state ---
    t0 = time.monotonic()
    iteration    = [0]
    last_nll     = [float("nan")]
    nll_history: list[float] = []
    _FTOL = 1e-12   # must match options below

    def _objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = value_and_grad_np(x)
        last_nll[0] = val
        return val, grad

    def _callback(xk: np.ndarray) -> None:
        iteration[0] += 1
        elapsed = time.monotonic() - t0
        nll = last_nll[0]
        nll_history.append(nll)

        # Per-iteration improvement
        if len(nll_history) >= 2:
            delta = nll_history[-2] - nll_history[-1]
        else:
            delta = float("nan")
        delta_str = f"{delta:+.3e}" if np.isfinite(delta) else "       n/a"

        # ETA via log-linear extrapolation of recent NLL improvements
        target_delta = _FTOL * max(abs(nll), 1.0)
        eta_str = "     ?"
        pct = 0.0

        if len(nll_history) >= 4:
            window     = nll_history[max(0, len(nll_history) - 25):]
            raw_deltas = [window[i - 1] - window[i] for i in range(1, len(window))]
            pos        = [(i, d) for i, d in enumerate(raw_deltas) if d > 0]

            if len(pos) >= 3 and np.isfinite(delta) and delta > 0:
                xs = np.array([p[0] for p in pos], dtype=float)
                ys = np.log(np.array([p[1] for p in pos]))
                slope, _ = np.polyfit(xs, ys, 1)

                if slope < 0:
                    log_cur = np.log(max(delta, 1e-300))
                    log_tgt = np.log(max(target_delta, 1e-300))
                    remaining_iters  = max(0.0, (log_tgt - log_cur) / slope)
                    eta_secs = remaining_iters * (elapsed / iteration[0])
                    if eta_secs < 60:
                        eta_str = f"{eta_secs:5.0f}s"
                    elif eta_secs < 3600:
                        eta_str = f"{eta_secs / 60:5.1f}m"
                    else:
                        eta_str = f"{eta_secs / 3600:5.1f}h"

                    first_pos = next((d for d in raw_deltas if d > 0), None)
                    if first_pos and first_pos > target_delta:
                        pct = float(np.clip(
                            (np.log(first_pos) - log_cur) / (np.log(first_pos) - log_tgt),
                            0.0, 1.0,
                        ))

        bar_w  = 25
        filled = int(pct * bar_w)
        bar    = "█" * filled + "░" * (bar_w - filled)

        print(
            f"  iter {iteration[0]:4d} │ nll={nll:13.3f} │ Δ={delta_str}"
            f" │ {elapsed:7.1f}s │ ETA≈{eta_str} │ [{bar}] {pct * 100:5.1f}%",
            flush=True,
        )

    print(
        f"Optimising {n_params} params "
        f"({Q}q + {P}p×2 + 5 hyperparams) "
        f"over {len(gamerooms)} game rooms …",
        flush=True,
    )
    result = minimize(
        _objective,
        x0,
        method="L-BFGS-B",
        jac=True,              # _objective returns (value, gradient) together
        callback=_callback,
        options={"maxiter": 2000, "ftol": _FTOL, "gtol": 1e-8},
    )
    elapsed_total = time.monotonic() - t0
    print(
        f"Done: {iteration[0]} iterations, "
        f"final nll={result.fun:.4f}, "
        f"success={result.success}, "
        f"total={elapsed_total:.1f}s",
        flush=True,
    )

    log_r     = result.x[:Q]
    log_lam   = result.x[Q : Q + P]
    logit_eta = result.x[Q + P : Q + 2 * P]

    log_sigma_r   = result.x[Q + 2 * P]
    mu_lam,    log_sigma_lam = result.x[Q + 2 * P + 1], result.x[Q + 2 * P + 2]
    mu_eta,    log_sigma_eta = result.x[Q + 2 * P + 3], result.x[Q + 2 * P + 4]

    # Return plain numpy arrays
    r   = np.array(np.exp(np.clip(log_r,   -20.0, 20.0)))
    lam = np.array(np.exp(np.clip(log_lam, -20.0, 20.0)))
    eta = np.array(expit(logit_eta))

    hyperparams = {
        "mu_r":      0.0,
        "sigma_r":   float(np.exp(log_sigma_r)),
        "mu_lam":    float(mu_lam),
        "sigma_lam": float(np.exp(log_sigma_lam)),
        "mu_eta":    float(mu_eta),
        "sigma_eta": float(np.exp(log_sigma_eta)),
    }

    return r, lam, eta, hyperparams


# ---------------------------------------------------------------------------
# Synthetic data simulation (for testing)
# ---------------------------------------------------------------------------

def _simulate_question(
    room: GameRoom,
    q: int,
    r_true: np.ndarray,
    lam_true: np.ndarray,
    eta_true: np.ndarray,
    rng: np.random.Generator,
) -> PlayedQuestion:
    """Simulate one played question under the true model."""
    all_players = list(room.team_a | room.team_b)
    rq = r_true[q]

    # Sample each player's buzz time in information space ~ Exp(lam[i])
    buzz_times = {p: rng.exponential(1.0 / lam_true[p]) for p in all_players}
    first = min(all_players, key=lambda p: buzz_times[p])
    first_time = buzz_times[first]

    if first_time >= rq:
        return PlayedQuestion(question_number=q)

    celerity = first_time / rq
    correct  = rng.random() > eta_true[first]
    buzz1    = Buzz(player=first, celerity=celerity, correct=correct)

    if correct:
        return PlayedQuestion(question_number=q, buzzes=[buzz1])

    # Neg: bounce-back from opposing team
    team_x = room.team_a if first in room.team_a else room.team_b
    team_y = list((room.team_a | room.team_b) - team_x)

    knowers = [
        j for j in team_y
        if rng.random() < (1.0 - eta_true[j]) * (1.0 - np.exp(-lam_true[j] * rq))
    ]

    if not knowers:
        return PlayedQuestion(question_number=q, buzzes=[buzz1])

    p2    = rng.choice(knowers)
    buzz2 = Buzz(player=p2, celerity=float("nan"), correct=True)
    return PlayedQuestion(question_number=q, buzzes=[buzz1, buzz2])


def _run_synthetic_test() -> None:
    """
    Generate data from known parameters, fit, and report recovery error.

    World: 6 questions, 6 players split into two 3-player teams.
    We run many game rooms (each room replays all questions once) so the
    optimizer has enough data to recover the truth.
    """
    rng = np.random.default_rng(50)

    Q = 6   # question_world_size
    P = 6   # player_world_size

    r_true   = np.array([0.5, 1.0, 1.5, 2.0, 0.8, 1.2])
    lam_true = np.array([0.5, 1.0, 2.0, 0.8, 1.5, 0.3])
    eta_true = np.array([0.05, 0.15, 0.20, 0.10, 0.25, 0.08])

    # Two fixed team configurations; many rooms = many independent observations
    config_a = (frozenset({0, 1, 2}), frozenset({3, 4, 5}))
    config_b = (frozenset({0, 2, 4}), frozenset({1, 3, 5}))

    n_rooms = 300
    gamerooms: list[GameRoom] = []
    for i in range(n_rooms):
        ta, tb = config_a if i % 2 == 0 else config_b
        room = GameRoom(team_a=set(ta), team_b=set(tb))
        for q in range(Q):
            pq = _simulate_question(room, q, r_true, lam_true, eta_true, rng)
            room.played_questions.append(pq)
        gamerooms.append(room)

    print(f"Fitting on {n_rooms} game rooms × {Q} questions each …")
    r_fit, lam_fit, eta_fit, hyp = fit_params(Q, P, gamerooms)

    # Report recovery
    print("\n--- Recovery check ---")
    print(f"{'':12s}  {'true':>8s}  {'fitted':>8s}  {'rel err':>8s}")
    print("-" * 45)

    for q in range(Q):
        rel = abs(r_fit[q] - r_true[q]) / r_true[q]
        print(f"r[{q}]        {r_true[q]:8.3f}  {r_fit[q]:8.3f}  {rel:8.3f}")

    for i in range(P):
        rel = abs(lam_fit[i] - lam_true[i]) / lam_true[i]
        print(f"lam[{i}]      {lam_true[i]:8.3f}  {lam_fit[i]:8.3f}  {rel:8.3f}")

    for i in range(P):
        rel = abs(eta_fit[i] - eta_true[i]) / max(eta_true[i], 1e-6)
        print(f"eta[{i}]      {eta_true[i]:8.3f}  {eta_fit[i]:8.3f}  {rel:8.3f}")

    r_rmse   = np.sqrt(np.mean(((r_fit   - r_true)   / r_true)   ** 2))
    lam_rmse = np.sqrt(np.mean(((lam_fit - lam_true) / lam_true) ** 2))
    eta_rmse = np.sqrt(np.mean(((eta_fit - eta_true) / eta_true) ** 2))

    print(f"\nRelative RMSE:  r={r_rmse:.4f}  lam={lam_rmse:.4f}  eta={eta_rmse:.4f}")

    print("\n--- Fitted hyperparameters ---")
    for k, v in hyp.items():
        print(f"  {k:12s} = {v:.4f}")

    # Soft assertions
    assert r_rmse   < 0.20, f"r recovery too poor: RMSE={r_rmse:.4f}"
    assert lam_rmse < 0.20, f"lam recovery too poor: RMSE={lam_rmse:.4f}"
    assert eta_rmse < 0.50, f"eta recovery too poor: RMSE={eta_rmse:.4f}"
    print("\nAll recovery checks passed.")


if __name__ == "__main__":
    _run_synthetic_test()
