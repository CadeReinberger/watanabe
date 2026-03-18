"""
fit_params.py — MLE parameter fitting for the quiz bowl buzz model.

Parameters
----------
r[q]    : information rate for question q  (0, ∞)
lam[i]  : buzz rate for player i           (0, ∞)
eta[i]  : neg probability for player i     (0, 1)

Total params: question_world_size + 2 * player_world_size

See quizbowl_mle_spec.md for the full probability model.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid(x) = 1 / (1 + exp(-x))


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
# Bounce-back likelihood helper
# ---------------------------------------------------------------------------

def _log_p2_chosen(
    p2: int,
    team_y: list[int],
    log_k: dict[int, float],
    log_1mk: dict[int, float],
) -> float:
    """
    Log-probability that player p2 is chosen as the conversion player after a neg.

    Model: each player j ∈ team_y independently "knows" the answer with probability
    k[j]. One knower is chosen uniformly at random.

    P(p2 chosen) = Σ_{S ⊆ team_y, p2 ∈ S}  [∏_{j∈S} k[j] · ∏_{j∉S} (1-k[j])] / |S|

    Computed in log-space via logsumexp over all valid subsets.
    """
    others = [j for j in team_y if j != p2]
    log_terms: list[float] = []

    # S = {p2} ∪ T for every T ⊆ others
    for size_t in range(len(others) + 1):
        for T in itertools.combinations(others, size_t):
            T_set = set(T)
            S_size = 1 + size_t
            log_prob_S = (
                log_k[p2]
                + sum(log_k[j] for j in T)
                + sum(log_1mk[j] for j in others if j not in T_set)
                - np.log(S_size)
            )
            log_terms.append(log_prob_S)

    return np.logaddexp.reduce(np.array(log_terms))


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def _neg_log_likelihood(
    params: np.ndarray,
    question_world_size: int,
    player_world_size: int,
    gamerooms: list[GameRoom],
) -> float:
    """Negative log-likelihood from flat unconstrained parameter vector."""
    # --- Unpack and transform to constrained space ---
    log_r    = params[:question_world_size]
    log_lam  = params[question_world_size : question_world_size + player_world_size]
    logit_eta = params[question_world_size + player_world_size :]

    r   = np.exp(np.clip(log_r,   -20.0, 20.0))
    lam = np.exp(np.clip(log_lam, -20.0, 20.0))
    eta = expit(logit_eta)

    total_nll = 0.0

    for room in gamerooms:
        all_players = list(room.team_a | room.team_b)
        Lambda = lam[all_players].sum()

        for pq in room.played_questions:
            q      = pq.question_number
            rq     = r[q]
            buzzes = pq.buzzes

            # ------------------------------------------------------------------
            # Case 1: No buzzes
            # ------------------------------------------------------------------
            if len(buzzes) == 0:
                log_L = -Lambda * rq

            # ------------------------------------------------------------------
            # Case 2: One correct buzz
            # ------------------------------------------------------------------
            elif len(buzzes) == 1 and buzzes[0].correct:
                p = buzzes[0].player
                c = buzzes[0].celerity
                log_L = (
                    np.log(rq)
                    + np.log(lam[p])
                    - Lambda * rq * c
                    + np.log1p(-eta[p])
                )

            # ------------------------------------------------------------------
            # Case 3: One incorrect buzz (neg), no conversion
            # ------------------------------------------------------------------
            elif len(buzzes) == 1 and not buzzes[0].correct:
                p1 = buzzes[0].player
                c1 = buzzes[0].celerity
                team_x = room.team_a if p1 in room.team_a else room.team_b
                team_y = list((room.team_a | room.team_b) - team_x)

                # k[j] = (1 - eta[j]) * (1 - exp(-lam[j] * rq))
                lam_r  = np.clip(lam[team_y] * rq, 0.0, 500.0)
                k      = (1.0 - eta[team_y]) * (1.0 - np.exp(-lam_r))
                log_no_convert = np.sum(np.log(np.clip(1.0 - k, 1e-300, None)))

                log_L = (
                    np.log(rq)
                    + np.log(lam[p1])
                    - Lambda * rq * c1
                    + np.log(eta[p1])
                    + log_no_convert
                )

            # ------------------------------------------------------------------
            # Case 4: Neg then correct conversion
            # ------------------------------------------------------------------
            elif len(buzzes) == 2 and not buzzes[0].correct and buzzes[1].correct:
                p1 = buzzes[0].player
                c1 = buzzes[0].celerity
                p2 = buzzes[1].player
                team_x = room.team_a if p1 in room.team_a else room.team_b
                team_y = list((room.team_a | room.team_b) - team_x)

                lam_r = np.clip(lam[team_y] * rq, 0.0, 500.0)
                k     = (1.0 - eta[team_y]) * (1.0 - np.exp(-lam_r))

                log_k   = {j: np.log(max(k[idx], 1e-300))         for idx, j in enumerate(team_y)}
                log_1mk = {j: np.log(max(1.0 - k[idx], 1e-300))   for idx, j in enumerate(team_y)}

                log_p2 = _log_p2_chosen(p2, team_y, log_k, log_1mk)

                log_L = (
                    np.log(rq)
                    + np.log(lam[p1])
                    - Lambda * rq * c1
                    + np.log(eta[p1])
                    + log_p2
                )

            else:
                # Unrecognised buzz pattern — skip this question
                continue

            total_nll -= log_L

    return total_nll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_params(
    question_world_size: int,
    player_world_size: int,
    gamerooms: list[GameRoom],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit quiz bowl model parameters via MLE.

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
    """
    n_params = question_world_size + 2 * player_world_size

    # Initial values in unconstrained space:
    #   log_r = 0    → r   = 1.0
    #   log_lam = 0  → lam = 1.0
    #   logit_eta=-2 → eta ≈ 0.12

    x0 = np.zeros(n_params)
    x0[question_world_size + player_world_size:] = -2.0

    t0 = time.monotonic()
    iteration = [0]

    def _callback(xk: np.ndarray) -> None:
        iteration[0] += 1
        elapsed = time.monotonic() - t0
        nll = _neg_log_likelihood(xk, question_world_size, player_world_size, gamerooms)
        print(f"  iter {iteration[0]:4d}  nll={nll:14.4f}  elapsed={elapsed:6.1f}s", flush=True)

    print(
        f"Optimising {n_params} params "
        f"({question_world_size}q + {player_world_size}p×2) "
        f"over {len(gamerooms)} game rooms …",
        flush=True,
    )
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(question_world_size, player_world_size, gamerooms),
        method="L-BFGS-B",
        callback=_callback,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    elapsed_total = time.monotonic() - t0
    print(
        f"Done: {iteration[0]} iterations, "
        f"final nll={result.fun:.4f}, "
        f"success={result.success}, "
        f"total={elapsed_total:.1f}s",
        flush=True,
    )

    log_r    = result.x[:question_world_size]
    log_lam  = result.x[question_world_size : question_world_size + player_world_size]
    logit_eta = result.x[question_world_size + player_world_size :]

    r   = np.exp(np.clip(log_r,   -20.0, 20.0))
    lam = np.exp(np.clip(log_lam, -20.0, 20.0))
    eta = expit(logit_eta)

    return r, lam, eta


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

    p2   = rng.choice(knowers)
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

    n_rooms = 5_000
    gamerooms: list[GameRoom] = []
    for i in range(n_rooms):
        ta, tb = config_a if i % 2 == 0 else config_b
        room = GameRoom(team_a=set(ta), team_b=set(tb))
        for q in range(Q):
            pq = _simulate_question(room, q, r_true, lam_true, eta_true, rng)
            room.played_questions.append(pq)
        gamerooms.append(room)

    print(f"Fitting on {n_rooms} game rooms × {Q} questions each …")
    r_fit, lam_fit, eta_fit = fit_params(Q, P, gamerooms)

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

    # Soft assertions
    assert r_rmse   < 0.20, f"r recovery too poor: RMSE={r_rmse:.4f}"
    assert lam_rmse < 0.20, f"lam recovery too poor: RMSE={lam_rmse:.4f}"
    assert eta_rmse < 0.50, f"eta recovery too poor: RMSE={eta_rmse:.4f}"
    print("\nAll recovery checks passed.")


if __name__ == "__main__":
    _run_synthetic_test()
