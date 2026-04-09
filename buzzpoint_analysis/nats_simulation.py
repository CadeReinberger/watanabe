"""
nats_simulation.py — Simulates ACF Nationals and evaluates parameter recovery.

Runs N_TRIALS full tournament simulations using the probabilistic buzz model
from fit_params.py, fits parameters after each trial, and saves 6 diagnostic
plots to nats_sims_results/.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit

from fit_params import fit_params, GameRoom, PlayedQuestion, Buzz


# ---------------------------------------------------------------------------
# Constants (hyperparameters and tournament structure — easy to change)
# ---------------------------------------------------------------------------

# Population-level hyperparameters (heuristic calibration)
MU_R = 0.0           # log-space mean for question information rate (fixed, anchors scale)
SIGMA_R = 0.5        # log-space std for question information rate
MU_LAM = -1.2115     # log-space mean for player buzz rate
SIGMA_LAM = 0.6026   # log-space std for player buzz rate
MU_ETA = -1.0986     # logit-space mean for neg probability
SIGMA_ETA = 1 #0.3914   # logit-space std for neg probability

# Tournament structure
NUM_TEAMS = 48
PLAYERS_PER_TEAM = 4
NUM_PLAYERS = NUM_TEAMS * PLAYERS_PER_TEAM  # 192
NUM_PRELIM_PACKETS = 7
NUM_PLAYOFF_PACKETS = 10
QUESTIONS_PER_PACKET = 20
TOTAL_QUESTIONS = (NUM_PRELIM_PACKETS + NUM_PLAYOFF_PACKETS) * QUESTIONS_PER_PACKET  # 340

NUM_PRELIM_BRACKETS = 6
TEAMS_PER_PRELIM_BRACKET = 8   # = NUM_TEAMS / NUM_PRELIM_BRACKETS
NUM_PLAYOFF_BRACKETS = 4
TEAMS_PER_PLAYOFF_BRACKET = 12  # = NUM_TEAMS / NUM_PLAYOFF_BRACKETS

# Number of simulation trials (each trial fits ~724 params via optimization — may be slow)
N_TRIALS = 500

# Output directory for plots
OUTPUT_DIR = "nats_sims_results"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_players(num_players: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample lam and eta for each player from their population priors."""
    log_lam = rng.normal(MU_LAM, SIGMA_LAM, size=num_players)
    logit_eta = rng.normal(MU_ETA, SIGMA_ETA, size=num_players)
    return np.exp(log_lam), expit(logit_eta)


def sample_questions(num_questions: int, rng: np.random.Generator) -> np.ndarray:
    """Sample r for each question from the population prior."""
    log_r = rng.normal(MU_R, SIGMA_R, size=num_questions)
    return np.exp(log_r)


# ---------------------------------------------------------------------------
# Team strength (for pre-ranking and seeding)
# ---------------------------------------------------------------------------

def compute_team_strength(
    team_players: list[int],
    lam_true: np.ndarray,
    eta_true: np.ndarray,
) -> float:
    """
    Approximate probability that this team converts a tossup against an average
    opposing team on an average question. Used for seeding only; closed-form.

    P(team buzzes first) = team_rate / (team_rate + opponent_rate)
    P(team converts)     ≈ P(team buzzes first) * (1 - avg_eta_of_team)
    """
    lam_avg = np.exp(MU_LAM)
    team_rate = sum(lam_true[p] for p in team_players)
    opponent_rate = PLAYERS_PER_TEAM * lam_avg
    total_rate = team_rate + opponent_rate
    p_buzz_first = team_rate / total_rate
    avg_eta = float(np.mean([eta_true[p] for p in team_players]))
    return p_buzz_first * (1.0 - avg_eta)


# ---------------------------------------------------------------------------
# Tossup simulation
# ---------------------------------------------------------------------------

def simulate_tossup(
    team_a: set[int],
    team_b: set[int],
    question_number: int,
    r_true: np.ndarray,
    lam_true: np.ndarray,
    eta_true: np.ndarray,
    rng: np.random.Generator,
) -> PlayedQuestion:
    """Simulate one tossup. Returns a PlayedQuestion with 0, 1, or 2 buzzes."""
    all_players = list(team_a | team_b)
    rq = r_true[question_number]

    # Draw buzz times for all players in information-space: T_i ~ Exp(rate=lam_i)
    buzz_times = {p: rng.exponential(1.0 / lam_true[p]) for p in all_players}

    # First buzzer
    p1 = min(all_players, key=lambda p: buzz_times[p])
    t1 = buzz_times[p1]

    if t1 >= rq:
        # Nobody buzzes before the question is fully read
        return PlayedQuestion(question_number=question_number, buzzes=[])

    c1 = t1 / rq
    correct1 = rng.random() > eta_true[p1]  # True with probability 1 - eta[p1]

    if correct1:
        return PlayedQuestion(question_number=question_number, buzzes=[Buzz(p1, c1, True)])

    # Neg: opposing team gets a bounce-back opportunity.
    # Player j converts if they would have buzzed before rq AND they know the answer.
    team_x = team_a if p1 in team_a else team_b
    team_y = list((team_a | team_b) - team_x)

    converters = [
        j for j in team_y
        if buzz_times[j] < rq and rng.random() > eta_true[j]
    ]

    if not converters:
        return PlayedQuestion(question_number=question_number, buzzes=[Buzz(p1, c1, False)])

    p2 = rng.choice(converters)
    c2 = buzz_times[p2] / rq  # recorded but not used by the fitter
    return PlayedQuestion(
        question_number=question_number,
        buzzes=[Buzz(p1, c1, False), Buzz(p2, c2, True)],
    )


# ---------------------------------------------------------------------------
# Game simulation
# ---------------------------------------------------------------------------

def simulate_game(
    team_a: set[int],
    team_b: set[int],
    packet_questions: list[int],
    r_true: np.ndarray,
    lam_true: np.ndarray,
    eta_true: np.ndarray,
    rng: np.random.Generator,
    prerank_a: int,
    prerank_b: int,
) -> tuple[GameRoom, bool]:
    """
    Simulate one game (one packet of 20 tossups).
    Returns (GameRoom, team_a_wins).
    Tie-breaks by pre-rank: lower index = stronger team = wins ties.
    """
    played: list[PlayedQuestion] = []
    score_a = 0
    score_b = 0

    for q in packet_questions:
        pq = simulate_tossup(team_a, team_b, q, r_true, lam_true, eta_true, rng)
        played.append(pq)
        if pq.buzzes:
            last = pq.buzzes[-1]
            if last.correct:
                if last.player in team_a:
                    score_a += 1
                else:
                    score_b += 1

    room = GameRoom(team_a=set(team_a), team_b=set(team_b), played_questions=played)

    if score_a > score_b:
        team_a_wins = True
    elif score_b > score_a:
        team_a_wins = False
    else:
        team_a_wins = prerank_a < prerank_b  # lower rank = stronger

    return room, team_a_wins


# ---------------------------------------------------------------------------
# Round-robin schedule (circle method)
# ---------------------------------------------------------------------------

def generate_round_robin_schedule(num_teams: int) -> list[list[tuple[int, int]]]:
    """
    Generate a complete round-robin schedule for num_teams (must be even).
    Uses the standard circle method: fix team 0, rotate teams 1..num_teams-1.

    Returns a list of (num_teams - 1) rounds.  Each round is a list of
    (num_teams / 2) matchups, each matchup being a (team_a, team_b) pair
    using 0-indexed local team labels.

    For 12 teams, round 0 pairs are: (0,1), (2,11), (3,10), (4,9), (5,8), (6,7).
    This is exploited by the playoff scheduler to align pre-played pairs with
    round 0 so that round 0 can be dropped cleanly.
    """
    assert num_teams % 2 == 0, "num_teams must be even"
    rotating = list(range(1, num_teams))
    n = len(rotating)  # num_teams - 1 rounds total
    rounds: list[list[tuple[int, int]]] = []
    for k in range(n):
        rotated = [rotating[(k + i) % n] for i in range(n)]
        # Fixed team 0 plays rotated[0]; remaining teams pair symmetrically
        games: list[tuple[int, int]] = [(0, rotated[0])]
        for i in range(1, n // 2 + 1):
            games.append((rotated[i], rotated[n - i]))
        rounds.append(games)
    return rounds


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def run_one_tournament(
    trial_num: int,
    n_trials: int,
    seed: int,
) -> tuple[dict, dict]:
    """
    Run one full ACF Nationals simulation.
    Returns (true_params, fitted_params) where each dict has keys 'r', 'lam', 'eta'.
    """
    rng = np.random.default_rng(seed)
    print(f"\n=== Trial {trial_num}/{n_trials} ===")

    # ------------------------------------------------------------------
    # Step 1: Sample players and questions
    # ------------------------------------------------------------------
    print(f"  Sampling {NUM_PLAYERS} players and {TOTAL_QUESTIONS} questions...", flush=True)
    lam_true, eta_true = sample_players(NUM_PLAYERS, rng)
    r_true = sample_questions(TOTAL_QUESTIONS, rng)

    # ------------------------------------------------------------------
    # Step 2: Form teams — team t has players [4t, 4t+1, 4t+2, 4t+3]
    # ------------------------------------------------------------------
    team_players: list[list[int]] = [
        list(range(4 * t, 4 * t + 4)) for t in range(NUM_TEAMS)
    ]

    # ------------------------------------------------------------------
    # Step 3: Pre-rank teams by strength score (descending)
    # ------------------------------------------------------------------
    print(f"  Pre-ranking {NUM_TEAMS} teams...", flush=True)
    strengths = [
        compute_team_strength(team_players[t], lam_true, eta_true)
        for t in range(NUM_TEAMS)
    ]
    sorted_teams = sorted(range(NUM_TEAMS), key=lambda t: -strengths[t])
    # team_rank[t] = pre-rank index (0 = strongest team)
    team_rank: dict[int, int] = {t: r for r, t in enumerate(sorted_teams)}

    # ------------------------------------------------------------------
    # Step 4: Organize questions into packets
    # Packet k contains questions [20k, ..., 20k+19].
    # Packets 0–6 → prelims; packets 7–16 → playoffs.
    # All brackets share the same packet within each round.
    # ------------------------------------------------------------------
    packets: list[list[int]] = [
        list(range(20 * k, 20 * k + 20))
        for k in range(NUM_PRELIM_PACKETS + NUM_PLAYOFF_PACKETS)
    ]

    # ------------------------------------------------------------------
    # Step 5: Prelim bracketing
    # 8 tiers of 6 teams (tier 0 = ranks 0–5, …, tier 7 = ranks 42–47).
    # Shuffle within each tier, then assign one team per tier to each bracket.
    # Each bracket ends up with 8 teams (one from each tier).
    # ------------------------------------------------------------------
    tiers: list[list[int]] = [sorted_teams[6 * k : 6 * k + 6] for k in range(8)]
    tier_assignments: list[list[int]] = []
    for tier in tiers:
        shuffled = list(tier)
        rng.shuffle(shuffled)
        tier_assignments.append(shuffled)

    # prelim_brackets[b] = list of 8 team indices for prelim bracket b
    prelim_brackets: list[list[int]] = [
        [tier_assignments[k][j] for k in range(8)]
        for j in range(NUM_PRELIM_BRACKETS)
    ]

    # ------------------------------------------------------------------
    # Step 6: Simulate prelims — 7-round round-robin within each bracket
    # ------------------------------------------------------------------
    rr8 = generate_round_robin_schedule(TEAMS_PER_PRELIM_BRACKET)
    all_gamerooms: list[GameRoom] = []
    prelim_wins: dict[int, int] = {t: 0 for t in range(NUM_TEAMS)}

    for b_idx, bracket in enumerate(prelim_brackets):
        for r_idx, round_games in enumerate(rr8):
            print(
                f"  Prelims: bracket {b_idx+1}/{NUM_PRELIM_BRACKETS},"
                f" round {r_idx+1}/{NUM_PRELIM_PACKETS}...",
                flush=True,
            )
            packet = packets[r_idx]
            for local_a, local_b in round_games:
                team_a_idx = bracket[local_a]
                team_b_idx = bracket[local_b]
                room, a_wins = simulate_game(
                    set(team_players[team_a_idx]),
                    set(team_players[team_b_idx]),
                    packet,
                    r_true, lam_true, eta_true, rng,
                    team_rank[team_a_idx],
                    team_rank[team_b_idx],
                )
                all_gamerooms.append(room)
                if a_wins:
                    prelim_wins[team_a_idx] += 1
                else:
                    prelim_wins[team_b_idx] += 1

    print("  Prelims complete. Re-bracketing...", flush=True)

    # ------------------------------------------------------------------
    # Step 7: Re-bracket for playoffs
    # Within each prelim bracket, rank by wins (tie-break: lower pre-rank wins).
    # Slots 0–1 → playoff bracket 0 (championship)
    # Slots 2–3 → playoff bracket 1
    # Slots 4–5 → playoff bracket 2
    # Slots 6–7 → playoff bracket 3
    #
    # playoff_bracket_contents[pb][b_idx] = [team_a, team_b] from that prelim bracket.
    # Teams from the same prelim bracket in the same playoff bracket are the pre-played pair.
    # ------------------------------------------------------------------
    playoff_bracket_contents: list[dict[int, list[int]]] = [
        defaultdict(list) for _ in range(NUM_PLAYOFF_BRACKETS)
    ]
    for b_idx, bracket in enumerate(prelim_brackets):
        ranked = sorted(bracket, key=lambda t: (-prelim_wins[t], team_rank[t]))
        for slot, team in enumerate(ranked):
            pb_idx = slot // 2  # 0,0,1,1,2,2,3,3
            playoff_bracket_contents[pb_idx][b_idx].append(team)

    # ------------------------------------------------------------------
    # Step 8: Simulate playoffs — 10-round schedule for each playoff bracket
    #
    # Strategy for avoiding the pre-played matchup:
    #   The 12-team circle-method round-robin has 11 rounds.
    #   Round 0 pairs (with teams labeled 0–11) are:
    #     (0,1), (2,11), (3,10), (4,9), (5,8), (6,7)
    #   We assign labels to the 12 teams so that their pre-played pair falls in
    #   round 0, then simply drop round 0 and use rounds 1–10 as the schedule.
    # ------------------------------------------------------------------
    rr12 = generate_round_robin_schedule(TEAMS_PER_PLAYOFF_BRACKET)
    # Round-0 label pairs — must match what generate_round_robin_schedule produces
    round0_label_pairs: list[tuple[int, int]] = [(0, 1), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]

    for pb_idx in range(NUM_PLAYOFF_BRACKETS):
        pre_played_pairs: list[list[int]] = list(playoff_bracket_contents[pb_idx].values())
        # pre_played_pairs[i] = [team_a_idx, team_b_idx] from same prelim bracket

        # Assign labels 0–11 so pre-played pairs align with round-0 pairings
        label_to_team: dict[int, int] = {}
        for pair_idx, pair in enumerate(pre_played_pairs):
            la, lb = round0_label_pairs[pair_idx]
            label_to_team[la] = pair[0]
            label_to_team[lb] = pair[1]

        # Skip round 0 (pre-played pairs); use rounds 1–10 for playoffs
        playoff_rounds = rr12[1:]

        for r_idx, round_games in enumerate(playoff_rounds):
            print(
                f"  Playoffs: bracket {pb_idx+1}/{NUM_PLAYOFF_BRACKETS},"
                f" round {r_idx+1}/{NUM_PLAYOFF_PACKETS}...",
                flush=True,
            )
            packet = packets[NUM_PRELIM_PACKETS + r_idx]
            for label_a, label_b in round_games:
                team_a_idx = label_to_team[label_a]
                team_b_idx = label_to_team[label_b]
                room, _ = simulate_game(
                    set(team_players[team_a_idx]),
                    set(team_players[team_b_idx]),
                    packet,
                    r_true, lam_true, eta_true, rng,
                    team_rank[team_a_idx],
                    team_rank[team_b_idx],
                )
                all_gamerooms.append(room)

    print(f"  Playoffs complete. {len(all_gamerooms)} game rooms collected.", flush=True)

    # ------------------------------------------------------------------
    # Step 9: Fit parameters via MAP with empirical Bayes
    # ------------------------------------------------------------------
    print("  Fitting parameters (this may take a while)...", flush=True)
    r_fit, lam_fit, eta_fit, _hyperparams = fit_params(
        question_world_size=TOTAL_QUESTIONS,
        player_world_size=NUM_PLAYERS,
        gamerooms=all_gamerooms,
    )
    print(f"  Trial {trial_num} done.", flush=True)

    true_params = {"r": r_true, "lam": lam_true, "eta": eta_true}
    fitted_params = {"r": r_fit, "lam": lam_fit, "eta": eta_fit}
    return true_params, fitted_params


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(all_true: list[dict], all_fitted: list[dict]) -> None:
    """Generate and save 6 plots to OUTPUT_DIR: 3 scatterplots + 3 histograms."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nGenerating plots...")

    param_info = [
        ("r",   "information rate r",  "scatter_r.png",   "hist_r.png"),
        ("lam", "buzz rate λ",          "scatter_lam.png", "hist_lam.png"),
        ("eta", "neg probability η",    "scatter_eta.png", "hist_eta.png"),
    ]

    for key, label, scatter_fname, hist_fname in param_info:
        true_vals = np.concatenate([d[key] for d in all_true])
        fit_vals  = np.concatenate([d[key] for d in all_fitted])

        # --- Scatter: fitted vs true, with y=x reference line ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(true_vals, fit_vals, alpha=0.25, s=8, linewidths=0, color="steelblue")
        lo = min(float(true_vals.min()), float(fit_vals.min()))
        hi = max(float(true_vals.max()), float(fit_vals.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Fitted {label}")
        ax.set_title(f"Predicted vs Actual: {label}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, scatter_fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {scatter_fname}")

        # --- Histogram: relative error distribution ---
        rel_err = np.abs((fit_vals - true_vals) / np.clip(true_vals, 1e-9, None))
        # Clip to 99th percentile for readability
        p99 = float(np.percentile(rel_err, 99))
        rel_clipped = np.clip(rel_err, 0, p99)

        mean_err   = float(np.mean(rel_err))
        median_err = float(np.median(rel_err))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rel_clipped, bins=60, edgecolor="none", alpha=0.8, color="steelblue")
        ax.set_xlabel("Absolute relative error  |fitted − true| / true")
        ax.set_ylabel("Count")
        ax.set_title(f"Absolute Relative Error Distribution: {label}")
        ax.text(
            0.97, 0.95,
            f"Mean:   {mean_err:.3f}\nMedian: {median_err:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, hist_fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {hist_fname}")

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate ACF Nationals and evaluate parameter recovery."
    )
    parser.add_argument(
        "--trials", type=int, default=N_TRIALS,
        help=f"Number of simulation trials (default: {N_TRIALS}). "
             "Each trial fits ~724 parameters via L-BFGS-B — may be slow.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed; trial k uses seed + k (default: 42).",
    )
    args = parser.parse_args()

    all_true: list[dict] = []
    all_fitted: list[dict] = []

    for trial in range(1, args.trials + 1):
        true_p, fit_p = run_one_tournament(trial, args.trials, seed=args.seed + trial)
        all_true.append(true_p)
        all_fitted.append(fit_p)

    make_plots(all_true, all_fitted)


if __name__ == "__main__":
    main()
