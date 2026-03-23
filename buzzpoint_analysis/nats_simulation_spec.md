# Technical Specification: ACF Nationals Simulation Suite

## Overview

Implement `nats_simulation.py`, a script that simulates ACF Nationals tournaments using the probabilistic buzzing model from `fit_params.py`, then evaluates parameter recovery by calling `fit_params` on the simulated data. All output plots are saved to a `nats_sims_results/` subdirectory.

Import `fit_params`, `GameRoom`, `PlayedQuestion`, `Buzz` (and whatever else is needed) from `fit_params.py`. Everything else goes in this single file.

---

## Constants (Hardcoded at Top of File)

These are the population-level hyperparameters, derived from heuristic calibration. Hardcode them as module-level constants with comments explaining what they are. They should be easy to find and change.

```python
MU_R = 0.0           # log-space mean for question information rate (fixed, anchors scale)
SIGMA_R = 0.5        # log-space std for question information rate
MU_LAM = -1.2115     # log-space mean for player buzz rate
SIGMA_LAM = 0.6026   # log-space std for player buzz rate
MU_ETA = -1.0986     # logit-space mean for neg probability
SIGMA_ETA = 0.3914   # logit-space std for neg probability

NUM_TEAMS = 48
PLAYERS_PER_TEAM = 4
NUM_PLAYERS = NUM_TEAMS * PLAYERS_PER_TEAM  # 192
NUM_PRELIM_PACKETS = 7
NUM_PLAYOFF_PACKETS = 10
QUESTIONS_PER_PACKET = 20
TOTAL_QUESTIONS = (NUM_PRELIM_PACKETS + NUM_PLAYOFF_PACKETS) * QUESTIONS_PER_PACKET  # 340

NUM_PRELIM_BRACKETS = 6
TEAMS_PER_PRELIM_BRACKET = 8  # = NUM_TEAMS / NUM_PRELIM_BRACKETS
NUM_PLAYOFF_BRACKETS = 4
TEAMS_PER_PLAYOFF_BRACKET = 12  # = NUM_TEAMS / NUM_PLAYOFF_BRACKETS
```

---

## Simulation: Sampling Players, Questions, and Teams

### Step 1: Sample Players

For each player `i` in `{0, ..., NUM_PLAYERS-1}`, sample independently:

```
log_lam_true[i] ~ Normal(MU_LAM, SIGMA_LAM^2)    =>  lam_true[i] = exp(log_lam_true[i])
logit_eta_true[i] ~ Normal(MU_ETA, SIGMA_ETA^2)   =>  eta_true[i] = sigmoid(logit_eta_true[i])
```

### Step 2: Form Teams

Assign players to 48 teams of 4: team `t` consists of players `{4t, 4t+1, 4t+2, 4t+3}`.

### Step 3: Sample Questions

For each question `q` in `{0, ..., TOTAL_QUESTIONS-1}`, sample independently:

```
log_r_true[q] ~ Normal(MU_R, SIGMA_R^2)    =>  r_true[q] = exp(log_r_true[q])
```

### Step 4: Organize Questions into Packets

Packet `k` (0-indexed) contains questions `{20k, 20k+1, ..., 20k+19}`. Packets 0–6 are for prelims, packets 7–16 are for playoffs.

---

## Simulation: Pre-Ranking Teams

Compute a strength score for each team to determine seeding. The score is the probability that this team gets a tossup against an "average" opposing team (a hypothetical team of 4 players each with `lam = exp(MU_LAM)` and `eta = sigmoid(MU_ETA)`), on an "average" question (`r = exp(MU_R) = 1.0`).

Concretely, for team `t` with players having parameters `(lam_j, eta_j)`, facing 4 average opponents with `(lam_avg, eta_avg)`:

The simplest reasonable heuristic: the probability that at least one player on team `t` buzzes correctly before any opponent does, on a typical question. For a closed-form approximation:

```
team_rate_t = sum of lam_j for j in team t
opponent_rate = 4 * lam_avg
total_rate = team_rate_t + opponent_rate
P(team t buzzes first) = team_rate_t / total_rate
P(team t gets it) ≈ P(team t buzzes first) * (1 - avg_eta_of_team_t)
```

This ignores the possibility of a neg-and-convert but is fine for ranking. Sort teams by this score descending.

---

## Simulation: Prelims (7 Rounds)

### Bracketing

Divide the 48 teams into 8 tiers of 6 by pre-ranking: tier 0 = ranks 0–5, tier 1 = ranks 6–11, ..., tier 7 = ranks 42–47. Then create 6 brackets by assigning one team from each tier to each bracket. The assignment within each tier can be random (shuffle the 6 teams in each tier, then the j-th team in each tier goes to bracket j).

Each bracket has 8 teams. A round-robin of 8 teams requires 7 rounds — this matches the 7 prelim packets.

### Round-Robin Schedule

Generate a valid round-robin schedule for 8 teams across 7 rounds. Each round has 4 games. Each team plays every other team exactly once. Use the standard "circle method": fix team 0, rotate teams 1–7.

### Simulating a Game

A game consists of one packet (20 tossup questions) played between two teams. For each question in the packet, simulate a tossup (see Tossup Simulation below). Track:
- The number of tossups each team converts (correct buzzes by their players)
- The list of `PlayedQuestion` objects (for later parameter fitting)

The team with more tossups wins the game. If tied, the team with the higher pre-ranking wins.

### Tracking Results

For each bracket, track each team's win-loss record across the 7 prelim rounds. Also store who played whom (needed for playoff scheduling).

---

## Simulation: Re-Bracketing

Within each prelim bracket, rank teams by record (wins descending). Break ties by pre-ranking. The top 2 go to the championship playoff bracket, the next 2 to the second bracket, the next 2 to the third, and the bottom 2 to the fourth. This produces 4 playoff brackets of 12 teams each.

---

## Simulation: Playoffs (10 Rounds)

### Playoff Schedule

Each playoff bracket has 12 teams. Each team must play every other team in their bracket **except** the one team they already faced in prelims. That's 10 opponents, matching the 10 playoff packets.

Key observation: each playoff bracket of 12 contains exactly 2 teams from each of the 6 prelim brackets. Those 2 teams already played each other. So the 6 pre-played pairs form a perfect matching on the 12 teams. A full round-robin of 12 teams decomposes into 11 perfect matchings (rounds). Remove the one round that matches the pre-played pairs, and the remaining 10 rounds are the playoff schedule.

Implementation: generate an 11-round round-robin for 12 teams (circle method). Find which round contains all 6 pre-played pairs (they will all be in the same round if the schedule is constructed correctly — if not, try a different construction or permute team labels so they align). Remove that round. Assign the remaining 10 rounds to packets 7–16 in order.

If the standard circle method doesn't naturally align the pre-played pairs into a single round, an alternative approach: construct the schedule by first placing the pre-played pairs as "round 0" (the round to remove), then building the remaining 10 rounds via any valid algorithm (e.g., backtracking, or using a known round-robin construction and permuting). The user is fine with any correct schedule.

### Simulating Playoff Games

Same as prelim games: simulate 20 tossups per game, track results and `PlayedQuestion` data.

---

## Tossup Simulation

Given: `team_a` (set of player indices), `team_b` (set of player indices), `question_number` (int), and the true parameters `r_true[q]`, `lam_true[i]`, `eta_true[i]`.

```
r_q = r_true[question_number]
all_players = team_a ∪ team_b

1. For each player i in all_players:
     draw T_i ~ Exponential(rate=lam_true[i])    # in information-space (delta)

2. Find p1 = argmin_i T_i.
   If T_p1 > r_q:
     → No buzz. Return PlayedQuestion(question_number, buzzes=[])

3. Otherwise, p1 buzzes at celerity c1 = T_p1 / r_q.
   Draw correct1 ~ Bernoulli(1 - eta_true[p1]).
   
4. If correct1:
     → Return PlayedQuestion(question_number, buzzes=[Buzz(p1, c1, True)])

5. If not correct1 (neg):
     Determine the opposing team: if p1 ∈ team_a, then team_Y = team_b, else team_Y = team_a.
     
     For each player j in team_Y:
       j "converts" if T_j < r_q AND Bernoulli(1 - eta_true[j]) is True
     
     Let converters = set of players in team_Y who convert.
     
     If converters is empty:
       → Return PlayedQuestion(question_number, buzzes=[Buzz(p1, c1, False)])
     
     Else:
       Pick p2 uniformly at random from converters.
       c2 = T_p2 / r_q   (this celerity is recorded but ignored by the fitter)
       → Return PlayedQuestion(question_number, 
             buzzes=[Buzz(p1, c1, False), Buzz(p2, c2, True)])
```

---

## Parameter Recovery: Calling fit_params

After simulating the full tournament (prelims + playoffs), collect all `GameRoom` objects. Each game produces one `GameRoom`:

```python
GameRoom(
    team_a=set_of_team_a_player_indices,
    team_b=set_of_team_b_player_indices,
    played_questions=list_of_20_PlayedQuestion_objects
)
```

The total number of game rooms per tournament is: (6 brackets × 28 games) + (4 brackets × 60 games) = 168 + 240 = 408 games. Wait let me recount:

- Prelims: 6 brackets × C(8,2) = 6 × 28 = 168 games. But actually 6 brackets × 7 rounds × 4 games/round = 168. ✓
- Playoffs: 4 brackets × 10 rounds × 6 games/round = 240 games. ✓
- Total: 408 game rooms.

Call `fit_params(question_world_size=TOTAL_QUESTIONS, player_world_size=NUM_PLAYERS, gamerooms=all_gamerooms)`.

Compare the fitted parameters against the true sampled parameters.

---

## Running Multiple Trials

Run `N_TRIALS` simulations (default: set this to something reasonable like 5–10 given that each trial involves fitting ~340 + 384 = 724 parameters via optimization — this may be slow; include a command-line argument or constant to control it). For each trial:

1. Sample fresh players, questions, teams
2. Run the full tournament simulation
3. Call `fit_params` on the simulated data
4. Store `(true_params, fitted_params)` for all three parameter types

Print progress liberally: which trial we're on, when sampling is done, when prelims start/finish, when playoffs start/finish, when fitting starts/finishes, and the fitting convergence info.

---

## Output Plots (6 total, saved to `nats_sims_results/`)

Create the `nats_sims_results/` directory if it doesn't exist.

Use `matplotlib`. For all plots, use clear axis labels, titles, and reasonable figure sizes.

### Scatterplots (3 plots): Predicted vs. Actual

One scatterplot each for `r`, `lam`, `eta`. Each point is one parameter from one trial (so with 5 trials, the r plot has 5 × 340 = 1700 points, the lam plot has 5 × 192 = 960 points, same for eta).

- X-axis: true parameter value
- Y-axis: fitted parameter value
- Include a diagonal y=x reference line
- Title: e.g., "Predicted vs Actual: r (information rate)"
- Save as: `nats_sims_results/scatter_r.png`, `scatter_lam.png`, `scatter_eta.png`

### Histograms (3 plots): Relative Error Distribution

One histogram each for `r`, `lam`, `eta`. Relative error = `(fitted - true) / true` for `r` and `lam`. For `eta`, since values can be near zero, use `(fitted - true) / true` but consider clipping extreme outliers for readability (or use absolute error if relative error is too noisy — use your judgment).

- Print the mean relative error on the plot (e.g., as a text annotation)
- Print the median relative error too
- Title: e.g., "Relative Error Distribution: r"
- Save as: `nats_sims_results/hist_r.png`, `hist_lam.png`, `hist_eta.png`

---

## Progress Printing

Print at least the following during execution:

```
=== Trial 1/5 ===
  Sampling 192 players and 340 questions...
  Pre-ranking 48 teams...
  Prelims: bracket 1/6, round 1/7...
  ...
  Prelims complete. Re-bracketing...
  Playoffs: bracket 1/4, round 1/10...
  ...
  Playoffs complete. 408 game rooms collected.
  Fitting parameters (this may take a while)...
  Fitting complete. Final NLL: 12345.6, Converged: True
  Trial 1 done.
=== Trial 2/5 ===
...
Generating plots...
Saved scatter_r.png
...
Done.
```

---

## Summary of File Structure

```
nats_simulation.py
├── Constants (hyperparameters, tournament structure)
├── sample_players(num_players) → (lam_true, eta_true)
├── sample_questions(num_questions) → r_true
├── compute_team_strength(team, lam_true, eta_true) → float
├── simulate_tossup(team_a, team_b, question_number, r_true, lam_true, eta_true) → PlayedQuestion
├── simulate_game(team_a, team_b, packet_questions, r_true, lam_true, eta_true) → (GameRoom, team_a_wins: bool)
├── generate_round_robin_schedule(num_teams) → list of rounds
├── run_one_tournament(seed) → (true_params, fitted_params)
│   ├── Sample players & questions
│   ├── Pre-rank teams
│   ├── Bracket for prelims
│   ├── Simulate prelims (7 rounds)
│   ├── Re-bracket for playoffs
│   ├── Simulate playoffs (10 rounds)
│   ├── Collect all GameRooms
│   └── Call fit_params
├── main()
│   ├── Run N_TRIALS tournaments
│   ├── Aggregate results
│   └── Generate and save 6 plots
```
