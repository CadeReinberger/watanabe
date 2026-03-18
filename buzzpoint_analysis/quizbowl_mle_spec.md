# Technical Specification: Quiz Bowl MLE Parameter Fitting

## Overview

Implement a Python script (`fit_params.py`) that performs maximum likelihood estimation (MLE) to fit parameters of a probabilistic model of quiz bowl buzzing behavior. The script takes structured game data as input and returns fitted parameter vectors.

---

## Data Structures

Define the following as Python `dataclasses`:

### `Buzz`
A dataclass (or NamedTuple — your call) with three fields:
- `player`: `int` — player index in `{0, 1, ..., player_world_size - 1}`
- `celerity`: `float` — fraction of question read at buzz time, in `(0, 1]`
- `correct`: `bool` — whether the player answered correctly

### `PlayedQuestion`
- `question_number`: `int` — question index in `{0, 1, ..., question_world_size - 1}`
- `buzzes`: `list[Buzz]` — ordered list of buzzes on this question (0, 1, or 2 elements; see Observation Patterns below)

### `GameRoom`
- `team_a`: `set[int]` — player indices on team A
- `team_b`: `set[int]` — player indices on team B
- `played_questions`: `list[PlayedQuestion]`

---

## Parameters to Fit

There are `question_world_size + 2 * player_world_size` scalar parameters total:

| Parameter | Index Range | Domain | Meaning |
|-----------|------------|--------|---------|
| `r[q]` | `q ∈ {0, ..., question_world_size-1}` | `(0, ∞)` | Information rate for question `q`. Total information revealed over the full question is `r[q]`. |
| `lam[i]` | `i ∈ {0, ..., player_world_size-1}` | `(0, ∞)` | Buzz rate for player `i`. Higher means player buzzes earlier (more trigger-happy). |
| `eta[i]` | `i ∈ {0, ..., player_world_size-1}` | `(0, 1)` | Neg probability for player `i`. Probability of answering incorrectly given a buzz. |

---

## Probability Model

### Information Scale

Each question `q` maps celerity `c ∈ [0, 1]` to an information level `δ = r[q] * c`. The full question corresponds to `δ_max = r[q]` (at `c = 1`).

### Buzz Timing (Exponential Model)

Each player `i` independently has a random buzz time (in information-space) drawn from an Exponential distribution with rate `lam[i]`:

- PDF: `f(δ) = lam[i] * exp(-lam[i] * δ)`
- CDF (probability of buzzing by information level `δ`): `F(δ) = 1 - exp(-lam[i] * δ)`
- Survival (probability of NOT buzzing by `δ`): `S(δ) = exp(-lam[i] * δ)`

### First Buzz

Consider all players in the room (team_a ∪ team_b). Their buzz times are independent exponentials. The minimum of independent exponentials with rates `lam[i]` is itself exponential with rate `Λ = Σ_i lam[i]` (sum over all players in the room).

A buzz event occurs if the earliest buzz time falls before `δ_max = r[q]`. The probability that player `p` is the first to buzz, at information level `δ_p = r[q] * c_p`, has density (in δ-space):

```
f_first(p, δ_p) = lam[p] * exp(-Λ * δ_p)
```

where `Λ = Σ_{all players j in room} lam[j]`.

To convert to a density in celerity-space, multiply by the Jacobian `r[q]`:

```
f_first(p, c_p) = r[q] * lam[p] * exp(-Λ * r[q] * c_p)
```

### Correct / Incorrect Answer

Given that player `p` buzzed:
- P(correct) = `1 - eta[p]`
- P(incorrect, i.e., neg) = `eta[p]`

### After a Neg (Bounce-Back Model)

When the first buzz is incorrect (a "neg") by player `p1` on team X, team X is eliminated. The opposing team Y then has all its players attempt to answer through the end of the question. The celerity on any subsequent buzz in the data is **ignored** in the likelihood — instead, the model assumes every player `j` on team Y independently either "knows the answer" or doesn't by the end of the question.

**Per-player probability of knowing the answer (and answering correctly) by end of question:**

```
k[j] = (1 - eta[j]) * (1 - exp(-lam[j] * r[q]))
```

This is: P(player j's exponential buzz time fired before δ_max) × P(player j answers correctly given they buzzed).

**If a second (correct) buzz is observed from player `p2` on team Y:**

Player `p2` was chosen uniformly at random from the set of players on team Y who knew the answer. The likelihood contribution is:

```
P(p2 chosen) = Σ over all subsets S ⊆ team_Y such that p2 ∈ S:
    [ ∏_{j ∈ S} k[j] * ∏_{j ∈ team_Y \ S} (1 - k[j]) ] * (1 / |S|)
```

Since quiz bowl teams are small (typically 4 players), this combinatorial sum has at most `2^(|team_Y| - 1)` terms, which is very manageable.

**If no second buzz is observed after a neg:**

All players on team Y failed to know the answer by end of question:

```
P(no conversion) = ∏_{j ∈ team_Y} (1 - k[j])
```

---

## Observation Patterns and Their Likelihoods

Each `PlayedQuestion` falls into exactly one of these cases. Let `all_players = team_a ∪ team_b` and `Λ = Σ_{j ∈ all_players} lam[j]`.

### Case 1: No buzzes (`len(buzzes) == 0`)

No player buzzed before end of question. Likelihood:

```
L = ∏_{j ∈ all_players} exp(-lam[j] * r[q])
  = exp(-Λ * r[q])
```

### Case 2: One correct buzz (`len(buzzes) == 1` and `buzzes[0].correct == True`)

Player `p` buzzed at celerity `c`, answered correctly. Likelihood (density):

```
L = r[q] * lam[p] * exp(-Λ * r[q] * c) * (1 - eta[p])
```

### Case 3: One incorrect buzz, no conversion (`len(buzzes) == 1` and `buzzes[0].correct == False`)

Player `p1` buzzed at celerity `c1`, negged. The opposing team failed to convert. Determine which team `p1` is on; the other team is `team_Y`.

```
L = r[q] * lam[p1] * exp(-Λ * r[q] * c1) * eta[p1] * ∏_{j ∈ team_Y} (1 - k[j])
```

where `k[j] = (1 - eta[j]) * (1 - exp(-lam[j] * r[q]))`.

### Case 4: Incorrect buzz then correct conversion (`len(buzzes) == 2`, `buzzes[0].correct == False`, `buzzes[1].correct == True`)

Player `p1` negged at celerity `c1`. Player `p2` on the opposing team `team_Y` gave the correct answer (celerity of second buzz is ignored).

```
L = r[q] * lam[p1] * exp(-Λ * r[q] * c1) * eta[p1] * P(p2 chosen from team_Y)
```

where `P(p2 chosen from team_Y)` is the combinatorial sum described above.

---

## Optimization

### Reparameterization

Since `r[q]` and `lam[i]` live in `(0, ∞)` and `eta[i]` lives in `(0, 1)`, reparameterize to unconstrained space for the optimizer:

- `r[q] = exp(log_r[q])` — optimize over `log_r[q] ∈ (-∞, ∞)`
- `lam[i] = exp(log_lam[i])` — optimize over `log_lam[i] ∈ (-∞, ∞)`
- `eta[i] = sigmoid(logit_eta[i])` — optimize over `logit_eta[i] ∈ (-∞, ∞)`

### Objective

Minimize the **negative log-likelihood** (sum of `-log(L)` over all played questions across all game rooms).

### Solver

Use `scipy.optimize.minimize` with method `L-BFGS-B` (or plain `BFGS` since we've reparameterized to unconstrained space). Compute gradients numerically (i.e., you don't need to hand-derive gradients; let scipy do finite differences). If performance is a concern, consider using `jax` for autodiff, but scipy with finite differences is fine as a first pass.

### Initialization

Reasonable starting points:
- `log_r[q] = 0` (i.e., `r[q] = 1.0` for all questions)
- `log_lam[i] = 0` (i.e., `lam[i] = 1.0` for all players)
- `logit_eta[i] = -2` (i.e., `eta[i] ≈ 0.12`, a low neg rate)

---

## Function Signature

```python
def fit_params(
    question_world_size: int,
    player_world_size: int,
    gamerooms: list[GameRoom],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        r: np.ndarray of shape (question_world_size,) — fitted information rates
        lam: np.ndarray of shape (player_world_size,) — fitted buzz rates
        eta: np.ndarray of shape (player_world_size,) — fitted neg probabilities
    """
```

---

## Implementation Notes

1. **Numerical stability**: Work in log-likelihood space throughout. Use `np.logaddexp` where needed. Clip exponentials to avoid overflow/underflow (e.g., cap `lam[j] * r[q]` at ~500 before exponentiating).

2. **Combinatorial sum for bounce-back**: For the `P(p2 chosen)` calculation, enumerate all subsets of `team_Y` that include `p2`. Since teams are small (≤ ~6 players), this is at most 2^5 = 32 subsets. Compute in log-space for stability.

3. **Edge cases**:
   - A player who never appears in any buzz contributes only through the "no buzz" survival terms. Their `lam` will be pushed toward 0 and `eta` will be poorly identified — this is fine, the optimizer will find whatever MLE exists.
   - A question that is never played contributes nothing to the likelihood. Its `r[q]` is unidentified. You can either skip it or leave it at the initial value. Document this behavior.
   - If a player appears in multiple game rooms, their parameters are shared across all rooms (this is the whole point — we're fitting global player skill parameters).

4. **Validation**: Add a small test with synthetic data. Generate data from known parameters, fit, and verify recovery. This is optional but strongly recommended.

5. **Dependencies**: `numpy`, `scipy`, and standard library only. No need for `jax` or `pytorch` unless scipy is too slow.

---

## Output

Print or return the three arrays. Optionally also return the final negative log-likelihood value and the optimizer's convergence info for diagnostic purposes.
