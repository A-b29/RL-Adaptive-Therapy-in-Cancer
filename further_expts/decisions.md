# Decisions: follow-up experiments

Every non-obvious choice behind `further_expts_experiments.ipynb` and `fx.py`. It continues the root `decisions.md` (entries D1–D43), so entries here are numbered **F0, F1, …**. Each records the choice, the alternatives, why, and what would change it. Entries marked **[critique]** are the ones most worth arguing about.

---

## A. Engineering

### F0. The engine (`src/atrl.py`) was extended, not forked, and every extension is off by default **[critique]**
- **Choice.** New optional arguments, each defaulting to the original behaviour:
  - `patient(..., mu=, eps=, h=)`: the keys are only added when given, and `deriv` applies a term only if its key is present;
  - `reward`: new names `R3s`, `R3d{f*}`, `R1p`;
  - `solve_vi(..., actions=)`, `Lookahead` over any action set;
  - `ActorCritic(na=)`, `train(..., actions=)` (categorical policy), `Agent.dist()`.
- **Alternative.** A forked copy of the engine in this folder. Rejected: two diverging copies of the environment would make "same environment" claims unverifiable.
- **Verification** (run before any training):
  1. `python src/atrl.py` self-check passes;
  2. the VI value functions and policies on PoP are identical to the original module (max |ΔV| = 0) for all four objectives;
  3. four cached agents (E3, E11b, E11d) give identical trajectories;
  4. binary PPO, full-state and memory, is **bit-identical** to the original module over two updates;
  5. notebook §0 re-asserts the published E1/E2 numbers and the E3/E11b/E11c/E11d TTP medians.
- **Would change if** the module passes ~800 lines (root D35). It is now ~750.

### F1. A separate folder; the original notebook is untouched
- **Choice.** Everything new is in `further_expts/`, with its own `results/` and `figures/`. The original checkpoints are reloaded read-only from `../results/agents/`.
- **Consequence.** The analysis helpers of `experiments.ipynb` (score matrix, TTP tests, permutation test, policy space, plots) are **copied verbatim** into the new notebook. Refactoring them into a module would have meant editing the original notebook. `policy_space` is generalised to graded doses, and an assertion checks that the generalised KL equals the original `sym_kl` for binary policies.

### F2. Training runs in a launcher, never in the notebook
- **Choice.** `python fx.py` trains every missing job with 6 spawn workers (root D19: 14 workers ran a 16 GB machine out of memory). The notebook's `load(exp)` asserts that all checkpoints exist and otherwise refuses with a message.
- **Why.** About 430M environment steps (≈ 3–4 h). A kernel crash or a memory kill would lose a long in-kernel run. The launcher is resumable because each finished job is a `.pkl`.
- **What happened.** Claude Code's low-memory guard stopped two background runs (6 and 4 workers). The resumable cache meant nothing was lost. The last 90 jobs ran as `python fx.py 4` in the user's own terminal. Total ≈ 5 h at 4 workers, about 2 jobs/min at 10⁶ steps.

### F3. Seeds and budgets
- **10 seeds** wherever an experiment feeds a policy-space test or replaces an objective (A1, A2, D1, D2, E1, B2's `R3s`). **5 seeds** per target for B2's `R3d` sweep, matching E6 (root D22).
- **10⁶ steps** everywhere except A2 (3·10⁶, which is that experiment's point).

---

## B. Reward and analysis checks

### F4. B1 Pareto frontier
- **Axes.** Cumulative drug (days at full dose) vs TTP, *and* dose intensity (% of MTD) vs TTP. Cumulative drug grows with survival time by construction, so a long-surviving policy can never sit on a "low drug" frontier; intensity removes that coupling.
- **Third criterion.** The resistant fraction at progression, in its own plot with its own dominance. It is not combined into a score, as the doc requires.
- **Who is on it.** Every *trained* agent of three observation settings (E3, E11b, E11d), the four VI optima and the eight baselines. Dominance is computed over all of them together.
- **Spread.** Greedy evaluation gives one point per agent (root D21), so 50 stochastic rollouts per agent are pooled into a median/IQR cross per setting × objective.

### F5. B2 smooth coexistence rewards **[critique]**
- **(b) `R3s`** = 0.1·(2H/H_max − 1). It is the continuous version of the band reward: −0.1 for a pure tumour, +0.1 at 50:50, and monotone in between.
- **(c) `R3d{f*}`** = max(−0.1, 0.1·(1 − abs(log₁₀(f/f*)))): +0.1 at the target, 0 one decade away, −0.1 two decades away.
  - **Why log-distance.** A linear distance |f − f*| gives almost the full reward to the initial state (f = 0.001) when f* is small (0.05), so it would not reward reaching the target at all. Log-distance is scale-free across f*. At f* = 0.5 and f = 0.001 it gives −0.1, matching R3's boundary penalty.
- **Kept identical.** The −0.1/(1−γ) = −100 progression penalty and the ±0.1 per-step scale (doc: "keep comparable").
- **Ceilings.** VI is re-solved for each shape. Each agent is scored against its own shape's ceiling and floor.
- **Would change if** a reviewer prefers a quadratic or Gaussian bump. Any smooth, unimodal shape tests the same question (signal density for PPO).

### F6. A3 grid refinement
- **Choice.** 321 × 481 (each dimension doubled, 4× the states), all four objectives. The criterion is that both the realised-ceiling TTP and the normalised span (G_VI − G_none) move by < 1%.

### F10. B3 switch counting and the D4 gate
- **Definition.** A switch is D_t ≠ D_{t−1} along the greedy PoP trajectory. Reported per episode, per year alive and as a share of decisions.
- **Gate (stated before looking).** Run D4 if the containment optima (VI-R1, R2, R4) switch on > 25% of their weekly decisions. The doc: "if switching is already rare, skip".

---

## C. Reducing limitations

### F7. A1 patient: turnover 0.04, cost 0 **[critique]**
- **Choice.** PoP with turnover 0.04. One parameter changes, as in E3c (root D37).
- **Why 0.04.**
  - It is the largest turnover the doc allows (≤ 0.04) and the E7 cohort's edge, so turnover has the most room to change behaviour.
  - It stays below Strobl's bound (0.1 for cost 0), and MTD progresses at 474 d.
  - The VI ceilings are valid: VI-R1 ≈ 3390 d, far below the 30-year cap (root D6).
- **Alternative.** 0.02 (VI-R1 2569 d). Its episodes are shorter, so it would train more easily at a fixed budget, but it gives turnover less leverage.
- **Caveat stated in advance.** Episodes are ≈ 1.7× PoP's, like the cost-0.42 patient where 5/10 R3 seeds failed (root D37). R3 seeds may fail again, and the 0.7 rule then applies.

### F8. A2 equal budgets
- **Choice.** The exact E3 (full state) and E11b (burden actor + full-state critic) configurations at 3·10⁶ steps, all objectives, 10 seeds. The learning-rate schedule anneals over the longer run, exactly as in E11d. The 1M agents are kept, so 1M vs 3M isolates budget, and the three 3M settings (full, burden + critic, burden history) are compared at equal budget.
- **1M policy-space numbers.** Taken from `../results/summary.json`, not recomputed.

### F9. C1 map design
- **Grid.** 5 × 5 over cost {0, 0.1, …, 0.4} × turnover {0, 0.01, …, 0.04}, the E7 cohort box. Every patient is below Strobl's bound, so all progress and the VI ceilings are valid. Points whose optimum reaches the 30-year cap would be marked and excluded.
- **R1 vs R4.** Disagreement is measured over the states the two optima actually visit, not over the whole grid, which includes unreachable states.
- **Wrong-objective cost from VI alone.** For each optimum and each other objective, 1 − score, where score = (G − G_none)/(G_VI − G_none) at that patient.
- **Coexistence controllability.** The share of weeks VI-R3 spends inside its +0.1 entropy band.
- **PPO** only at a chosen point. A1's patient is the box corner (cost 0, turnover 0.04).

---

## D. Action space

### F11. D1 three doses **[critique]**
- **Policy.** A categorical over {0, 0.5, 1} from the same actor network (one logit per dose), with the same entropy coefficient (0.01) and everything else unchanged. The maximum entropy rises from ln 2 to ln 3. Keeping the coefficient fixed is the "no per-objective tuning" rule applied across action sets too.
- **Holiday bonus.** It is earned only at D = 0. That is the faithful reading of Gallagher's "+0.05 per timestep without treatment". **`R1p`** (bonus 0.05·(1 − D)) is solved with VI to show how much of any R1-vs-R4 difference this one choice causes. For binary doses `R1p` ≡ R1, which the notebook asserts.
- **No per-objective dose weights.** R2 charges D directly and handles 0.5 without change. Weight tuning risks loopholes like D14's "give up early".
- **Analysis generalisation.**
  - P(treat) becomes P(D > 0) for the sigmoid N\*.
  - KL is computed between dose distributions.
  - Disagreement is computed on the most likely dose.
  - All three reduce exactly to the originals for two doses.
- **Sanity check.** Every 3-dose ceiling must be ≥ its 2-dose ceiling, up to grid error (0.5% of the normalised span), because the action set is a superset.

### F12. D2 granularity
- **Sets.** {0, 1} ⊂ {0, 0.5, 1} ⊂ {0, 0.25, 0.5, 0.75, 1}. Monotone ceilings are asserted. **{0, 2/3, 1}** (2/3 is the dose that exactly halts sensitive growth, since 1 − 1.5·2/3 = 0) is VI-only, because it is not nested in the others.
- **PPO.** 5 levels × 4 objectives × 10 seeds. The 2-level agents are E3 and the 3-level agents are D1.

### F13. D3 dose-response: a power curve, VI-only
- **Choice.** The effect is D^h with h = 0.5 (saturating) and h = 2 (threshold-like), instead of a Hill function.
- **Why.** It is one parameter, and it keeps the effect at D = 0 and D = 1 fixed, so the comparison at maximum dose is fair (doc). A Hill curve normalised to E(1) = 1 has two parameters and the same qualitative shapes.
- **VI-only.** The question is whether the D1 optimum's use of half doses depends on the curve. VI answers it directly, and PPO adds nothing until VI shows a dependence.
- **Status.** A sensitivity analysis, not a calibrated model.

### F14. D4 switching penalty, VI-only
- **State.** (previous dose, N, log f). The grid is replicated per previous dose. The first decision is free (no previous dose), which makes the start value max_j V(j, s₀). λ = 0 reproduces the plain VI ceiling exactly, and the notebook asserts this.
- **λ grid.** {0, 0.01, 0.02, 0.05, 0.1}. Per-step rewards are ≈ 0.1, so 0.05 is half a week of survival reward per switch. The reported effectiveness is the normalised score on the **un-penalised** objective.
- **Why VI-only.** A PPO version needs the previous dose in the actor's observation (a new obs mode), which is a separate change.

---

## E. Biology

### F15. E1 conversion rate: chosen by a rule, before training **[critique]**
- **Model.** −μS in dS/dt and +μS in dR/dt (transfer, not creation). The state stays (S, R).
- **Rule.** Train PPO at the **smallest swept μ at which VI-R4's on-trajectory dose departs from VI-R1's by ≥ 0.5 percentage points**. Below that, VI says the two objectives are the same, and PPO could only find seed noise. Above it, episodes get very short (VI-R1 897 d at μ = 10⁻⁴, 345 d at 10⁻³).
- **Probe.**

  | μ | dose % R1 / R4 | TTP R1 / R4 (d) |
  |---|---|---|
  | 10⁻⁵ | 73.13 / 73.13 | identical |
  | 3·10⁻⁵ | 74.71 / 73.95 | 1246 / 1240 |

  So **μ = 3·10⁻⁵ per day**.
- **R4 caveat.** With μ > 0 the resistant fraction rises untreated, so the root D14 self-check no longer holds. R4 stays defined (the fraction after a simulated 180-day holiday now includes conversion). The atrl self-check still runs on PoP and the cost patient only.

### F16. E2 partial sensitivity, and a correction to the proposal **[critique]**
- **Model.** r_R R (1 − N/K)(1 − ε·d̂_D·D).
- **Correction.** The doc says ε > 0 "is what lets the drug reverse" resistant growth. At turnover 0 that is false for every ε < 2/3: R's per-capita rate under full dose, r_R(1 − N/K)(1 − 1.5ε), stays positive, so the drug slows R but never shrinks it. Shrinking R needs r_R(1 − N/K)(1 − 1.5ε) < d_T, i.e. turnover.
- **Choice.** The sweep (ε ∈ {0, …, 0.5}, all < 2/3) runs on PoP **and** on the A1 turnover-0.04 patient. On the latter, ε ≳ 0.4 makes a full dose shrink R near the progression line. The table reports that rate per row.
- **VI-only.** The question ("is R3's failure about the objective or about R being uncontrollable?") is about the optimum.

### F17. B4 progression threshold, VI-only
- **Choice.** VI at 1.1, 1.2 and 1.3 N₀ by setting `CFG['prog']` inside the kernel (restored in a `finally`). The VI grid's upper edge follows automatically.
- **Why VI-only.** Spawned PPO workers re-import `atrl` and would not see a changed `CFG['prog']` (root D35). The question (does the optimum's objective ranking move?) is answered by VI.

### F18. Not done: A4, A5, E3, E4 and Part F
See notebook §8 and `relevance.md`. The deciding reasons are cost against relevance (A4), the doc's own priority (A5, E4, Part F), and new observation plumbing (E3, the most relevant next step).

---

## F. Results of each decision (from the executed notebook, `results/summary.json`)

| Entry | Result |
|---|---|
| F0 | §0 reproduced E1, E2 and the E3/E11b/E11c/E11d TTP medians. Every new-path assertion passed: KL generalisation, R1p ≡ R1 for binary, monotone ceilings, h-independence of binary, λ = 0 ≡ plain VI. |
| F3 | Seeds that did not train (< 0.7): A1 R3 3/10, D2 R3 2/10. None in A2, D1, E1, B2. |
| F4 | All good policies run at ≈ 72% intensity, so cumulative drug is nearly a function of TTP; intensity was the right second axis. The E11d R2 agents are on the intensity frontier (40% dominated vs 90–100% for other groups). |
| F5 | `R3s` is worse than the band (VI 1387 vs 1625 d). `R3d` makes f\* matter (VI 1989 → 1443 d), but no policy holds f above ≈ 0.12. The log-distance choice did not change the conclusion. |
| F6 | Largest change: TTP 0.36% (R4, a near-tie with R1), span 0.02%. Criterion met. |
| F7 | 7/10 R3 seeds trained, as anticipated. Every headline result reproduced (R3 cost 0.172; R1 vs R4 at seed noise). |
| F8 | 3M: full 0.94, burden + critic 0.97, memory 0.92 of the ceiling (R1). At equal budget, burden + critic > full > memory for every objective. |
| F9 | 0/25 points capped. R1 vs R4 disagree on ≤ 0.3% of visited states; R3 penalty 20–23%; band time ≤ 1.5%. |
| F10 | VI-R1/R2/R4 switch on 54% of decisions (gate 25%), so D4 ran. |
| F11 | The holiday-bonus choice mattered: R1 avoids D = 0.5 and R1 ≢ R4 (2038 vs 2143 d), while `R1p` reproduces the R4 optimum exactly. Agents learn the half dose on ≈ 54% of decisions (R2/R3/R4). |
| F12 | VI ceilings are monotone. {0, 2/3, 1} beats 5 levels for R2 (2142 vs 2135 d). PPO scores fall from 3 to 5 levels. |
| F13 | R4's 3-dose gain: +112 d (h = 0.5), +105 d (h = 1), +56 d (h = 2). The curve matters for size, not sign. |
| F14 | Halving switches costs 3–8%. At λ = 0.1, VI-R1 still beats AT-PSC by 55%. |
| F15 | At μ = 3·10⁻⁵, VI-R4's post-holiday fraction is 0.444 vs 0.490, at −6 d of TTP. The agents' R1 vs R4 is not distinguishable (p = 0.18). The rule picked a μ where the effect is real but small. |
| F16 | The correction held. On PoP, R3's penalty stays 18–20% for ε ≤ 0.5. On turnover 0.04, ε ≥ 0.4 reaches the 30-year cap (no progression), so ε cannot rescue R3 while the patient progresses. |
| F17 | Ranking R1 = R4 > R2 > R3 at 1.1, 1.2 and 1.3 N₀. |
