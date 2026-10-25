# Decisions

Every non-obvious choice behind `experiments.ipynb` and `src/atrl.py`. Each entry records the choice, the alternatives, why this one was taken, where it comes from, and what evidence would change it. Entries marked **[critique]** are the ones most worth arguing about.

---

## A. Environment

### D1. E0 reproduction target: AT50 = 662.8 d (daily), not 672 d
- **Choice.** The pass criterion is CT = 451.6 d and AT50 = 662.8 d at a 1-day decision interval.
- **Evidence.** Gallagher 2023 quotes two AT50 values:
  - "CT and the AT50 therapy progress after 450 and 662 days" (Fig 2 text);
  - "672 days" (Table 2 caption).

  We reproduce:
  - 662.8 d with daily decisions;
  - 673.2 d at 7-day decisions;
  - **672 d exactly** at a 28-day cadence *with progression checked only at decision points*.

  So 672 is most likely a monthly-cadence number with the decision-point check. CT is cadence-independent (451.6 d). Our RK4 agrees with `solve_ivp` (rtol 1e-10) to < 0.01 d at every cadence.
- **Would change if** SI 2 of the preprint states a different patient or protocol.

### D2. Initial resistant fraction f_R = 0.001, not 0.01
- **Choice.** f_R = R₀/N₀ = 0.001.
- **Source.** `MathOnco/DRL_Personalized_AT/paper_figures/figure_2.ipynb` calls `convert_ode_parameters(n0=0.75, rFrac=0.001, cost=0, turnover=0)` for the proof-of-principle patient. The repo's `LotkaVolterraModel` defaults are rS = rR = 0.027, dD = 1.5, K = 1, dS = dR = 0.
- **Why.** The Phase 2 parameter table's f_R = 0.01 gives CT = 368.7 d and AT50 = 472.8 d, which reproduces nothing. `Claude Docs/figures and code/lv_model.py` (FR0 = 0.01, cost 0.42, turnover 0.35) was a figure-making default, not the benchmark patient.

### D3. Main patient = Gallagher's proof-of-principle (PoP): cost 0, turnover 0 **[critique]**
- **Choice.** E2–E6 and E8–E10 use the PoP patient. E1 and E7 cover the (cost, turnover) plane.
- **Why.** It is the only patient with a published reference number (D1), so every other number is anchored to it.
- **Consequence.** With cost = 0 we have rS = rR. Untreated regrowth then depends only on total burden N, not on composition. So the post-withdrawal regrowth metric is identical (0.022 rS) for every policy on this patient.
- **Correction (E3c).** I originally attributed R4 ≡ R1 to cost = 0. That was wrong. On the cost-0.42 patient regrowth *does* depend on composition, yet VI-R4 = VI-R1 still holds. It holds for both durability definitions (population regrowth and resistant fraction), and at 10× the weight once the give-up-early loophole is closed. In LV competition, survival, slow resistant regrowth and a low resistant fraction are all maximised by the same thing: abundant sensitive competitors, i.e. containment. See D14 and D37.
- **Would change if** you prefer a Bruchovsky-fitted patient (Kexin0902/AT-PSC has seven) as the main patient.

### D4. Decision interval 7 days (user choice)
- **Choice.** Doses are held for 7 days between decisions in E2–E8 and E10. E9 sweeps 1/7/28 days.
- **Why.** Weekly decisions give 100–300 steps per episode, which keeps training tractable, and AT50 (673 d) sits close to the paper's numbers. Daily episodes run 700+ steps. Monthly decisions risk progressing between decisions.
- **Knock-on.** Schedule baselines must align with the cadence. At 7 days, IT is 28 on/28 off and AT-PSC uses a 28-day surveillance cycle. Ma's exact daily versions (IT 30/30, AT-PSC 13 of 30) are reported in E1 for fidelity.

### D5. Integrator: fixed-step RK4, dt = 0.5 d, progression checked at every substep
- **Choice.** Progression (N ≥ 1.2 N₀) is tested at every substep, and the crossing time is linearly interpolated. Episodes end at progression or at the cap.
- **Why.** A fixed step vectorises over hundreds of patients and environments, which `solve_ivp` cannot. The continuous check is Phase 2 trap #1; E0 shows what the decision-point check does (it overstates TTP by up to 6 d at 7-day decisions).
- **Verified.** It agrees with `solve_ivp` to < 0.01 d (E0).

### D6. Episode cap = 30 years, with Gallagher's +5 survival bonus
- **Why.** This is R1's own definition. PoP patients always progress well before it. Some E7 cohort patients near Strobl's controllability bound reach it.
- **Caveat.** VI is stationary and has no cap (D16). VI ceilings are therefore only used where the cap is not reached: PoP and the E10 transfer environments.

### D7. Observation = full composition (oracle) **[critique]**
- **Choice.** The network sees [N/N₀, S/K, R/K, log₁₀(R/N)/4 + 1].
- **Alternatives.**
  - Burden-only, N/N₀ (Gallagher's s₁). This is the honest clinical case, since PSA does not measure composition.
  - Frame-stacked history (Aif's s₃).
- **Why.**
  1. R3 and R4 are *defined* on composition, so a burden-only agent could not even represent R3's optimum. The comparison would then be confounded with partial observability.
  2. VI works on the same (S, R) state, so the ceilings are comparable.
  3. The Phase 2 doc says to feed N/N₀, S/K and R/K.

  The log-fraction feature was added because R/K ≈ 10⁻³ early on is invisible to an MLP, and R3 needs to resolve f between 0.001 and 0.5.
- **Would change if** you want the clinically honest version. That is a single-line change in `features()` plus a retrain, and would be a natural E11.

### D8. Action = binary {0, D_max}
- **Why.** This follows Gallagher, Ma and Phase 2. Bang-bang is optimal in the related models (Gluzman, Wang), and the action stays clinically legible. Graded or continuous doses are out of scope.

### D9. γ = 0.999 per 7-day step
- **Why.** It is Gallagher's value. The effective horizon is 1/(1−γ) = 1000 steps ≈ 19 years, so PoP episodes (60–300 steps) are almost undiscounted. This is the reviewer attack "γ = 0.999 is effectively undiscounted", answered here by stating it.
- **E9.** E9 rescales γ_Δ = 0.999^(Δ/7) so the horizon stays fixed **in days** across cadences.

---

## B. Objectives (all in `atrl.reward`)

### D10. R1 survival + holiday: Gallagher 2023 Table 1, verbatim
- **Reward.** +0.1 per step survived, +0.05 per step with D = 0, −0.1 on the progression step (instead of +0.1), +5 on reaching 30 years.

### D11. R2 dose + time: penalty calibrated so that dying early never pays **[critique]**
- **Choice.** r = −0.1·(D + σ) per step, with σ = 0.01. Progression costs −(worst step)/(1−γ) = −0.101/0.001 = −101. Reaching the cap gives +5.
- **Why not literal Gluzman.** Gluzman minimises dose plus time *until recovery*, with a failure barrier. Our environment has no recovery state, and every PoP episode ends in progression. With all rewards negative and a small progression penalty, the optimal policy is "never treat and progress at day 41". That rewards early death.

  Setting the penalty equal to the discounted cost of the worst possible infinite future removes that incentive *by construction*. A side effect is that the marginal value of one more week alive (≈ 0.101·γ^t) is roughly equal to the cost of one week of full dose (0.1). R2 therefore sits at the dose-efficiency margin: it treats only when a week of drug buys at least about a week of delay.
- **Would change if** a recovery barrier existed (e.g. the E7 controllable patients).

### D12. R3 coexistence: Giles & Newton entropy, scaled and with a progression penalty
- **Choice.** x = (S/N, R/N) and H = −Σ x log₁₀ x, so H_max = log₁₀ 2.
  - +0.1 if H ≥ 0.9995 H_max, which means f ∈ [0.487, 0.513].
  - −0.1 if H < 0.05 H_max, which means f < ≈0.0055 or f > ≈0.9945. **The initial state (f = 0.001) is in the −0.1 zone.**
  - 0 otherwise.
  - Progression gives −0.1/(1−γ) = −100, the same no-suicide rule as D11.
- **Why ±0.1, not ±1.** It puts R3 on R1's per-step scale (Phase 2 "reward scales are mismatched" failure mode).
- **Known property, not a bug.** The two-population maximum is 50% resistant, so the faithful port deliberately grows resistance. E6 generalises this.

### D13. R3-f* (E6): same relative band around a target fraction
- **Reward.** +0.1 if |f − f*| ≤ 0.026·f*. The 2.6% is the faithful band's half-width relative to 0.5. The boundary and progression terms match R3.

### D14. R4 durability: the resistant *fraction* after a simulated holiday, scored per step **[critique]**
- **Choice (current).** r = +0.1 − 0.1·f₁₈₀ per step. f₁₈₀ is the resistant fraction R/N the tumour would have after **180 untreated days**, simulated from the post-step state. Progression gives −0.1 and the cap gives +5, as in R1. Because f ≤ 1, the per-step reward stays in [0, 0.1], so the default weight has no early-progression incentive.
- **Why the fraction's *level*, not its growth rate.** Untreated, d ln f/dt = (1 − N/K)(r_R − r_S)(1 − f). That is ≤ 0 whenever cost ≥ 0, so the resistant fraction **never rises without drug** in this model. A penalty on its post-withdrawal growth would therefore always be zero. This is asserted in `atrl.py`'s self-check.
- **Why per step, not terminal-only (Yao).** Every episode ends at progression with N = 1.2 N₀. A terminal durability term is then nearly constant across policies, or else it is minimised by progressing early while f is still small. The per-step version scores every decision on how durable a state it leaves. It is not Yao's literal reward.
- **Superseded version (`R4pop`).** The first version penalised the 180-day untreated growth rate of the resistant *population*, r = +0.1 − 0.1·clip(ĝ, 0, 1). Its 35 checkpoints are kept in `results/agents/superseded_R4pop/`.
- **Result.** Both versions give the **same optimum as R1** on both reference patients. Containment keeps sensitive cells as abundant as the burden cap allows, which maximises survival, minimises resistant regrowth and minimises the resistant fraction all at once. The E3c weight check shows:

  | Form | Weight | Optimum |
  |---|---|---|
  | Fraction | 0.1 (default) | identical to R1 |
  | Fraction | 0.5, 1.0 | TTP drops, but only via a loophole: the episode ends exactly where 0.1 − w·f = 0 (f ≈ 0.1/w), i.e. the agent stops treating once a week of survival would score negatively |
  | Fraction, `+` (D11 progression penalty) | 0.5, 1.0 | identical to R1 again |
  | Population | 1.0 | identical to R1 |
  | Population | 10 | "progress immediately" (the same loophole) |
- **Conclusion.** In this model, durability is not separable from survival. A durability weight changes the policy only by rewarding giving up.
- **Would change if** the environment had a mechanism that makes a high sensitive population costly. Examples: toxicity (R5), mutation pressure proportional to burden (West §3.2), or turnover high enough that the resistant fraction can be pushed down. These are exactly the model extensions the Phase 2 doc lists for E10/future work.

### D15. Reward scaling
- **Choice.** Per-step magnitudes are hand-normalised to ~0.1. On top of that, PPO divides rewards by a running standard deviation of the discounted return.
- **Why it is fair.** The running-std scaling is the *same mechanism* for every objective. It is not per-objective tuning.

### D16. R5 toxicity (Gevertz 2026) and the toxicity-feedback baseline are skipped
- **Why.** They are optional in Phase 2. They add a third state variable, which means a 3-D VI grid, and are not needed to answer the E0–E10 questions.
- **Upgrade path.** A toxicity ODE in `deriv`, a `R5` branch in `reward`, and a 3-D `Grid`.

---

## C. Optimality ceiling (E2)

### D17. Policy iteration on an (N, log f) grid, ceiling = realised return
- **Grid.** 161 linear burden levels in [0.02, 1.2]·N₀ by 241 log-spaced resistant fractions in [10⁻⁴, 1].
  - Burden is linear because thresholds live there.
  - Fraction is logarithmic because f spans 10⁻³ to 1.
- **Method.** Transitions use the *same* `advance` and `reward` code as training, with bilinear interpolation and absorbing progression. Exact Howard policy iteration (sparse `spsolve`) converges in 3–10 iterations, about 1–2 s.
- **Ceiling.** The greedy one-step-lookahead policy is simulated in the continuous environment, and its realised return is the ceiling. It agrees with the grid value to < 1%.
- **Score.** Agents are scored as (G − G_no-treatment) / (G_VI − G_no-treatment).
- **Not a strict upper bound.** It is a discretised optimum, and an agent can in principle score slightly above 1.

### D18. "Did not train" threshold = 0.7 of own ceiling
- **Choice.** Agents below 0.7 are reported but excluded from the E4 aggregates and from E5. This is the Phase 2 exclusion of "agents that did not train".

---

## D. Learning

### D19. Custom vectorised PPO (user choice), one configuration for every objective
- **Configuration.**
  - Network: actor and critic are separate 2×128 tanh MLPs, and the policy is a Bernoulli over the action.
  - Rollouts: 32 parallel environments, 128 steps each.
  - Optimisation: 10 epochs, minibatch 512, Adam 3e-4 decaying linearly to 0, clip 0.2.
  - Losses: GAE λ = 0.99, entropy 0.01, value coefficient 0.5, gradient clip 0.5.
  - Budget: **10⁶ environment steps**.
- **Why custom.** It adds no new dependencies, and batched numpy environments make one run take about 2–3 minutes on one core.
- **Parallelism.** Runs go through a spawn pool of **6 workers**. 14 workers, each importing PyTorch, exhausted 16 GB of RAM.

### D20. λ = 0.99 and 10⁶ steps were chosen on pilot runs, applied to all objectives at once **[critique]**
- **Budget pilot.** 3×10⁵ steps gave R1 0.73 of its ceiling and was still rising. 10⁶ steps gave 0.88.
- **λ pilot.**

  | | λ = 0.95 | λ = 0.99 |
  |---|---|---|
  | R3, seeds 0/1/2 | 0.27 / 0.62 / 0.76 | 0.76 / 0.77 |
  | R1 | 0.88 / 0.88 | 0.87 / 0.87 |

  Most of R3's achievable value comes from *delaying* a distant −100 penalty, which short-λ advantages cannot propagate.
- **Fairness.** λ was changed for every objective at once, and R1 was unaffected. The Phase 2 rule "same algorithm, architecture, budget and seed set" holds. The pilot checkpoints are not part of the reported results.

### D21. Greedy evaluation is primary; stochastic is a check
- **Why.** The environment is deterministic, so a greedy policy gives one trajectory per agent per patient. Gallagher and Ma average stochastic rollouts, so E4 also reports a TTP column from 50 stochastic rollouts per agent.

### D22. Seeds
- **Assignment.**
  - E3: seeds 0–9. Ten seeds are structural, because they form the noise floor in E5.
  - E6, E7, E9: seeds 0–4. These are robustness experiments.
  - Cohort: training seed 1, test seed 2.
  - Perturbation: seeds 15, 20 and 25.
- **Config.** Everything is recorded in `results/config.json` and `atrl.jobs()`.

---

## E. Baselines and metrics

### D23. AT-x definition
- **Rule.** Treat until N < x·N₀, then withhold until N ≥ N₀, for x ∈ {0.25, 0.5, 0.75, 0.9}. AT50 is Zhang's protocol. A higher x withdraws earlier and keeps more burden.

### D24. AT-PSC is tuned, and its tuned optimum is a full cycle **[critique]**
- **Ma's literal rule fails here.** Treating 13 of every 30 days when N > N₀ gives 80 d on this patient. That rule was fitted to cytostatic dynamics and Bruchovsky patients.
- **Tuning.** Because it is a tuned heuristic, we tuned N_T ∈ {0.5…1.1}·N₀ and T1 ∈ {7, 14, 21, 28} of 28 days. The optimum is **T1 = 28, N_T = N₀**: treat the whole cycle if burden is at or above baseline. That is AT-FSC, a pure containment rule, and it is the best heuristic at 1135 d.
- **Naming.** Tables keep the name "AT-PSC" for this tuned rule. `BEST_H` (best heuristic overall) is used for reference lines; `BEST_AT` is the best Zhang-style AT-x.

### D25. Metric changes from the Phase 2 table
- **Maximum burden.** It equals 1.2 N₀ by construction for every progressed episode, so it was replaced with **time-averaged burden**.
- **Dose.** "% of MTD dose" is reported as `dose_pct` (days on drug ÷ days alive). Time off treatment = 100 − `dose_pct`.
- **Regrowth.** Post-withdrawal regrowth is the 180-day untreated growth rate of R from the final state, in units of r_S. It is degenerate on PoP (D3).
- **Objective returns.** Every policy also gets G_R1…G_R4, its discounted return under each objective. That makes the score matrix possible.

---

## F. Analyses

### D26. E5 divergence test
- **Shared state set.** The union of states visited on PoP by all E3 agents, VI policies and baselines. These are states that matter, rather than an arbitrary grid that includes unreachable regions.
- **Threshold.** Gallagher's sigmoid of P(treat) against N/N₀. Its R² is reported as "how much burden alone explains".
- **Divergence.** Symmetric KL with probabilities clipped at 10⁻⁶, plus the greedy-action disagreement rate. KL is dominated by disagreements once policies are near-deterministic, and the disagreement rate is the interpretable version.
- **Noise floor.** Within-objective seed-to-seed divergence.
- **Test.** For each pair of objectives, a label-permutation test (5000 permutations) of mean between-objective minus mean within-objective divergence, Bonferroni α = 0.05/6. "Different" requires **both** KL and disagreement to be significant.

### D27. E7 cohort range: cost ∈ [0, 0.4], turnover ∈ [0, 0.04]
- **Why.** Above Strobl's bound, turnover > 0.1·(1 − cost) for n₀ = 0.75, the tumour never progresses even under MTD, and TTP is censored. The range keeps every patient below the bound.
- **Generalists.** They do **not** see patient parameters, only the state.

### D28. E8 perturbs kinetic parameters only
- **Choice.** r_S, r_R, d_D, f_R and d_T are each scaled by an independent U[1 − p, 1 + p]. N₀ and K are fixed.
- **Why.** With n₀ > 1/1.2 the tumour can never reach 1.2 N₀, so perturbing N₀ or K mostly censors the endpoint instead of testing robustness. r_R is perturbed independently of r_S, so the cost of resistance can become negative. On PoP, d_T = 0, so it stays 0.

### D29. E9 keeps the same step budget at every cadence (confound stated)
- **Choice.** Every cadence gets 10⁶ steps.
- **Confound.** Daily agents see 7× fewer simulated days than weekly agents, and 28-day agents see 4× more. Holding simulated days fixed would cost 7× more compute for the daily runs. Any "lower frequency trains better" result is therefore partly a data-volume effect.

### D30. E10 uses two drug-mechanism variants, evaluation only
- **Cytostatic.** max(0, 1 − d_D·D): the drug can at most arrest proliferation. At zero turnover nothing ever dies, so AT50 never reaches its withdrawal threshold and becomes identical to MTD. This is reported, not hidden.
- **Log-kill.** Growth-independent kill κ = r_S(1 − n₀)·d_D, so the net per-capita rate at N₀ matches Strobl's. This tests the Norton–Simon assumption without degenerating.
- **Ceilings.** VI is re-solved in each variant (VI*).
- **Correlation.** The Aif-style Spearman correlation across agents (training environment vs transfer environment) is reported.

---

### D31. E6b: a VI-only f* check on controllable and progressing patients (added after the run)
- **Why it was added.** E6 showed f* had no effect on PoP. Before reporting a null, we checked whether it is physics or a bug.
- **The argument.** R's per-capita rate r_R(1 − N/K) − d_T is negative only when N > K(1 − d_T/r_R). Below Strobl's bound, that threshold lies above 1.2 N₀, so R increases under every policy that has not yet progressed.
- **The check.** VI on one patient below the bound and one above it. Below the bound, f* changes nothing. Above it, VI holds every target exactly. The null is therefore a property of the model class in progressing patients, not of the port.

### D32. Normalised score is undefined when the ceiling does not exceed the floor
- **Rule.** `score()` returns NaN when G_VI − G_none < 0.05 (half a step's reward).
- **Where it happens.** R2 in the cytostatic environment, where no treatment is R2-optimal. Dividing by a near-zero, negative span had produced a meaningless "12.8".

### D33. "Did not train" is checked per training cadence in E9
- **Rule.** Each (objective, cadence) is scored against the VI optimum *at that cadence*, using γ_Δ and that cadence's no-treatment floor. Rows below 0.7 are starred in the figure.
- **Result.** **Every daily-trained objective falls below 0.7** (R1 0.68, R2 0.59, R4 0.67, and R3 0.23, collapsed to MTD at 452 d). Monthly-trained agents reach 0.89–1.00. The fixed step budget (D29) is the likely cause. Holding simulated days fixed would cost about 7× more compute for the daily runs, and is the first thing to try if cadence becomes a headline.

### D34. Threshold fit uses a grid-search start and bounded least squares
- **Why.** Trained policies are near step functions, and unbounded `curve_fit` diverged on them (N* = −13 000). The fit now grid-searches a start and refines with N* ∈ [0, 2] and k ∈ [0, 10⁴].
- **VI policies.** They depend strongly on composition, so a burden-only sigmoid describes them poorly (R² ≤ 0.3, and ≈ 0 for VI-R3). The R² is reported precisely so that this is visible.

### D37. E3c–E5c patient: PoP with cost 0.42, turnover still 0
- **Choice.** r_R = 0.58 r_S (Zhang 2022's fitted cost, as already given in the Phase 2 table). Every other parameter matches PoP.
- **Why.** It changes exactly one thing relative to PoP, so any difference is attributable to the cost of resistance. Turnover stays 0 so the patient still progresses under every policy (it is below Strobl's bound).
- **Same treatment as PoP.**
  - Baselines re-tuned: AT-PSC is now genuinely partial, 21 of 28 days.
  - VI ceilings re-solved.
  - Identical PPO configuration, 10⁶-step budget, seeds and 0.7 training rule.
- **Caveat.** Episodes are about 1.7× longer, so the fixed budget buys fewer episodes. 5 of 10 R3 seeds fell below 0.7 of the optimum and were excluded, as the rule requires. R3's statistics on this patient rest on 5 seeds. Raising the budget for every objective is the fix if that matters.

### D39. Redefining R4 retrains every R4 agent, not only E3–E5
- **Choice.** All 35 R4 checkpoints were retrained with the fraction definition: E3 (10), E3c (10), E7 (5) and E9 (10). The old ones were moved to `results/agents/superseded_R4pop/`.
- **Why.** Every table scores agents with the *current* `reward('R4', ...)`. Keeping E7/E9 agents trained on the old definition under the label "R4" would mislabel them.
- **New metric.** `f_holiday` (the resistant fraction after a 180-day holiday) was added to every table, since it is what R4 now optimises.

### D40. E11 burden-only observation: memoryless N/N₀, everything else identical
- **Choice.** Agents see one number, N/N₀ (`features(..., mode='burden')`), with no history, no time and no previous action. That is the reference doc's s₁ and Gallagher's input before his LSTM. The observation mode is stored in each checkpoint (`res['obs']`); older checkpoints default to `full`.
- **Why memoryless.** It is the minimal clinical case, and it isolates "losing composition" from "adding memory". A frame-stacked or LSTM variant (s₃; Kiram) is the natural follow-up. The Phase 2 doc warns that an LSTM alone is not a new question.
- **Ceilings.** E2's full-information VI optima are used unchanged. They are an upper bound for any burden-only policy, so a burden-only score is "fraction of what full information could achieve".
- **Same everything else.** PoP patient, PPO configuration, 10⁶-step budget, seeds 0–9, the 0.7 training rule, and identical analysis functions.
- **Extra test.** For each objective, full-state vs burden-only agents are compared in policy space with the same permutation test (labels `R1-full` vs `R1-burden`). This asks whether losing composition changes that objective's policy beyond seed noise.
- **Result.** No burden-only agent reached 0.7 of the optimum (all 40 scored 0.21–0.37), so the cross-objective and full-vs-burden tests are skipped, as the training rule requires. A memoryless rule over the *same* single input (`Contain(1.17)`: treat iff N ≥ 1.17 N₀) reaches 0.96 of the optimum, so the failure is in learning, not information. The agents learn low thresholds (0.65–0.85 N₀), which is consistent with critic aliasing under partial observability.
- **Would change if** the critic were given the full state during training (asymmetric actor-critic; the deployed policy stays burden-only), or the policy had memory. Either is a contained change to `train()`.

### D41. E11b asymmetric actor–critic: burden-only actor, full-state critic during training
- **Choice.** The actor (the deployed policy) sees N/N₀ only. The critic sees the full state [N/N₀, S/K, R/K, log f] and is used only to compute advantages during training. `train(obs='burden', critic_obs='full')`; `Agent.probs` calls the actor alone.
- **Why this fix.** E11's diagnosis was critic aliasing. A memoryless burden rule reaches 0.96 of the optimum, so the policy class is adequate and only the value estimate was the problem. An asymmetric critic is the standard remedy for partial observability in simulation-trained RL. It does not leak composition into the deployed policy.
- **Why not memory first.** Memory changes the *policy* class (it can infer composition from history). The asymmetric critic changes only *training*, so it isolates the failure E11 diagnosed. Memory remains the next step if a policy needs information a single burden reading cannot carry (e.g. R3's composition target).
- **Same everything else.** Same PPO configuration, 10⁶ steps, seeds 0–9 and 0.7 rule. Symmetric runs are unaffected: with `critic_obs=None` the critic sees the actor's input exactly as before, and all cached checkpoints load unchanged.
- **Result.** All 40 agents trained (0.80–0.92 of the optimum), and every objective beats its full-state counterpart. Every policy is a sharp threshold at N\* ≈ 1.15–1.16. The objective effect shrinks to a few percent: the R1–R3 gap is 76 d and the wrong-objective cost ≤ 0.035. A burden-only actor cannot act on composition, which is what made R3 distinct.
- **Caveat.** The full-state agents' ~15% gap to VI is therefore at least partly a learning inefficiency of the larger input, not only an information advantage. A larger budget or input-feature ablation for the full-state agents would separate the two.

### D42. E11c memory: frame-stacked burden chart, full-state critic (memory is the only change from E11b)
- **Choice.** The actor sees the last **8 weekly burden readings and the last 8 dose decisions** (`obs='burden_hist'`, 16 inputs, about two months of chart). The critic sees the full state during training only, as in E11b. Episodes start with the chart filled with N/N₀ = 1 and no doses. `VecEnv` and the deployed `Agent` keep identical rolling histories; a smoke test asserts that they match `history_features()` exactly.
- **Why frame-stacking, not an LSTM.** It keeps the same MLP-based PPO, so the only change is what the actor sees. It is the reference doc's s₃. The Phase 2 doc warns that adding an LSTM for its own sake is replication. Here the memory answers a specific question: is composition *inferable* from the burden response to dosing, and does R3's behaviour return when it is?
- **Why k = 8.** One treated or untreated week already moves N measurably (by ~1–3%), and the response to a dose change reveals the sensitive share. Two months covers several decisions. Longer histories (Aif uses 64) are an ablation.
- **Policy-space test.** A memory policy has no P(treat | state). Divergence is therefore measured over a shared set of **charts**: every 8-week history seen along trajectories of the E11c and E11b agents and the baselines. The permutation test is the same as in E5.
- **Extra metric.** `early_dose` is the share of the first 26 weeks spent on drug, R3's signature front-loaded treatment. It is reported for all four observation settings.
- **Result.** R1, R2 and R4 trained 10/10 and R3 6/10, with scores 0.71–0.86 (below E11b's 0.80–0.92). Within-objective policy noise is about 4× E11b's. The R1–R3 TTP gap returns to 220 d, but R3's early-treatment signature does not (0.58 like every objective; 3/10 seeds reach the full-state 0.62). R2 becomes the outlier (−20% drug, −15% TTP) while scoring slightly *below* R1-trained agents on its own objective. That is a flat-objective learning artefact (D11 puts R2 at the indifference margin), not a different optimum.
- **Would change if** the memory agents had a larger budget, an LSTM, or a longer chart. The gap between E11c and E11b is a learning-difficulty gap, and it should be closed before reading E11c's objective differences as properties of the objectives.

### D43. E11d: memory agents at 3x budget, kept alongside the 1M agents
- **Choice.** The identical E11c configuration (burden-history actor, full-state critic) is retrained at **3×10⁶ steps** for all four objectives and all 10 seeds (`jobs('E11d')`). The learning-rate schedule anneals over the longer run. The E11c (1M) agents are kept, so E11c vs E11d isolates the effect of budget.
- **Why 3×.** It is large enough to tell "still learning" from "converged" on the learning curves, and it stays affordable: about an hour for 40 runs on 6 workers.
- **Confound stated.** Only the memory agents get the larger budget. Comparisons between E11d and the full-state (E3) or burden-only (E11b) agents now differ in budget as well as observation, so the clean comparison is E11c vs E11d. Giving every setting 3M is the follow-up if E11d changes the picture.
- **Result.** Scores rose to 0.80–0.92 (E11b level), and 8/10 R3 seeds trained. The learning curves are near a plateau. The objective effect was unchanged: the R1–R3 gap is 235 d (220 at 1M), the wrong-objective cost 0.133 (0.105 at 1M), and the TTP ordering R1 ≈ R4 > R3 > R2 is fixed from about 0.5M steps. R2's dose-sparing local optimum persists (R1-trained agents still score higher on R2's own objective). R2- and R3-trained agents score no better on their own objectives than R1-trained agents. Within-objective policy noise stayed at 0.157, so the variability is intrinsic to 16-input policies, not under-training.

### D38. E4/E5 analysis is written as functions
- **What changed.** `score_matrix`, `ttp_tests`, `plot_ttp`, `policy_space`, `plot_divergence` and `plot_agent_maps` live in the E4/E5 cells and are called again for the cost patient. The analysis is therefore identical by construction.
- **Side effect.** The permutation test now seeds its RNG per call (0), rather than sharing one generator across pairs. The PoP p-values moved in the fourth decimal; no verdict changed.

---

## G. Engineering

### D35. One module plus one notebook, no YAML
- **Choice.** `src/atrl.py` holds the environment, objectives, baselines, metrics, VI, PPO and analysis helpers in about 600 lines. The config is the `CFG` dict, dumped to `results/config.json`. Training jobs are defined once in `atrl.jobs()`, so the notebook and any background launcher share the same cache paths.
- **Upgrade path.** Split into the Phase 2 `src/` layout when the module passes about 800 lines or a second environment family is added.
- **Spawn safety.** Defaults that workers need (λ, budget) live in `CFG` inside the module, because spawned workers re-import it and never see notebook-side edits.

### D36. Caching
- **Choice.** Each PPO run is pickled to `results/agents/<exp>_<obj>_s<seed>.pkl` (weights plus learning curve). `run_many` trains only the missing ones.
- **Re-running.** Delete a file to retrain that run. Evaluations are recomputed on every run (about 15 minutes), because they are cheap and must reflect the current code.
