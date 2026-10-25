# Relevance of the proposed follow-up experiments

`further expts.txt` proposes 20 items in six parts. This file judges each one against the **original plan**, so you can see which ones belong in the paper and in what role. The original plan is the Phase 2 document (`Claude Docs/Phase 2.pdf`), as implemented in E0–E11d.

## The yardstick: what the original plan claims

Phase 2 set out **three claims**:
1. **Divergence:** do the objectives produce different policies?
2. **Cost of the wrong objective:** the off-diagonal of the score matrix.
3. **Instrumental convergence:** does coexistence-training outlast survival-training?

It also set a **kill criterion**: every objective converging on the same threshold policy. And it listed **reviewer attacks** that each answer had to survive ("one parameter set", "your agents trained badly", "artefact of one environment", ...).

E0–E11d then produced ten findings (README). Five carry most of the weight:
- **F-a** R1 ≡ R4 (durability is not separate from survival);
- **F-b** "R3 last" (coexistence costs 12–18% elsewhere);
- **F-c** instrumental convergence in reverse;
- **F-d** why coexistence fails (a two-population target cannot be held in progressing patients);
- **F-e** observation matters more than objective (E11b–E11d).

Each proposed item is put in one of three roles:

| Role | Meaning | Where it goes in a paper |
|---|---|---|
| **Defends** | stress-tests an existing claim against a named reviewer attack; no new question | robustness paragraph, supplement table |
| **New insight** | asks something the Phase 2 plan did not, and can change or extend a finding | its own results subsection or figure |
| **Low / out of scope** | not attached to a current question, or excluded by the doc itself | limitations / future work |

## Summary

| Item | Touches | Role | Done here | Outcome |
|---|---|---|---|---|
| A1 turnover > 0 patient | F-a, F-b; attack "one parameter set"; README limitation "two reference patients, both turnover 0" | Defends | PPO + VI | Defended: same matrix shape as PoP and cost 0.42 (R1 ≡ R4, R3 cost 0.172, R1/R2/R4 ≤ 0.025) |
| A2 3× budget for every setting | F-e; root D41, D43 confound | Defends (and can overturn F-e's "burden beats full") | PPO | Defended **and sharpened**: burden + critic > full > memory at equal budget; divergence grows with budget |
| A3 finer VI grid | every score; attack "your agents trained badly" | Defends | VI | Defended: ceilings move < 0.4% |
| A4 equal simulated days per cadence | E9 (daily agents failed) | Low: cadence is not headline | no | — |
| A5 E6–E10 on the cost patient | E6–E10 | Low (doc: optional) | no | — |
| B1 Pareto frontier | claim 2; R2's anomaly (E11c/d) | **New insight** | analysis | **New**: one drug–TTP line; R2 memory agents are Pareto-efficient, not failed |
| B2 R3 reward shape | F-b, F-d; attack "it is your reward maths" | Defends **and** new (smooth targets make f\* matter) | PPO + VI | Defended (structural, VI agrees) + new: R3's cost scales with its target f\* |
| B3 switch counts | gate for D4 | New (clinical practicality) | analysis | Optima and agents switch on ≈ 54% of weeks (heuristics ≤ 22%) → D4 ran |
| B4 progression threshold | the whole control problem | Defends | VI-only | Defended: ranking unchanged at 1.1/1.2/1.3 |
| C1 cost × turnover map | F-a, F-b over Strobl's plane; Phase 2 asked for a Gallagher-5c-comparable map | Defends, turning "two patients" into "a region" | VI-only (+ A1) | Defended over the whole box: R1 ≡ R4 at 25/25 points, R3 penalty 20–23% |
| D1 three doses | F-a; root D8 excluded graded doses | **New insight** | PPO + VI | **New**: +70–168 d; R1 ≡ R4 breaks, but only through the all-or-nothing holiday bonus (`R1p` ≡ R4) |
| D2 2/3/5 levels | D1 | **New insight** (planner vs learner) | PPO + VI | **New**: VI gains with levels, PPO loses at 5; {0, 2/3, 1} gets most of the gain |
| D3 dose-response shape | D1/D2 | Defends D1/D2 | VI-only | Qualifies D1/D2: R4's gain +56 to +112 d depending on the curve |
| D4 switching penalty | clinical legibility | **New insight** (gated by B3) | VI-only | **New**: halving switches costs 3–8%; still 55% above the best heuristic at 9 switches/yr |
| E1 S → R conversion | F-a; README "would change if … mutation pressure" | **New insight**: the named route to durability ≠ survival | PPO + VI | **New, qualified**: first split of R4 from R1 at the optimum, but small and invisible to agents |
| E2 partly sensitive R | F-d | Defends F-d | VI-only | Defended: R3 penalty ≈ 20% at every ε while the patient progresses |
| E3 noisy / delayed readings | F-e | New, but not built (needs new observation plumbing) | no | recommended next |
| E4 agent chooses next visit | E9 | New, but the most invasive item; the doc says optional | no | — |
| Part F (R5 toxicity, space, PK, plasticity, LSTM, algorithm swap) | limitations | Out of scope by the doc itself | no | — |

## Per-item notes (from this run, `results/summary.json`)

**Defends: what each check closes**
- **A3.** Doubling the VI grid moves no ceiling TTP by > 0.36% and no normalised span by > 0.02%. Every score is safe.
- **A1.** Turnover 0.04 reproduces E3/E3c:
  - VI-R1 = VI-R2 = VI-R4 (3389 d), VI-R3 21% shorter;
  - agents R1 2685 / R2 2757 / R3 2277 / R4 2713 d, R3 7/10 seeds trained;
  - 5/6 pairs different, R1 vs R4 at seed noise.
  It closes the README limitation "two patients, both turnover 0".
- **C1.** Over 25 VI patients (cost 0–0.4 × turnover 0–0.04):
  - R1 and R4 disagree on ≤ 0.3% of visited states;
  - R3's survival penalty is 20–23%, its wrong-objective cost 0.17–0.26;
  - costs among R1/R2/R4 are ≤ 0.019, and VI-R3 is in its band on ≤ 1.5% of weeks.
  This turns "two patients" into "this region".
- **A2.** At 3M steps, full-state scores rise 0.87 → 0.94 and burden + critic 0.92 → 0.97. With full state, divergence grows (R1 vs R3: 3.8× → 6.1× seed noise) and R3's cost stays 0.154. At equal budget the observation ranking is burden + critic > full > memory (p ≤ 4·10⁻⁴). **F-e is confirmed, not overturned.** Its sharper form: a burden-only actor collapses the objectives (largest cost 0.023), because it cannot see the resistant fraction R3 rewards.
- **B2.**
  - Smooth entropy is *worse* than the band (VI 1387 d vs 1625 d).
  - Smooth targets make f\* matter (VI 1989 → 1443 d as f\* goes 0.05 → 0.5), yet no policy, VI included, holds f above ≈ 0.12. Structural, as F-d says.
  - R1-trained agents beat R3-trained agents on every coexistence reward.
  - New nuance: a target near the containment fraction (f\* = 0.05) is nearly free (cost 0.029).
- **E2.** On PoP, ε up to 0.5 lengthens TTP 2.4× but R3's penalty stays 18–20%. At turnover 0.04, ε ≥ 0.4 stops the patient progressing within the 30-year cap, which removes the control problem rather than helping R3. Defends F-d.
- **B4.** The ranking R1 = R4 > R2 > R3 holds at every threshold; R3's penalty grows with it (14% → 27%).

**New insight: what each adds to the paper**
- **B1.** Every good policy runs at ≈ 72% intensity, so TTP is nearly linear in cumulative drug. The E11d R2 agents trade ≈ 285 d for ≈ 4 points of intensity and lie on the intensity and resistance frontiers. R2's "failure" is a trade-off.
- **D1.** A half dose raises the R2/R3/R4 optima by 70–105 d, and agents gain 68–168 d. R1 never uses it, because its holiday bonus pays only at D = 0, so R1 ≢ R4 with three doses. A dose-proportional bonus (`R1p`) reproduces the R4 optimum *exactly*. **F-a holds once the holiday credit is proportional to the drug withheld.**
- **D2.** VI improves monotonically with levels (R4 2038 → 2143 → 2205 d), but PPO's scores fall from 3 to 5 levels. A single growth-halting dose (2/3) captures most of the planner's gain.
- **D3.** The size of the graded-dose gain depends on the dose–response (R4 +56 d at h = 2, +112 d at h = 0.5). State graded-dose results as conditional on the curve.
- **B3 + D4.** Optima and agents switch on ≈ 54% of weeks. A switching penalty halves that for 3–8% of the benefit. At 9 switches a year the VI-R1 schedule still beats AT-PSC by 55% (1755 vs 1135 d). Simple, clinically legible schedules exist.
- **E1.** At μ = 3·10⁻⁵ the optimum separates R4 from R1 for the first time: 0.444 vs 0.490 resistant fraction after a holiday, at 6 d less TTP. Agents do not (policy disagreement 1.04× seed noise, p = 0.18). The README's named route to durability ≠ survival exists but is weak at this rate.

**Recommended next step.** E3 (noisy or delayed burden readings) is now the most relevant missing item, because A2 made observation the strongest lever in the study.
