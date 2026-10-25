# Model flow

How data moves from the ODE to every number in the notebook. Code references point into `src/atrl.py`.

## 1. The tumour model (`deriv`, `rk4`, `advance`)

Strobl et al. 2021, in Gallagher et al. 2023's form:

$$
\frac{dS}{dt} = r_S\,S\left(1-\frac{S+R}{K}\right)\left(1-\hat d_D\,D\right) - d_T S,\qquad
\frac{dR}{dt} = r_R\,R\left(1-\frac{S+R}{K}\right) - d_T R
$$

| Symbol | Value (PoP patient) | Meaning |
|---|---|---|
| r_S | 0.027 /day | sensitive growth rate |
| r_R | r_S(1 − cost), cost = 0 | resistant growth rate |
| d_T | turnover · r_S, turnover = 0 | death rate, same for both populations |
| d̂_D | 1.5 | drug kill factor (Norton–Simon: kills dividing cells) |
| K | 1 | carrying capacity |
| N₀ | 0.75 K | initial burden |
| f_R | 0.001 | initial resistant fraction R₀/N₀ |
| progression | N ≥ 1.2 N₀ | checked at every 0.5-day RK4 substep |

E10 swaps the drug term (`drug=` argument):
- `cytostatic`: max(0, 1 − d̂_D·D)
- `logkill`: r_S·S·(1 − N/K) − κ·D·S, with κ = r_S(1 − n₀)·d̂_D

## 2. The MDP

| Element | Definition |
|---|---|
| State | (S, R), fully observed |
| Observation (`features`) | `full` (default): [N/N₀, S/K, R/K, log₁₀(R/N)/4 + 1]; `burden` (E11): [N/N₀] only; E11b: actor `burden`, critic `full` (training only); E11c: actor `burden_hist` = last 8 burden readings + last 8 doses, critic `full` |
| Action | D ∈ {0, 1}, held for one decision interval Δ (7 days; 1 or 28 in E9) |
| Transition (`advance`) | ⌈Δ/0.5⌉ RK4 substeps; stops at the substep where N crosses 1.2 N₀ (crossing time interpolated) |
| Termination | progression, or t ≥ 30 years (cap) |
| Reward (`reward`) | R1, R2, R3, R3-f*, R4, all computed from the **post-step** state, see `decisions.md` §B |
| Discount | γ_Δ = 0.999^(Δ/7) |

## 3. One environment step

```mermaid
sequenceDiagram
    participant P as Policy (PPO / VI / baseline)
    participant E as VecEnv / rollout
    participant O as ODE (advance)
    participant Rw as reward(obj)
    P->>E: D in {0,1} for every env in the batch
    E->>O: integrate Δ days at dose D (RK4, dt = 0.5 d)
    O-->>E: S', R', progressed?, time alive in step
    E->>Rw: (S', R', D, progressed, capped)
    Note over Rw: R4 only: simulate S', R' untreated 180 d -> ĝ
    Rw-->>E: r
    E-->>P: features(S', R'), r, done
    Note over E: done envs are reset to a freshly sampled patient (Fixed or Cohort sampler)
```

## 4. Pipeline: from environment to claims

```mermaid
flowchart LR
    subgraph ENV[Environment]
        ODE[Strobl LV ODE<br/>advance] --> MDP[MDP<br/>features / reward]
    end
    subgraph OBJ[Objectives]
        R1[R1 survival+holiday]
        R2[R2 dose+time]
        R3[R3 coexistence]
        R4[R4 durability]
        R3f[R3-f* sweep]
    end
    MDP --> OBJ
    OBJ --> VI[solve_vi<br/>policy iteration<br/>per objective]
    OBJ --> PPO[train<br/>PPO x seeds]
    BL[baselines<br/>None MTD AT-x IT AT-PSC] --> EV
    VI -->|Lookahead policy| EV[rollout<br/>every policy x every metric<br/>x every objective return]
    PPO -->|Agent policy| EV
    VI -->|ceiling G_VI| SC[score = G - G_none<br/>over G_VI - G_none]
    EV --> SC
    SC --> X[E4 cross-evaluation<br/>score matrix, loss table]
    EV --> DV[E5 policy space<br/>sigmoid N*, KL, disagreement<br/>permutation test vs seed noise]
    EV --> RB[E6-E10 robustness<br/>f*, cohort, perturbation,<br/>cadence, transfer]
```

## 5. Notebook sections and the files they produce

```mermaid
flowchart TD
    S0[Setup: CFG, helpers] --> E0
    E0[E0 reproduce CT 451.6 / AT50 662.8] --> E1
    E1[E1 baselines + cost x turnover map<br/>-> BEST_AT, BEST_H, FLOOR] --> E2
    E2[E2 VI per objective<br/>-> CEIL, VIP] --> E3
    E3[E3 PPO R1-R4 x 10 seeds<br/>results/agents/E3_*.pkl] --> E4
    E4[E4 cross-evaluation<br/>results/E4_*.csv] --> E5[E5 policy divergence]
    E3 --> E6[E6 f* sweep<br/>results/agents/E6_*.pkl]
    E3 --> E7[E7 cohort generalists<br/>results/agents/E7_*.pkl]
    E3 --> E8[E8 perturbation]
    E3 --> E9[E9 cadence<br/>results/agents/E9_*.pkl]
    E3 --> E10[E10 drug-model transfer]
    E7 --> E10
    E4 & E5 -->|same functions,<br/>cost 0.42 patient| EC[E3c-E5c replication<br/>results/agents/E3c_*.pkl<br/>results/E4c_*.csv]
    E4 & E5 -->|same functions,<br/>burden-only observation| E11[E11 burden-only agents<br/>results/agents/E11_*.pkl<br/>results/E11_*.csv]
    E11 -->|diagnosis: critic aliasing| E11b[E11b burden-only actor,<br/>full-state critic<br/>results/agents/E11b_*.pkl]
    E11b -->|+ memory| E11c[E11c burden-history actor,<br/>full-state critic<br/>results/agents/E11c_*.pkl]
    E4 & E5 & E6 & E7 & E8 & E9 & E10 & EC & E11 & E11b & E11c --> SUM[results/summary.json -> README]
```

Every figure is written to `figures/<experiment>_*.png` by the cell that draws it.

## 6. What is shared, and what is not

- **Shared by all objectives:** the environment, the integrator, the observation, the action space, PPO and its hyperparameters, the step budget, the seeds and the evaluation code. This is the Phase 2 rule that keeps the off-diagonal meaningful.
- **Differs per objective:** only `reward(obj, ...)`, and therefore each objective's VI ceiling.
- **Not shared, deliberately:** the VI planner's model in E8 and E10. It plans with the *nominal* cytotoxic model while the true patient is perturbed or uses a different drug mechanism. That model mismatch is what those experiments test.
