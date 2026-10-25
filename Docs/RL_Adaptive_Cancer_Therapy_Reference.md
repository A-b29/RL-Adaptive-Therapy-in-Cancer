# Evolution-Aware Reinforcement Learning for Adaptive Cancer Therapy Under Drug Resistance
## Literature Reference & Working Model Specification

**Compiled:** 13 September 2026 · **Source:** all 39 PDFs in `Papers/papers/` · **For:** Arnav Kumar (MD24B003), AI in Medicine, Sem 5

---

## 0. How to use this document

Part I states the seven ideas your project rests on — if you can defend these, you can defend the project. Part II is one entry per paper, with the actual equations, parameter values, RL hyperparameters and headline numbers, so you can cite specifics rather than gestures. Part III collapses all 39 papers into comparison tables (models, algorithms, state/action/reward, baselines, metrics). Part IV is a concrete build spec — environment, MDP, algorithm, baselines, metrics, experiment list — assembled from the values in Part II. Part V is the honesty section: what this literature does *not* support, and the critiques your report must acknowledge.

### 0.1 File integrity — three problems in the downloaded corpus

I verified every PDF's title against its filename. Three do not match, and one paper is missing. **Fix these before you cite anything.**

| Filename | What it actually is | Action |
|---|---|---|
| `01_gatenby2009_adaptive_therapy.pdf` | Gatenby & Brown, *The Evolution and Ecology of Resistance in Cancer Therapy*, **Cold Spring Harb Perspect Med 2020;10:a040972** — a review, not the 2009 paper | Re-download Gatenby, Silva, Gillies & Frieden, *Adaptive Therapy*, Cancer Res 2009;69(11):4894–4903, doi:10.1158/0008-5472.CAN-08-3658. The review is still useful — keep both. |
| `06_viossat2021_tumour_containment.pdf` | Strobl, Gallaher, Robertson-Tessi, West & Anderson, *Treatment of evolving cancers will require dynamic decision support*, **Ann Oncol 2023;34(10):867–884**, doi:10.1016/j.annonc.2023.08.008 | Re-download Viossat & Noble, *A theoretical analysis of tumour containment*, Nat Ecol Evol 2021;5(6):826–835, doi:10.1038/s41559-021-01428-w. This one matters — it is the formal containment-vs-elimination result. |
| `12a_cunningham2018_optimal_control_mcrpc.pdf` | **Byte-identical to 12b** (same MD5). Both are Cunningham et al. 2020, PLOS ONE 15(12):e0243386 | Cunningham et al. 2018, *J Theor Biol* 459:67–78 (doi:10.1016/j.jtbi.2018.09.022) was never downloaded. |

The manifest's `downloaded` column is unreliable — it reads `FALSE` for several files that are present and correct (07, 13, 19, 21, 22, 23, 24, 25, 26, 30, 30b, 31, 32, 33, 33b all extracted fine). `Papers/fetch_papers.py` appears to have silently substituted open-access alternatives when a DOI resolved to a paywall. **Net: 38 unique papers on disk, 2 wrong, 2 missing.**

Text extraction succeeded on all 39 files (no scanned/OCR-needed PDFs); page counts ranged 2–105, character counts 8.8k–217k.

---

## Part I — The seven ideas the project rests on

**1. Resistance emerges in two steps, and only the second one is controllable.**
Acquiring a resistance mechanism is near-inevitable given tumour population sizes and genomic diversity; *proliferating to clinical dominance* is not. Targeting the molecular machinery of resistance fails because one mechanism is simply replaced by another. Targeting the *ecology* — the Darwinian conditions under which resistant cells expand — is the opening. [Gatenby & Brown 2020 review; Gatenby et al. 2009]

**2. Competitive release is the enemy, not the tumour's size.**
MTD maximises cell kill, which maximally removes the sensitive population — the resistant clone's only competitor. The resistant clone then expands unopposed. This is the mechanism, observed directly in MCF7/MCF7-Dox co-culture and in MDA-MB-231 mouse models. [Enriquez-Navas 2016; Gatenby & Brown 2020]

**3. Adaptive therapy trades cure for control, deliberately.**
Therapy is cycled off *before* sensitive cells are driven to extinction, so they can resume suppressing the resistant population during the holiday. Tumour burden is deliberately maintained at a controllable level. The endpoint is time to progression (TTP), not response rate. [Gatenby et al. 2009; Zhang 2017]

**4. Adaptive therapy does not require a cost of resistance — but it does require turnover.**
This is the single most useful modelling result for your project. Strobl et al. showed with a 2-population Lotka–Volterra model that when a tumour sits near carrying capacity, competition alone makes AT beat continuous therapy even with `rR = rS` (no cost). Far from carrying capacity, a cost is needed. And cell turnover `dT` modulates everything: high turnover makes AT beneficial even with zero cost; above a threshold of `(1 − 1.2·n0)·r̂R`, the tumour becomes indefinitely controllable in the model. Fits to 67 IADT patients recovered a **negative correlation of 0.77 between fitted cost and fitted turnover** — a trade-off that was not imposed. [Strobl 2021]

**5. The AT50 rule used in the clinic is a clinically convenient heuristic, not an optimum.**
"Treat until burden falls 50%, hold until it returns to baseline" was a compromise between model output and what an oncologist could actually execute. Optimal-control analyses (HJB, Pontryagin) show optimal policies are typically **bang-bang** — full dose or nothing — but with switching boundaries that depend on the full state, not on one threshold. And higher withdrawal thresholds (treat less, hold at higher burden) generally extend TTP further. [Gluzman 2020; Viossat via West 2023 §2.2; Gallagher 2024; Ma 2026]

**6. RL's value here is not beating optimal control on a known model — it is not needing the model.**
Where dynamics and parameters are exactly known, optimal control wins or ties. RL wins under **parametric uncertainty, stochasticity, partial observability, and model misspecification** — precisely the clinical situation. Tordesillas showed DQN/DDPG matched OC on nominal parameters and beat it under perturbation; Eastman showed RL schedules held a median scaled score of 0.991 (sd 0.01) across 600 perturbed virtual patients while the optimal controller degraded to 0.916 (sd 0.14). [Tordesillas 2019; Eastman 2021]

**7. The honest failure mode: RL agents learn the *environment*, not the *disease*.**
Aif et al. trained PPO agents that reached TTF of 149 ± 19 generations in a 2-ODE training environment — a near-fivefold improvement over AT50's 33.8 ± 0.8 — and then watched that ranking collapse when the same policies were evaluated in a spatially explicit PhysiCell agent-based model. High average tumour burden, the conventional driver of benefit, correlated r = 0.91 with TTF in the ODE world but only r = 0.77 in the spatial one. **Any claim your project makes is a claim about your environment.** Plan a transfer test from day one.

---

## Part II — Per-paper reference entries

Entries give: full citation · system & model · method · key parameters/equations · headline numbers · limitations · **what to take for this project**.

---

# Section 1 — Evolutionary foundations

### [01] Gatenby & Brown (2020) — *The Evolution and Ecology of Resistance in Cancer Therapy*
*Cold Spring Harb Perspect Med 10:a040972.* ⚠️ **This is the file on disk; the manifest expected Gatenby et al. 2009 (Cancer Res 69:4894–4903, doi:10.1158/0008-5472.CAN-08-3658), which is not present.**

- **Type:** Conceptual review — the framing paper for the whole field.
- **Core argument:** Two steps to clinical resistance — (i) acquiring the mechanism, (ii) proliferating to dominance. Step (i) is essentially inevitable; step (ii) is governed by Darwinian dynamics and is where therapy can intervene. Molecular targeting of resistance fails because the human genome offers alternative mechanisms.
- **Three evolution-informed strategies:** adaptive therapy (cycle to maintain sensitive cells), **double-bind therapy** (a second agent that exploits the resistance phenotype's vulnerability), **extinction therapy** (drive small populations to true extinction using Allee effects).
- **Conditions under which AT beats MTD (stated explicitly):** (1) therapy is highly effective against sensitive cells; (2) sensitive cells outcompete resistant cells off-treatment; (3) sensitive cells proliferate faster off-treatment than resistant cells do on-treatment. Benefit increases with **cell turnover rate** and **cost of resistance** — explicitly citing Strobl.
- **Key nuance:** "Even without a cost of resistance, simply having competition between cell types can render AT superior to continuous therapy, but therapy failure becomes inevitable."
- **Evidence cited:** MCF7 (sensitive) vs MCF7/Dox (resistant) co-culture without doxorubicin — the resistant phenotype's metabolic cost let sensitive cells outgrow it. Verapamil-resistance cost in breast cancer *in vitro* (Silva 2012) → mouse MDA-MB-231/MCF-7 paclitaxel trials.
- **Take for the project:** Use this for your Introduction's framing and for the three-condition list — it gives you a citable, non-hand-wavy statement of *when* adaptive therapy should work, which is exactly what your RL agent will be probing.

---

### [02] Enriquez-Navas et al. (2016) — *Exploiting evolutionary principles to prolong tumor control in preclinical models of breast cancer*
*Sci Transl Med 8(327):327ra24.* doi:10.1126/scitranslmed.aad7842

- **System:** MDA-MB-231/luc (triple-negative) and MCF7 (ER+) breast cancer, orthotopic in mouse mammary fat pad; paclitaxel; tumour volume by MRI **twice weekly**.
- **The two algorithms — copy these exactly, they are the canonical dose-modulation vs dose-skipping pair:**
  - **AT-1 (dose modulation):** start 20 mg/kg. Then per measurement: if volume **fell ≥20%** → **halve** the previous dose; if volume **rose ≥20%** → **increase dose by 50%** of previous (cap 20 mg/kg); if within ±20% → **repeat** the same dose. Skip treatment entirely below a **150 mm³** volume cutoff.
  - **AT-2 (dose skipping):** start 15 mg/kg. If volume rose ≥25% relative to **two measurements prior (Tₙ vs Tₙ₋₂)** → give a fixed 15 mg/kg. Otherwise skip. No minimum cutoff.
  - **ST (standard):** high-dose paclitaxel, conventional schedule.
- **Results (MDA-MB-231, progression = 1000 mm³):** control median PFS **75.5 d**; ST **108.5 d** (p=0.3835 vs control); AT-2 **109 d** (p=0.1573); **AT-1 significantly better (p=0.0006)** — only 2 of 12 (16%) progressed within the 155-day trial. AT-1 vs ST: p=0.9163 for AT-2, AT-1 clearly separates.
- **Counterintuitive and important:** cumulative paclitaxel in AT-1 **exceeded** both AT-2 and ST — because the animals lived longer, and because *initial* control of an exponentially growing tumour required aggressive dosing to bend the curve onto a plateau. Only **after** the plateau could doses be reduced, and then continuously so — in 3/5 MCF7 and 6/7 MDA-MB-231 tumours, burden fell below the treatment threshold and therapy was withheld for >8 consecutive decision points.
- **The two-phase structure is the lesson:** (Phase 1) intensive induction to reach a plateau — dose-skipping is *worse* here than dose-modulation; (Phase 2) maintenance at progressively smaller doses, often zero.
- **Take for the project:** Gives you a biologically grounded reason to let your agent use *graded* doses, not just on/off, and a reason to expect the learned policy to be **non-stationary** (aggressive early, sparse late). Also gives you a real baseline pair (AT-1/AT-2) that is not AT50.

---

### [03] Zhang, Cunningham, Brown & Gatenby (2017) — *Integrating evolutionary dynamics into treatment of mCRPC*
*Nat Commun 8:1816.* doi:10.1038/s41467-017-01968-5 — **The trial that started it all. Read this one in full.**

- **Model — 3-species Lotka–Volterra competition:**
  ```
  dxᵢ/dt = rᵢ · xᵢ · (1 − (Σⱼ αᵢⱼ xⱼ) / Kᵢ)
  ```
  with cell types **T+** (androgen-dependent), **TP** (testosterone-producing, expresses CYP17A), **T−** (androgen-independent = abiraterone-resistant).
- **Parameters (use these — they are the most-reused numbers in the field):**
  - Growth rates from measured doubling times of LNCaP (60 h), H295R (48 h), PC-3 (25 h) → `r = [0.278, 0.355, 0.665] /day`, then **scaled to 10%** of these for *in vivo* plausibility. Growth rates do **not** affect equilibrium frequencies, only the speed of dynamics.
  - Competition: `αᵢᵢ = 1`; `0 < αᵢⱼ < 1` for i≠j. Six inequality constraints (α₃₁>α₂₁, α₃₂>α₁₂, α₁₃>α₂₃, α₁₃>α₁₂, α₂₃>α₂₁, α₃₂>α₃₁) admit **22 valid rank orderings**; values drawn from {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}.
  - Carrying capacities: `K_TP = 10,000`, dropping to **100** under abiraterone (drug modelled as reducing carrying capacity, since cytotoxic-vs-cytostatic is unknown). `K_T+ = 1.5·x_TP` off-drug ("cheating" on TP-produced testosterone), falling to `0.5·x_TP` on-drug.
  - **PSA surrogate:** `dPSA/dt = Σᵢ xᵢ − 0.5·PSA` (each cell makes 1 unit/time; 50% serum decay per step).
  - **Initial condition:** cell-type frequencies at the untreated LV equilibrium (the ESS); total population and PSA at **25%** of untreated equilibrium; simulate until PSA hits **80%** of equilibrium = progression to mCRPC = trigger to start abiraterone.
  - Representative competition matrices are given for Patient #1 and #2 (Table 3) — e.g. Patient #1: rows T+ (1, 0.7, 0.8), TP (0.4, 1, 0.5), T− (0.6, 0.9, 1).
- **The 22 matrices sort into three phenotypes:** 12 "best responders" (no T− at equilibrium, sustained PSA cycling under AT), 4 "responders" (progression still happens but TTP greatly extended), 6 "non-responders" (T− ≥20% at equilibrium, no 50% PSA drop — these patients were ethically excluded from the trial).
- **Clinical protocol (NCT02415621):** abiraterone + prednisone; stop when PSA < 50% of the pre-treatment value; restart when PSA returns to that pre-treatment value. Primary endpoint: PSA response after two complete adaptive cycles.
- **Results (11 patients, interim):** 10/11 maintained stable PSA oscillation; cycle lengths **3 months to >1 year**; median radiographic TTP **≥27 months** vs **16.5 months** in the Phase III COU-AA-302 comparator (Ryan et al.); **cumulative dose 47% of SOC**, with 3/11 patients under 25%. Fisher exact p=0.006 for PSA progression; Poisson exact p<0.001 (z=4.10) for radiographic TTP. In the contemporaneous cohort, 14/16 progressed vs 1/11 on AT.
- **Mechanistic insight to quote:** cycle length is set by the **T+ population size** — a large T+ pool makes PSA rebound fast (short cycles), a small one makes long cycles. This is *why* fixed intermittent schedules (SWOG 9346) failed: they were not synchronised to patient-specific evolutionary dynamics.
- **Take for the project:** This is your reference for parameter values, for the PSA observation model (a partially observable biomarker — relevant if you do a POMDP variant), and for the baseline policy definition (AT50). Code: github.com/cunninghamjj/Integrating-evolutionary-dynamics-into-treatment-of-mCRPC

---

### [04] Zhang, Cunningham, Brown & Gatenby (2022) — *Evolution-based mathematical models significantly prolong response to abiraterone in mCRPC*
*eLife 11:e76284.* doi:10.7554/eLife.76284 — **The mature trial readout. Use these numbers, not the 2017 interim ones.**

- **Cohorts:** 17 adaptive therapy vs 16 contemporaneous SOC (all had >50% PSA decline and met trial eligibility but chose SOC).
- **Headline results:**
  | Metric | Adaptive | SOC | Statistic |
  |---|---|---|---|
  | Median TTP (radiographic) | **33.5 months** | 14.3 months | p<0.001 |
  | Median overall survival | **58.5 months** | 31.3 months | HR **0.41** (95% CI 0.20–0.83), p<0.001 |
  | Time off drug | **46%** of time on trial | 0% | dosing rate 54% of SOC |
  | Still on study, no progression | 4 patients (53–70 months) | 0 | all 16 SOC progressed and died |
- **Parameter estimation (the numbers you should adopt):** growth rates estimated from exponential fits to PSA rises **during treatment holidays** (assumed sensitive-driven): **rS = 0.0156/day, rR = 0.0091/day** (t = 2.11, p<0.05, 13 patients, 1–6 estimates each). Constrained nonlinear least-squares over the whole cohort gave a global minimum at **α_RS = 6** (competition coefficient of sensitive on resistant) and **β_SC = 8** (a PSA-dynamics scaling multiplier, searched over [0, 20]).
- **The α_RS sensitivity result — this is the cleanest illustration of the mechanism in the entire literature:** with ideal cycling (stop at exactly 50%, restart at exactly baseline),
  - α_RS = 0.8 → sensitive regrowth does **not** suppress xR → **AT fails**
  - α_RS = 2 → regrowth slows xR → **treatment failure delayed**
  - α_RS = 6 (the fitted value) → xR has a **negative growth rate** during sensitive regrowth → over 3–4 cycles xR → 0 → **indefinite control**
- **Also:** TTP declines with the estimated pre-treatment resistant fraction, for both arms — but no AT patient fell below the CT regression line and no CT patient above the AT line.
- **Limitation to state:** non-randomised, single-institution, small, with a selection criterion (>50% PSA decline) that the comparator group shares but the Phase III historical control does not.
- **Take for the project:** α_RS = 6 with rS = 0.0156, rR = 0.0091 is a *defensible, literature-fitted* parameter set for a "responder" virtual patient. The α_RS sweep {0.8, 2, 6} is a ready-made robustness experiment for your agent. Code: github.com/cunninghamjj/Evolution-based-mathematical-models-...-in-mCRPC (includes `TrialPatientData.xlsx`).

---

### [05a] Mistry (2021) — *On the reporting and analysis of a cancer evolutionary adaptive dosing trial*
*Nat Commun 12:316.* doi:10.1038/s41467-020-20174-4 — **The critique. Your report must engage with this.**

Three specific charges against Zhang 2017:
1. **No prognostic factors reported** for either the 11 AT patients or the 16 contemporaneous controls — LDH, alkaline phosphatase, haemoglobin are all established abiraterone prognostics and were available in the FDA statistical review. Without them, a prognostic imbalance cannot be excluded as the explanation for the TTP difference.
2. **The Phase III comparison is invalid by construction:** AT patients were only *eligible* if they had already experienced a >50% PSA fall; COU-AA-302 enrolled abiraterone-naive patients. These are not the same population.
3. **No information at all** on the contemporaneous cohort's characteristics.

**Conclusion as written:** "the evidence that adaptive dosing is superior to continuous dosing still does not exist, and so patients should remain on continuous dosing until a well-conducted randomized-control trial is performed."

**Take for the project:** This is the paragraph that keeps your report intellectually honest. It also *strengthens* the case for in-silico work: if the clinical evidence is contested, controlled computational comparison under matched conditions is exactly the right instrument — which is your project's stated rationale.

---

### [05b] Zhang, Cunningham, Brown & Gatenby (2021) — *Response to Mistry*
*Nat Commun 12:329.* doi:10.1038/s41467-020-20175-3

- **Concedes** the controls are imperfect, the trial was an explicitly non-randomised proof-of-concept pilot, and that they raised the same questions themselves.
- **Rebuts with matching data:** Gleason scores matched across arms (Kruskal–Wallis statistic 0.088, p=0.767, df=1); baseline pre-abiraterone PSA matched (KW 0.157, p=0.692); metastasis locations well matched; liver metastases (poor prognosis) excluded from the Phase III trial *and* from both of their cohorts; Hgb >10 in both. LDH was not routinely collected in the contemporaneous cohort — **this concession stands**.
- **Notes their analysis was deliberately conservative:** they assumed every non-progressed AT patient progressed at the time of writing (only 2 of 11 actually had).
- **Three years on:** all contemporaneous patients had progressed; four remained on AT.
- **Take:** Cite 05a and 05b together, always. The exchange is a good model for how to write your own Limitations section.

---

### [06] Strobl, Gallaher, Robertson-Tessi, West & Anderson (2023) — *Treatment of evolving cancers will require dynamic decision support*
*Ann Oncol 34(10):867–884.* doi:10.1016/j.annonc.2023.08.008
⚠️ **This is the file on disk. Viossat & Noble (2021), *A theoretical analysis of tumour containment*, Nat Ecol Evol 5:826–835, doi:10.1038/s41559-021-01428-w — the paper the manifest wanted — is missing and should be obtained.**

- **Type:** Perspective. Traces the history of dose scheduling and proposes a framework.
- **The historical arc (useful for your Introduction):** Skipper et al.'s leukemic-mouse experiments — a single high dose beat a daily low dose *despite a lower cumulative dose* — because the tumours were homogeneous and drug-sensitive, so a given dose always killed a constant fraction. That is the **log-kill law**, and MTD follows from it. MTD is therefore correct *for the disease Skipper studied* and wrong for heterogeneous, evolving metastatic disease. Schedule choice "crucially depends on processes driving treatment response and failure."
- **The ADAPT paradigm** (Adaptive Dosing Adjusted for Personalized Tumorscapes): a five-step loop of data collection → model calibration → integration of multi-source/multi-scale data → prediction → schedule adjustment. Each patient is mapped to their "tumorscape".
- **Reports the prostate trial as:** 46% reduction in drug use and a **19.2-month increase in TTP** vs historical control.
- **Also flags:** model-informed precision dosing for toxicity (Friberg-type myelosuppression models), Meille et al.'s Bayesian first-cycle calibration for docetaxel+epirubicin, melanoma AT trial NCT03543969, and Moffitt's **Evolutionary Tumor Board** (NCT04343365, 21 patients).
- **Take:** Use for the "why schedules matter historically" framing and for the argument that a model + a feedback loop, not a fixed rule, is the deliverable. The ADAPT loop maps almost one-to-one onto an RL loop — worth a sentence in your Discussion.

---

### [07] Strobl et al. (2021) — *Turnover modulates the need for a cost of resistance in adaptive therapy*
*Cancer Res 81(4):1135–1147.* doi:10.1158/0008-5472.CAN-20-0806 — **⭐ The model you should build your environment on.**

- **Model — 2-population Lotka–Volterra with turnover and Norton–Simon drug kill:**
  ```
  dS/dt = rS · S · (1 − (S+R)/K) · (1 − 2·dD·D(t)/Dmax) − dT·S
  dR/dt = rR · R · (1 − (R+S)/K)                        − dT·R
  N(t) = S(t) + R(t)
  ```
- **Assumptions, each stated and each worth knowing:**
  - Logistic growth; dividing fraction falls linearly to 0 at shared carrying capacity K.
  - Density-independent death at rate `dT`, **identical for both populations** (this is "turnover").
  - **Inter- and intra-specific competition coefficients are both set to 1** — deliberately the simplest case.
  - **Norton–Simon drug effect:** only dividing cells are killed; kill scales linearly with dose; the **factor of 2** accounts for losing both mother and daughter when a cell dies in mitosis.
  - Resistance is **pre-existing**, mutation during treatment neglected (justified by Viossat & Noble showing mutation doesn't change AT benefit much in simple models).
  - Cost of resistance enters **only through rR < rS**.
- **Parameters:** `rS = 0.027 /day` (from Zhang 2017) as the time scale; `d̂D = 1.5`; sweeps over `r̂R ∈ [0,1]`, `d̂T ∈ [0,1]`, `n0 ∈ [0.1,1]`, `fR ∈ [10⁻⁵, 0.25]`. Non-dimensionalised with rS as time scale and K as density scale.
- **Protocols compared:** CT (`D = Dmax` always) vs **AT50** (`Dmax` until N < 0.5·N₀; 0 until N = N₀). **Progression = 20% growth over baseline (RECIST-analogous).**
- **Four findings, in order of usefulness to you:**
  1. **AT extends TTP even with zero cost of resistance**, provided the tumour is near carrying capacity and pre-existing resistance is low. Competition alone suffices.
  2. AT works by **minimising the resistant population's growth rate**, not by minimising total burden.
  3. Without turnover, a cost of resistance only helps when resistant cells are initially rare.
  4. **Turnover is the master variable.** Low turnover → even a large cost yields small gains. High turnover → large benefit even with zero cost. Above `dT > (1 − 1.2·n0)·r̂R`, the tumour is **indefinitely controllable** in the model (both CT and AT hold it below progression forever).
- **Clinical validation:** fitted to 67 patients from the Bruchovsky IADT trial (15 excluded for r² < 50% or failure to capture late cycles). Levenberg–Marquardt via `lmfit`, 25 restarts per patient, AIC selection. Setting cost or turnover (or both) to zero significantly degraded fit. **Fitted cost and turnover are strongly negatively correlated (Pearson 0.77)** — an emergent trade-off, not an imposed one. No patient sits in the low-cost/low-turnover corner (they wouldn't have qualified) or the high-turnover/high-cost corner (they'd never have relapsed).
- **Bonus — three strategy ideas tested:** (1) AT with 50% withdrawal; (2) a secondary drug that **reduces K** by 50%; (3) a secondary drug that **increases turnover** by 20%. Strategy 3 is an underexplored action your agent could have.
- **Code:** github.com/MathOnco/AT_costOfResistance_LVModel
- **Take for the project:** Adopt this as your environment, verbatim. It is the model Gallagher's DRL papers use, the model Ma 2026's RL paper uses, and the model Aif 2025 calibrates against. Using it makes your results directly comparable to four other RL papers. **Cost of resistance `1 − rR/rS` and turnover `dT/rS` become your two-axis virtual-patient space.**

---

### [08] Kaznatcheev, Peacock, Basanta, Marusyk & Scott (2019) — *Fibroblasts and alectinib switch the evolutionary games played by NSCLC*
*Nat Ecol Evol 3(3):450–456.* doi:10.1038/s41559-018-0768-z

- **What it does:** Introduces a **game assay** — an experimental protocol that *measures* the evolutionary game a cancer population plays, rather than assuming a payoff matrix.
- **System:** ALK+ NSCLC line H3122 (parental, alectinib-sensitive) vs a derived alectinib-resistant line, ± cancer-associated fibroblasts (CAFs), ± alectinib. Four conditions: DMSO, DMSO+CAF, Alectinib, Alectinib+CAF.
- **Method:** Co-culture at a range of initial parental proportions `p`; measure each subpopulation's growth rate as a function of `p`; fit fitness functions `ŵP(p)`, `ŵR(p)`. For linear fits, the p=0 and p=1 intercepts **are** the payoff matrix entries: `ŵP = A·p + B·(1−p)`, `ŵR = C·p + D·(1−p)`. The game is then located in a 2-D game space by `x := C − A`, `y := B − D`.
- **Headline result:** the game is not fixed. Adding alectinib and CAFs **switches the game type from Leader to Deadlock** — a *qualitative* change, not just a quantitative one. "We can treat not only the player, but also the game."
- **Rigour worth noting:** 42 data points per model, 8 models; AIC/AICc/BIC model selection over polynomial degrees up to 6; AICc and BIC both select a **cubic** fitness function over the linear one, so the linear game is a convenient approximation, not the truth. Errors propagated from individual wells, not averaged across replicates.
- **Limitation:** *in vitro*, two cell lines, one drug.
- **Take for the project:** Two uses. (1) It justifies treating competition coefficients as **environment- and treatment-dependent** — a strong argument for an agent that doesn't assume fixed dynamics. (2) A stretch goal: make `α` a function of whether the drug is on, and see whether your agent still works. Also the honest citation for "competition coefficients between cancer cells are measurable, and are not always <1."

---

### [09] West, Adler, Gallaher, Strobl, Brady-Nicholls, Brown, Robertson-Tessi, Kim, Noble, Viossat, Basanta & Anderson (2023) — *A survey of open questions in adaptive therapy*
*eLife 12:e84263* (arXiv:2210.12062) — **Read §2.2 before you finalise your action space.**

Eleven open questions from the CATMo 2020 workshop, in three groups. The ones that bear on your project:

- **§Protocol taxonomy — use these exact definitions in your report:**
  1. **MTD** — periodic high dose, limited by toxicity.
  2. **Intermittent therapy** — high dose with *fixed, periodic* holidays.
  3. **AT (dose skipping)** — high dose until a response threshold (e.g. 50% reduction), then a holiday until an upper threshold (e.g. 100%), repeat.
  4. **AT (dose modulation)** — dose increased/decreased at regular intervals per response.
- **§1.1 Necessary conditions for AT to delay resistance** — competition must be strong enough; the **fitness of the resistant population when rare** is a key determinant; agent-based work (Gallaher) adds **spatial mixing rate** and **mutation rate** as determinants.
- **§2.2 What is the optimal protocol? (Viossat)** — the most useful section for you:
  - Models where competition scales with tumour size ⇒ **maintain the tumour at the maximal acceptable size**, which may mean *delaying* initial treatment if the tumour is small.
  - TTP can be extended by switching to high dose shortly before containment fails — but this risks making the tumour less treatable afterwards.
  - If resistant cells are only **partially** resistant, the sensible strategy is: stabilise to exploit competition first, then switch to MTD *long before* stabilisation would fail.
  - Cunningham's finding: **upward dose titration beats dose reduction**, because starting high immediately selects for resistance.
  - **Weighting dose by tumour responsiveness beats fixed ±10% modulation**; and the next dose should depend on both the recent *change* and the *absolute* size versus a target.
  - Caveat: some models with size-proportional death rates find **MTD superior** to dose-skipping AT.
  - ABMs sometimes find dose-skipping beats dose-modulation — opposite to the mouse data. Spatial structure can *increase* the cost of resistance by trapping resistant cells in the tumour core.
- **§3.2 Risks of maintaining high burden:** more mutation opportunity, possible evolution of traits that *reverse* the cost of resistance, more metastasis, worse quality of life. "Models of adaptive therapy have not considered the risk of additional progressive evolution beyond that expected regarding drug sensitivity." Also: trials measure *total* burden while disease is distributed across lesions that need not respond proportionately.
- **§3.1 Real-time prediction (Brady-Nicholls):** before predicting, a model must be shown **identifiable and sensitivity-analysed** on the available data — correlated parameters should not be estimated jointly. Worked example predicts resistance per adaptive cycle for prostate patients with cycle-specific cutoffs.
- **Take for the project:** Your Discussion writes itself from §2.2 and §3.2. Also: the §3.1 identifiability warning applies directly if you ever fit your LV model to data.

---

# Section 2 — Optimal control and dynamic programming

*Read this section as "what the right answer looks like when you know the model exactly." Your RL agent should be benchmarked against these, and where it matches them without being told the model, that is your result.*

### [10] Gluzman, Scott & Vladimirsky (2020) — *Optimizing adaptive cancer therapy: dynamic programming and evolutionary game theory*
*Proc R Soc B 287:20192454.* doi:10.1098/rspb.2019.2454 (arXiv:1812.01805) — **⭐ The bang-bang result.**

- **Model:** replicator dynamics over three subpopulations from Kaznatcheev's public-goods game — **GLY** (glycolytic, anaerobic, lactic-acid producing), **VOP** (vascular overproducer, spends energy on VEGF), **DEF** (defector, aerobic, free-rides on both). `ẋᵢ = xᵢ(ψᵢ − ⟨ψ⟩)`, `Σxᵢ = 1`.
- **Reduction to 2-D:** `q = xV/(xV+xD)`, `p = xG`. Controlled system:
  ```
  q̇ = q(1−q)[ (bv/(n+1))·Σₖ pᵏ − c ]
  ṗ = p(1−p)[ ba/(n+1) − (bv − c)·q − d(t) ]
  ```
  with `d(t) ∈ [0, dmax]` a GLY-targeting therapy intensity.
- **Boundaries:** patient **recovers** when `xG < rb` (recovery barrier); **dies** when `xV + xD < fb` (failure barrier). Untreated, the system is **cyclic** — the patient never recovers.
- **Objective and why its naive form fails:** minimising total drug `D = ∫d dt` alone is degenerate (d≡0 gives D=0 and T=∞). Restricting to recovering policies gives no minimiser — there's a sequence of ever-shorter drug bursts with unboundedly long treatment times. **Fix: add a time penalty σ.**
  ```
  minimise   ∫₀ᵀ d(t) dt + σ·T     (= +∞ if the patient does not recover)
  ```
  Solved as a **Hamilton–Jacobi–Bellman** equation for the value function u.
- **⭐ Central structural result: virtually all optimal policies are BANG-BANG** — at every instant either `d = dmax` or `d = 0`. This means the optimal policy is fully described by partitioning state space into a "drugs-on" region and a "drugs-off" region — which is both clinically implementable ("on or off") and directly comparable to a binary-action RL agent.
- **Parameters:** `ba = 2.5, bv = 2, c = 1, n = 4`, `dmax = 3`, `σ = 0.01`, `rb = fb = 10^−1.5` (relaxed from Kaznatcheev's 10⁻⁴ for mesh cost), initial state `(q₀,p₀) = (0.6, 0.9)` ≡ `(xD,xG,xV) = (0.04, 0.9, 0.06)`.
- **Results:** (a) The optimal policy dramatically reduces total drug versus MTD-until-recovery, which is wasteful. (b) **"Incurable" states:** if `ba/(n+1) > dmax`, MTD may never reach recovery — the MTD trajectory becomes **periodic**, cost +∞. With `ba = 4, dmax = 0.3, σ = 0.03`, neither "no drugs" nor "always MTD" reaches recovery, **but their adaptive combination does.** With a larger failure zone, MTD leads to death where the adaptive policy still recovers. (c) The incurable region under adaptive policies is far smaller than under MTD (Fig 7), and grows with larger `dmax` or smaller `bv`. (d) The σ/dmax ratio traces a Pareto front between total drug and total time.
- **Take for the project:** (1) Bang-bang optimality is your justification for a **binary action space** — you are not simplifying, you are matching the known optimum. (2) The "MTD cannot cure states that adaptive can" result is the strongest possible framing for your motive. (3) The objective `∫d dt + σT` is a clean, defensible **reward structure** — negate it: `r = −d(t)·Δt − σ·Δt`, plus a large terminal bonus on recovery and penalty on failure.

---

### [11] Wang, Scott & Vladimirsky (2024) — *Threshold-awareness in adaptive cancer therapy*
*PLOS Comput Biol 20(6):e1012165.* doi:10.1371/journal.pcbi.1012165

- **The problem it fixes:** in a stochastic world, any policy's cost is a *random variable*. A "deterministic-optimal" policy minimises the *expected* cost, which can still frequently blow past a clinically meaningful budget. Wang et al. instead **maximise P(cumulative cost ≤ a specified threshold s)** — "threshold-aware" policies.
- **Method:** stochastic optimal control; the budget `s` becomes an **extra state dimension**, and dynamic programming yields the whole family of threshold-aware policies for all `s` simultaneously in one sweep. Policies remain **bang-bang** but the drugs-on region is **strongly s-dependent** and looks completely different from the deterministic-optimal one.
- **Two environments:**
  1. **EGT model** — same GLY/VOP/DEF replicator system as Gluzman, with fitness perturbed by Brownian noise. `dmax = 3, ba = 2.5, bv = 2, c = 1, n = 4`; `γr = 1−γf = 10⁻²`, `δ = 0.05`; volatilities `σG = σD = σV = 0.15`.
  2. **Sensitive–Resistant model** (adapted from Carrère, calibrated on A549 vs A549-Epo40 lung cancer *in vitro*):
     ```
     żS = gS·(1 − zS − m·zR)·zS − α·zS·d(t)
     żR = gR·(1 − zS − m·zR)·zR − β·zS·zR
     ```
     with `m` = size ratio of R to S cells, `C` = dish carrying capacity (densities normalised by C), `α` = drug efficiency, `β` = competitive action of sensitive on resistant, volatilities `σS = σR = 0.15`. Mutation neglected. **Note the asymmetry:** sensitive cells suppress resistant ones through an explicit `−β·zS·zR` term, calibrated from data.
- **Results:** at its design threshold, a threshold-aware policy's cost CDF lies **above every other policy's** (the "threshold-specific advantage"). Worked examples: deterministic-optimal median cost 4.95 vs threshold-aware at s̄=4.5; in the SR model, deterministic-optimal median 4.71 vs threshold-aware from s̄=4.35 realising 4.02; in a harder case, deterministic-optimal median **69.45** vs threshold-aware designed at s̄=60 realising **53.63**. Advantages grow with volatility. Also: the policy that maximises unconstrained stabilisation probability was MTD, with median cost 5.95 vs 4.95 — so MTD is safest-but-costliest, as expected.
- **Take for the project:** (1) The SR model gives you a **second, independently calibrated two-population environment** for transfer testing — different structure from Strobl's (explicit βzSzR term, cell-size ratio m). (2) The conceptual point transfers directly to RL: **an agent trained to maximise expected return is not the same as an agent that reliably stays within a clinical budget.** If you have time, report the *distribution* of TTP/cumulative dose across seeds, not just the mean — and consider a CVaR or budget-constrained variant as a genuinely novel contribution.

---

### [12] Cunningham, Thuijsman, Peeters, Viossat, Brown, Gatenby & Staňková (2020) — *Optimal control to reach eco-evolutionary stability in mCRPC*
*PLOS ONE 15(12):e0243386.* doi:10.1371/journal.pone.0243386
⚠️ **Files 12a and 12b are byte-identical copies of this paper. Cunningham et al. 2018, *J Theor Biol* 459:67–78, doi:10.1016/j.jtbi.2018.09.022, is missing.**

- **Goal:** not to minimise burden or maximise TTP, but to **reach and hold a stable polymorphic equilibrium** where sensitive and resistant cells coexist — "evolutionary stable therapy."
- **Model:** Zhang 2017's 3-species LV (T+, TP, T−), with abiraterone dose `Λ(t) ∈ [0,1]` (continuous, 0 = none, 1 = MTD).
  - Growth rates **scaled to 1%** here (note: differs from Zhang 2017's 10%): `rT+ = 2.7726×10⁻³`, `rTP = 3.4657×10⁻³`, `rT− = 6.6542×10⁻³` per day, from LNCaP (60 h), H295R (48 h), PC-3 (25 h) doubling times.
  - `K_T− = 10,000` always. `K_TP(Λ) = 10,000 − 9,900·Λ` (linear). `K_T+(Λ) = μ(Λ)·x_TP` with **symbiosis coefficient** `μ(Λ) = 1.5 − Λ` (so 1.5 off-drug, 0.5 at MTD).
  - **Patient viability constraint: `Σᵢ xᵢ ≤ 9,000`.** (Chosen below the 10,000 max so that an untreated tumour always kills.)
- **⭐ The key structural finding:** with Zhang's original matrix (α ∈ [0,1]), **no stable equilibrium exists within the viability constraint** — all stable points are T−-dominated or require so many TP cells that the patient dies. Analytically, **α₃₁ and/or α₃₂ must exceed 1** (the competitive effect of T+ or TP *on* T−) for containment to be possible. They set **α₃₂ = 2**, giving
  ```
  A = [ 1   0.7  0.8 ;   0.4  1   0.6 ;   0.5  2   1 ]
  ```
  Justification: co-culture studies show cancer–cancer competition coefficients need not be ≤1 — a Gause-style MCF7 vs MDA-MB-231 experiment estimated coefficients **as high as 12.6**. α₃₂ = 5 was rejected because it eliminates T− entirely (clinically implausible).
- **Optimal control result:** forward–backward sweep from 100 initial tumour compositions to each of two stable equilibria (2-species and 3-species). The optimal dosing schedule is a **simple upward dose titration** — start small, increase slowly to the stabilising dose `Λ = 0.4` (2-species) or `Λ = 0.4848` (3-species).
- **Clinical translation:** since the physician knows neither the equilibrium nor the initial composition, they test generalised titration protocols driven only by measured total volume `V(t) = Σxᵢ`, sampled **every 100 simulated time units (~3 months)**, targeting either `Va` = incoming baseline volume or `Vb` = a maximum tolerable volume. Six clinically feasible protocols evaluated on **>200,000 virtual patients**.
- **Notable behaviour:** in the `Λ(t₀) = 0`, target `Vb = 7000` protocol, **no drug at all for >500 time units** — the tumour is allowed to grow to 7,700 (110% of target) before dosing begins, and letting T+/TP expand first *prevents* competitive release of T−.
- **Conclusions to quote:** "an increasing dose titration protocol, a very common clinical dosing process, can achieve tumor stabilization for a wide range of potential initial tumor compositions and volumes"; and, counterintuitively, "larger tumor volumes may be more likely to be stabilized if sensitive cells dominate... suggesting a delay of initial treatment could prove beneficial."
- **Code:** github.com/cunninghamjj/Optimal-Control-to-Reach-Eco-Evolutionary-Stability-in-mCRPC
- **Take for the project:** (1) **α₃₂ > 1 is a hard structural requirement** if you want containment to be achievable in a 3-species model — know this before you pick parameters, or your agent will be optimising in a world where stabilisation is impossible. (2) "Delay initial treatment" and "titrate upward" are two **non-obvious behaviours to check for in your learned policy** — if your agent discovers them unprompted, that is a genuine finding worth a figure. (3) `Σx ≤ 9000` is a clean template for a **hard safety constraint** in your environment.

---

### [13] West, You, Zhang, Gatenby, Brown, Newton & Anderson (2020) — *Towards multidrug adaptive therapy*
*Cancer Res 80(7):1578–1589.* doi:10.1158/0008-5472.CAN-19-2669

- **Framing:** a *frequency*-dependent (not population-dependent) model — assume roughly constant tumour volume and track composition only, which makes the system a replicator equation on a simplex and lets you draw phase portraits.
- **Three concepts for multidrug design:**
  1. **Evolutionary cycles** — a sequence of treatments that returns the tumour's composition to its starting point, trapping evolution in a periodic, controllable loop. Constructed per-patient from their initial position in the simplex.
  2. **Absorbing regions** — the set of composition states reachable by the tumour is *limited by which drugs are available*. More treatment options ⇒ larger region in which cycles can be found.
  3. **Velocity of evolution** — the *rate* at which composition moves determines optimal switch timing; cycles are only possible where the T− velocity is negative.
- **Parameterisation (frequency-dependent payoff `bᵢⱼ`, three cell types T+/TP/T−, two drugs Lupron and Abiraterone):**
  - Untreated: b₁₂=0.2, b₁₃=0.6, b₂₁=0.3, b₂₃=0.5, b₃₁=0.4, b₃₂=0.1
  - Lupron only: b₁₂=0.4, b₁₃=0.3, b₂₁=0.6, b₂₃=0.5, b₃₁=0.2, b₃₂=0.1
  - Lupron + Abiraterone: b₁₂=0.5, b₁₃=0.1, b₂₁=0.6, b₂₃=0.2, b₃₁=0.4, b₃₂=0.3
  - Therapy also enters via weights `wᵢ = Kᵢ/Kmax`: no treatment `K = (1.5e4, 1e4, 1e2)`; Lupron `(1.5e4, 1e4, 1e4)`; Lupron+Abi `(0.5e4, 1e2, 1e4)`; `Kmax = 1.5e4`.
  - Robustness: absorbing regions were stable across 36 random parameterisations satisfying the inequality constraints.
- **Key claim:** "the goal of adaptive therapy is to maintain a stable tumor volume in favor of designing therapies that alter tumor composition" — i.e. **composition, not size, is the real control target.**
- **Honest gap they name:** frequency-dependent models ignore volume; population models ignore composition; combining them is unfinished work. Also lists real biomarker proxies: PSA (prostate), LDH (melanoma), thyroglobulin (differentiated thyroid), calcitonin (medullary thyroid).
- **Take for the project:** If you extend to two drugs, this is your design framework. More immediately: **it is an argument for putting composition (or its proxy) in your state vector**, and a caution that a burden-only observation may be fundamentally insufficient. That is a testable hypothesis: compare an agent observing `N` alone vs one observing `(S, R)`.

---

### [14] Widdershins, Hansen, Read & Hohl (2026) — *Optimal control theory as a method for designing multidrug adaptive therapy regimens*
*npj Syst Biol Appl*, doi:10.1038/s41540-025-00613-y (in press, unedited version)

- **Model:** logistic ODE system with **four populations** and **two drugs**, resistance binary per drug: x₁ fully susceptible, x₂ resistant to B only, x₃ resistant to A only, x₄ fully resistant.
  ```
  ẋ₁ = x₁γ₁(1 − Σx/κ) − αA·CA·x₁ − αB·CB·x₁
  ẋ₂ = x₂γ₂(1 − Σx/κ) − αA·CA·x₂
  ẋ₃ = x₃γ₃(1 − Σx/κ) − αB·CB·x₃
  ẋ₄ = x₄γ₄(1 − Σx/κ)
  ```
- **Method:** Pontryagin's Maximum Principle. Controls are bang-bang wherever the switching functions `SA`, `SB` are non-zero, and **singular** where `S = 0` — and the singular arcs are where the interesting behaviour lives. With free final time, `H = 0`.
- **⭐ The "on-boundary" case is the key derivation:** when total population is held exactly at the allowed threshold `κT`, then `Σẋᵢ = 0`, which yields a closed-form expression for the drug concentration needed to *maintain* that threshold:
  ```
  CA = [ (1 − κT/κ)·(Σᵢ xᵢγᵢ) − αB·CB·(x₁+x₃) ] / [ αA·(x₁+x₂) ]
  ```
  This is a **competition-maintaining (CM) regimen** — a dose-modulation law derived, not guessed.
- **Two growth scenarios:** equal growth rates `γ₁=γ₂=γ₃=γ₄` (no fitness cost of resistance) vs unequal `γ₁=γ₂ > γ₃=γ₄` (cost).
- **Results:** regimens that maintain competition via **dose modulation for the entire course consistently outperform those using maximal dosing**, most clearly at the 50% and 80% thresholds. `umax/usingular` regimens do less well. Best and worst CM regimens cluster tightly — meaning **initial conditions and the details of how competition is maintained matter far less than whether the threshold is achievable and maintainable at all.** The authors call this out as a robustness property valuable for translation. Exceptions occur at a tight 25% threshold with low `umax`, where control capacity is simply insufficient.
- **Take for the project:** (1) The CM formula is a **strong non-RL baseline** for a continuous-dose action space — far better than "MTD or nothing." (2) The four-population structure is the natural two-drug extension if you get that far. (3) The robustness finding ("prioritising competition is robust") is a good prediction to test against your agent: does your agent converge to threshold-maintenance regardless of initialisation?

---

# Section 3 — Core RL for evolution-aware therapy

*This section is your project's direct ancestry. Papers 15, 16, 17, 20 and 22 are the ones to read line by line.*

### [15] Engelhardt (2020) — *Dynamic Control of Stochastic Evolution: A DRL Approach to Adaptively Targeting Emergent Drug Resistance* (CelluDose)
*J Mach Learn Res 21:1–30.* — **The most carefully engineered reward function in the corpus.**

- **System:** stochastic (SDE) heterogeneous cell proliferation with mutation to resistant phenotypes. Not a tumour model per se — a general resistance-emergence control problem, framed around bacterial-scale mutation rates.
- **Algorithm: DDPG.** Chosen because the observation space (cell-type concentrations) is high-dimensional and continuous, and the action space (IV dose) is continuous.
- **State (explicitly "model-informed feature engineering"):** at each decision step, the full composition `x = (x₁,…,x_d)` of cell types; types known to be possible but currently unobserved are set to `xᵢ = 0`.
- **Action:** continuous dose, single-drug `Iₜ` or two-drug `(Iₜᵅ, Iₜᵝ)`, on `[0, Imax]` with `Imax` set to **8× the largest ρ** (susceptibility parameter) — deliberately generous, so that dose restraint comes from the reward, not from a hard cap.
- **⭐ Reward — a three-part design worth copying wholesale:**
  1. **Terminal success** (population eliminated by `Tmax`), scaled by cumulative dose:
     `r_success = c_end,success · [ 1 − ( Σ_α (w_α/w₁) Σₙ I_{α,n} ) / ( c₁ · Imax · Tmax/τ ) ]`
     — drug-specific penalty weights `w_α` let a last-line drug cost more than a first-line one (`w₁ = 16`, `w₂ = 3w₁`).
  2. **Terminal failure**, scaled by how much population remains:
     `r_failure = −c_end,fail · [ 1 + log(1 + Σxᵢ(Tmax)/Σxᵢ(0)) ]`
  3. **Per-step signal** = potential-based reward shaping (Ng et al. 1999 — provably policy-invariant) plus a quadratic dose penalty:
     `r_shape(sₜ,a,s_{t+τ}) = γ·Φ(s_{t+τ}) − Φ(sₜ)`, with the potential Φ built from population **growth rate** (since the drugs act on cell birth, growth rate directly indicates efficacy);
     `r_step = r_shape − Θ(Σxᵢ) · Σ_α w_α (I_α/Imax)²`
     where Θ is a binary gate. **The dose penalty at every step was necessary** — the terminal dose term alone did not drive the policy to low dosing.
- **Mutation handling — a training trick worth stealing:** `P_step,mut = 10⁻⁶` (typical bacterial resistant-mutant rate). Crucially, only a fraction of **episodes** allow mutations at all: `P_epis,mut = 0.3`. This gave **better training** than `P_epis,mut = 1` with a proportionally reduced per-step rate — the authors attribute it to a clear no-mutation baseline signal in the majority of episodes plus a useful noise/regularisation effect from the rest.
- **Compute:** single-drug converged ~**50,000 episodes (~1M samples), ~10 h on a 3.5 GHz i7 CPU**; dual-drug ~**130,000 episodes (~2.6M samples), just over a day**. Wider hidden layers needed for dual-drug. PyTorch. Policy snapshots every 100 episodes.
- **Results:** **100% success rate** at suppressing proliferation for both single and combination policies, establishing low-dose no-event baselines, and robust across a wide sweep of `P_step,mut` values well beyond the training value. Policy analysis showed a **strong preference for short treatment times** — mean total dose rose steeply with mutation rate while episode duration stayed nearly constant.
- **Take for the project:** This is your reward-engineering template. Specifically: (i) separate terminal success/failure rewards with **graded** failure by residual burden; (ii) **potential-based shaping** to solve the credit-assignment problem over long episodes — this is the right answer to "my episodes are 2000 days and the reward is sparse"; (iii) a **per-step quadratic dose penalty**, because terminal penalties alone don't work; (iv) train with mutation enabled in only a fraction of episodes.

---

### [16] Gallagher, Strobl, Gatenby, Maini & Anderson (2023) — *Learning to Adapt — Deep Reinforcement Learning in Treatment-Resistant Prostate Cancer*
*bioRxiv 2023.04.28.538766.* — **⭐⭐ The single closest paper to your project. Read it first, in full.**

- **Environment: Strobl's 2-population LV model, verbatim** (with `dS`, `dR` written separately):
  ```
  dS/dt = rS·S·(1 − (S+R)/K)·(1 − dD·D) − dS·S
  dR/dt = rR·R·(1 − (S+R)/K)            − dR·R
  ```
  Cost of resistance defined as `1 − rR/rS`; **cell turnover defined as `dS/rS`**. Resistant cells fully resistant; `rS ≥ rR`. Parameters from Strobl 2021.
- **Baselines:** CT (`D = Dmax` ∀t) and **AT50** (`Dmax` until `N < 0.5N₀`, then 0 until `N ≥ N₀`). **Progression = 20% growth from initial size.**
- **Algorithm: A3C** (asynchronous advantage actor–critic) — chosen explicitly to avoid GPU/specialist-hardware requirements; a global network with asynchronous workers.
- **Network:** input = current tumour size `V(t)` → **LSTM** → dense **128 → 64 → 32 → 16 → 10** → policy over **2 actions** (treat / holiday). Actions sampled probabilistically from the softmax, so results are averaged over **100 evaluations**.
- **⭐ Reward function (Table 1) — the most directly reusable artefact in this whole corpus:**
  | Term | Value | When | Purpose |
  |---|---|---|---|
  | Base | **+0.1** | per timestep survived | maximise TTP |
  | Holiday | **+0.05** | per timestep without treatment | encourage intermittent therapy |
  | Progression | **−0.1** | on progression (20% growth from N₀) | punish progression |
  | Survival | **+5** | surviving 30 years | reward indefinite control |
  
  Discount **γ = 0.999** — deliberately near unity to prioritise long-horizon outcomes.
- **Results:**
  1. **DRL beats AT50 by up to ~25% in TTP** on a proof-of-principle patient. Reference values: **AT50 = 672 days, CT = 451 days.**
  2. **Decision frequency is a first-class design variable.** Training with *less frequent* decisions produced better policies. Evaluating at a *higher* frequency than training is fine; evaluating at a *lower* frequency than training causes **premature failure**. (Table 2.)
  3. **Interpretability — the headline result:** the learned policy is a clean **sigmoidal function of current tumour size**, with treatment near-certain above a threshold `N* = 0.83·n₀`. This **replicates an analytically derived optimal threshold** (derived in their SI as a function of the tumour parameters and treatment interval τ) — with *no knowledge of the underlying model*. The optimal threshold depends on the treatment interval, so **AT50 is only optimal for one specific interval (81 days)**; below it AT50 is suboptimal, above it AT50 fails prematurely.
  4. **Robustness:** transferred to a *different* tumour model (Lu et al., exponential growth, progression criterion based on the resistant subpopulation alone) — DRL achieved **1506 ± 3 days vs AT50's 1119 days** (95% CI [1110, 1128]). Across a cost × turnover grid, DRL is **robust to variation in the cost of resistance but sensitive to reductions in turnover** (faster proliferation → premature progression between decisions), where it can underperform AT50.
  5. Virtual cohort built from Strobl's fits to the Bruchovsky trial data.
- **Take for the project:** Copy the reward table. Copy the AT50/CT baselines. Copy the "extract a sigmoidal threshold from the learned policy" analysis — **it converts a black-box result into a clinically legible rule and is the single highest-value figure you can produce.** Also: make decision frequency an explicit experimental axis, and report a cost × turnover heatmap.

---

### [17] Gallagher, Strobl, Park, Spoendlin, Gatenby, Maini & Anderson (2024) — *Mathematical Model-Driven Deep Learning Enables Personalized Adaptive Therapy*
*Cancer Res 84(11):1929–1941.* doi:10.1158/0008-5472.CAN-23-2040 — **The peer-reviewed version of [16]. Cite this one.**

Same environment, same A3C framework, same reward table (reproduced identically). Adds:
- **Stronger headline:** DRL schedules **more than double TTP** relative to current adaptive protocols in a model calibrated to prostate cancer dynamics.
- **Kaplan–Meier evaluation across a 100-virtual-patient cohort:** less frequent decisions during training give more consistent outcomes, **outperforming other strategies in >90% of cases**. Evaluating below the training frequency causes a stark TTP drop; above it, no significant change.
- **Generalist vs specialist trade-off, made explicit:** a DRL model trained on a single patient significantly outperforms the clinic for that patient but **struggles on other patient profiles**. Training on a virtual cohort fixes this at the cost of specialisation — the generalist adopts a **more conservative strategy** that avoids premature progression for everyone but is suboptimal for the least aggressive tumours.
- **⭐ The five-step clinical pathway (Fig 6) — the most translatable idea in the corpus:**
  1. Patient undergoes an initial **"probing" cycle of AT50**.
  2. Fit the virtual-patient model to that cycle → patient-specific parameters.
  3. **Retrain (fine-tune) a copy of the generalist DRL model** on those parameters.
  4. Extract a **sigmoidal treatment strategy** fitted to the network's decisions.
  5. Use that interpretable rule for the rest of treatment.
  
  Applied to trial patient 25: switching to the DRL schedule after the probing cycle significantly increases TTP over AT50. Across patients, the personalised critical tumour size at which treatment starts/stops **varies substantially between individuals** — which is precisely why a single fixed threshold like AT50 cannot be optimal for everyone. The extracted sigmoid retains comparable performance to the network, **eliminating the black-box objection**.
- **Also notes:** raising the withdrawal threshold (treat less, hold at higher burden) increases competitive suppression and TTP — up to "range-adaptive" AT (Brady et al.) where the tumour is allowed *above* baseline — but at the cost of more phenotype switching, de novo mutation and metastasis risk.
- **Take for the project:** The five-step pathway is the ideal structure for your Discussion / Future Work, and step 4 (distil the policy into a sigmoid) is achievable within your project's scope. The generalist-vs-specialist result tells you to **train on a cohort, not one patient**, and to report both.

---

### [18] Weaver, King, Maltas & Scott (2024) — *Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance*
*PNAS 121(6):e2303165121.* doi:10.1073/pnas.2303165121

- **System (not cancer — but the methodology is the point):** *E. coli* evolving on **empirically measured fitness landscapes** of the TEM β-lactamase gene. 4 mutational sites → binary genotype strings → **16 genotypes**; **15 β-lactam antibiotics**, each defining its own fitness landscape over the same genotype space. Evolution under **strong-selection weak-mutation (SSWM)**, so each landscape is a Markov chain (following Nichol et al.) and the population fixes before transitioning.
- **MDP formulation — clean and worth imitating:**
  - **State set S** = all genotypes (16 in the base case)
  - **Action set A** = the 15 antibiotics
  - **Reward** `R = 1 − f(s|a)` — i.e. **reward = negative fitness**, bounded and dimensionless
  - **Finite horizon: 20 evolutionary steps**, one step ≈ one day. Finite rather than infinite horizon chosen "to more faithfully represent clinical disease courses."
- **Two solution methods, deliberately paired:**
  - **Perfect information:** solve the MDP by **backward induction** → provably optimal, time-and-state-dependent policy. This is the ceiling.
  - **Imperfect information:** **Deep Q-learning**, in two information regimes:
    - **RL-genotype** — input is the instantaneous genotype. Architecture: input → two 1-D conv layers → max-pool → dense(28) → linear output.
    - **RL-fit** — input is instantaneous population *fitness* plus the one-hot previous action. Architecture: input → dense(64) → dense(28) → linear output. MSE loss, target network updated rarely.
  - **Noise:** `sₜ = sₜ + w`, `w ~ N(μ, 0.05·σ²)`, with σ² swept from 0 (clean) to **100** (profound signal loss).
- **Results:** RL agents beat random and naive cycling, and **approach the optimal MDP policy**. Ranking: MDP ≳ RL-genotype > RL-fit > random. Performance holds under substantial measurement noise and under **delayed** (stale sequencing) information. Scaled to landscapes of up to **1024 genotypes (N=10 alleles)** — "by far the largest state space evaluated in the treatment optimization literature" — and **increasing genome size did not make control less feasible**, though the value/policy landscape becomes rugged. Near the horizon end (steps 19–20) the optimal policy switches to a **greedy** minimum-fitness choice.
- **Interpretability device — "opportunity landscape":** take the element-wise minimum fitness across the drugs a policy actually uses. For the MDP's 5 drugs (CTX, CPR, AMP, SAM, TZP), the combined landscape has **only one fitness peak, with 15/16 genotypes in or near valleys** — and the MDP's value function matches this topography closely. RL-genotype achieved a near-identical opportunity landscape using only 3 drugs (CPR, CTX, SAM); RL-fit typically used just 2 (AMC, CTX).
- **Limitations they state:** SSWM assumption; drug concentration held constant (no seascape); ≤10 genotypic positions vs *E. coli*'s ~5×10⁶ bp; no off-landscape resistance (efflux pumps).
- **Code:** github.com/DavisWeaver/evo_dm and github.com/DavisWeaver/rl_cycling
- **Take for the project:** (1) **`R = 1 − fitness` is the cleanest reward in the corpus** — bounded, interpretable, no weight tuning. Consider a burden-normalised analogue. (2) The paired **"solve exactly with perfect information, then learn with imperfect information, compare"** design is exactly the experimental structure your project should adopt — it gives you an optimality ceiling to report against. (3) The noise and measurement-delay sweeps are cheap, high-value robustness experiments that map directly onto realistic PSA monitoring intervals. (4) Finite horizon of 20 steps, justified clinically.

---

### [19] King, Stacy, Weaver, Maltas, Barker-Clarke, Dolson & Scott (2025) — *Fitness seascapes are necessary for realistic modeling of the evolutionary response to drug therapy*
*Sci Adv 11:eadv1268.* doi:10.1126/sciadv.adv1268

- **The concept:** a **fitness landscape** maps genotype → fitness at one drug concentration. A **fitness seascape** maps genotype **and drug concentration** → fitness — i.e. a collection of genotype-specific **dose–response curves**. This is what you need to model evolution under *fluctuating* (pharmacokinetic) drug levels.
- **Empirical work:** *E. coli* with all 16 combinations of four TEM resistance mutations; cefotaxime dose–response measured for all 16 (N=8 replicates per condition); time-kill assays to parameterise net cell **death** (net growth < 0), which was experimentally difficult.
- **Measured cost of resistance — a rare direct quantification:**
  - drug-free growth rate `g₀` **decreases** with number of mutations (r²=0.48, P=0.003)
  - IC₅₀ **increases** with number of mutations (r²=0.46, P=0.004)
  - `g₀` vs IC₅₀ **inversely related** (r²=0.38, P=0.011) — a direct fitness cost
  - Allele-level: **G238S** is the only mutation independently associated with IC₅₀; G238S and M182T are each independently *negatively* associated with drug-free growth, but their **interaction term is positive** — **reciprocal sign epistasis**, which explains why the two co-occur in clinical TEM isolates.
- **Simulation:** agent-based model + *in silico* PK curves (t½, Cmax), Latin hypercube over 300 parameter sets, 100 simulated patients per condition, dosing every 8 h for 10 days after 3 days of equilibration. Non-adherence modelled by `p_miss`/`p_forget`.
- **Results:** parameters driving **mutation supply** — mutation rate `rm` and carrying capacity/population size `K` — most strongly determine treatment failure. **Early adherence is critical**: risk ratios per dose show the **second dose** carries disproportionate weight; regimens with 8 h between the first two doses succeeded far more often than those with longer gaps. Confirmed *in vitro* (perfect vs late-adherence vs early-adherence, N=8).
- **Take for the project:** (1) Cite this when you justify treating resistance as **fully** resistant or when you *don't* model PK — it names exactly what you're leaving out. (2) The `g₀`–IC₅₀ trade-off is your empirical citation for "cost of resistance is real and measurable," and the meta-analysis it aligns with (cost magnitude correlates with MIC). (3) "Early doses matter disproportionately" is a striking, testable prediction: does your agent front-load treatment? Enriquez-Navas's two-phase finding says the same thing from the mouse side. That convergence is worth a paragraph.

---

### [20] Aif, Eiche, Appold, Fischer, Citak & Kayser (2025) — *Reinforcement Failing guides the discovery of emergent physical dynamics in adaptive tumor therapy*
*bioRxiv 2025.04.08.647768.* — **⭐ The paper that will save you from over-claiming. Read §Methods and the Discussion.**

- **The idea, inverted:** instead of using RL to find a good policy, use **systematic RL failure across environments** to discover what a simplified model is *missing*. "Reinforcement Failing" = AI-guided, human-in-the-loop refinement of abstracted training environments.
- **Two environments:**
  - **TE₀ (base training environment)** — a 2-ODE system with **asymmetric competition**:
    ```
    dS/dt = γS·S·(1 − (S+R)/K)   − δ·S − D
    dR/dt = γR·R·(1 − (R + c·S)/K) − δ·R
    ```
    Note two deliberate choices: the competition coefficient `c` acts **only on R** (asymmetric), and the **drug effect D is independent of population size** — to capture the periphery-to-core killing seen in the spatial model. `K = 18,000 cells`. Calibrated to the reference environment by **Bayesian inference** (PyMC 5.9.2, DEMetropolis MCMC, 8 chains), fitting `γR` first on continuous therapy, then `γS`, `D` and `c` on adaptive therapy. The authors explicitly warn these effective parameters should **not** be read as true biological rates.
  - **RE (reference environment)** — an off-lattice agent-based model in **PhysiCell**, spatially explicit.
- **Agent:** **PPO** (Stable-Baselines3), policy and value networks both MLPs with **two hidden layers of 128 units**. Observation space = **stacked past 64 cell counts and treatment decisions**, concatenated — explicitly to allow history dependence and learning beyond simple threshold rules. Trained **5×10⁶ environment steps**, ~4,900 policy updates, ~10⁵ episodes; updates every 1,024 steps; one step = one decision = **1.5 simulated generations**.
- **⭐ Results — the cautionary tale:**
  - In TE₀, all 10 agents converged and achieved **TTF = 149 ± 19 generations**, an almost **fivefold improvement over AT50's 33.8 ± 0.8**.
  - Evaluated in the spatial RE, agent performance was **highly heterogeneous** and the ranking did not transfer. Policy-space analysis via **KL divergence between state–action distributions** (their "GRAPE" framework) separated well- and poorly-performing agents.
  - **The conventional wisdom partly broke:** high average total burden — the supposed driver of competitive suppression — correlated with long TTF at **r = 0.91 in TE₀ but only r = 0.77 in RE**; cross-environment performance difference correlated r = 0.45 in the training environment vs **r = 0.98** in the reference.
  - Human-in-the-loop inspection of failing agents identified the missing mechanism: **growth-driven collective cell motion coupled to spatially heterogeneous proliferation accelerates resistance escape.** Agents that applied treatment at *low* tumour burden did badly.
  - Adding this to an **augmented environment (TE⁺)** produced second-generation agents with better cross-environment performance.
  - Also: adaptive scheduling's benefit here did **not** require a cost of resistance — corroborating Strobl and lattice-model results.
  - **Practical warning on probing cycles:** ill-timed patient-specific parameterisation cycles risk **premature competitive release**. Splitting data acquisition across multiple cycles of a **higher-threshold strategy (e.g. AT75)** substantially lowers that risk (TTF = 147 ± 52 generations). This directly qualifies Gallagher's five-step pathway.
  - Limitations they state: failure-mode detection needs **diverse** agents (poor reward design ⇒ the ensemble collapses to near-identical policies); PhysiCell still omits vasculature, immune infiltration, metabolic feedback; single isolated lesion; "conceptual demonstration with limited translational perspective."
- **Take for the project:** (1) **Build the transfer test in from the start** — even a crude second environment (different growth law, different progression criterion, added noise, or a small spatial/stochastic model) turns "my agent got a big number" into a defensible result. (2) Their observation space (**stacked history of burden + past actions**) is a cheap, powerful upgrade over "current burden only" and directly addresses the partial-observability problem. (3) Train **an ensemble of agents with different seeds** and report the spread — a single agent's number is not a result. (4) The AT75-probing caveat belongs in your Discussion.

---

### [21] Liu, Zhang, Hong, Nie & Sun (2025) — *Multiscale mathematical model-informed RL optimizes combination treatment scheduling in glioblastoma evolution* (M4RL)
*Sci Adv 11:eadv3316.* doi:10.1126/sciadv.adv3316

- **Biology:** glioblastoma; tumour-associated macrophages (TAMs) ↔ tumour cells (TCs) interaction driving **CSF1R-inhibitor resistance**; combination with an IGF1R inhibitor.
- **Three-layer architecture — the notable engineering contribution:**
  1. **MSABM** — a multiscale agent-based model integrating molecular, cellular and microenvironment scales. Cost: **~550 core-hours** for a 200-day, N=100 multi-prediction run at 0.5 h resolution. Far too slow for RL.
  2. **Surrogate model** — a hybrid of SDEs (cell densities) + ODEs (cytokine/drug concentrations); each SDE converted to a **Fokker–Planck equation** to track the *probability distribution* of cell density across a predicted population; approximated by a **physics-informed neural network (PINN)**. Cost: **<18 core-hours** for the same prediction — a **~30× speed-up**. Accuracy: RMSE < 0.06, R² > 0.97 against MSABM at days 50/100/150/200; survival curves compared by log-rank, KS statistic, RMSE and KL divergence.
  3. **RL** — **A3C** (parallel actor–critic) on the surrogate. **State = a two-dimensional time series: current predicted population survival probability, and the change in survival probability.** State → **LSTM** feature extractor → 4-D feature vector → agent → softmax policy over the two drugs; next action sampled probabilistically.
- **Validation:** optimal dynamic CSF1R_I + IGF1R_I regimen identified, then **verified against spatial transcriptomic data** and against published experimental survival curves under a switch-treatment protocol.
- **Also modelled:** quiescent TCs, whose presence degrades CSF1R_I efficacy (basal vs non-quiescent comparison, log-rank tested).
- **Take for the project:** This is the answer to "my realistic simulator is too slow for RL." Train a fast surrogate (here physics-informed; a plain neural ODE or emulator would also do) and run RL on it. Also note the unusual state choice — **survival probability and its derivative rather than raw cell counts** — a good example of designing an observation that is both low-dimensional and clinically meaningful.

---

### [22] Ma, Wang, Yang, Cheke & Tang (2026) — *Single-threshold-guided adaptive cancer therapy with partial-cycle treatment: a mechanistic and reinforcement learning analysis*
*PLOS Comput Biol 22(6):e1014457.* doi:10.1371/journal.pcbi.1014457 — **⭐ The most recent, most directly comparable RL paper. Your closest competitor and your best template.**

- **Clinical motivation:** existing AT protocols assume frequent monitoring to support precise decisions. That is clinically infeasible. So: **keep the surveillance cycle long (30 days) but treat only during part of it.**
- **Model — Lotka–Volterra with explicit asymmetric competition coefficients and a *cytostatic* drug:**
  ```
  dS/dt = rS·S·(1 − (S + α·R)/K)·(1 − dD·D(t)) − dS·S
  dR/dt = rR·R·(1 − (β·S + R)/K)               − dR·R
  ```
  **Note the drug term:** ADT is modelled as reducing the **proliferation rate** (`rS(1 − dD·D)`), not as increasing death — justified because ADT induces G0/G1 cell-cycle arrest rather than direct cytotoxicity. `D(t) ∈ {0,1}` binary (clinical on/off standard). Resistant cells completely insensitive; no treatment-driven phenotype conversion.
- **Parameters:** taken directly from **Strobl's fits** to **7 representative prostate patients** from the Bruchovsky IADT trial (S₀, R₀, rS, rR, dS, dR, K per patient); **α = β = 1** (symmetric competition). Surveillance cycle **T = 30 days**.
- **Strategies compared (a complete baseline suite — reuse this list):**
  - **CT / MTD**: `D(t)=1` ∀t
  - **IT** (intermittent): fixed on-period T₁ = 30 d, fixed holiday T₂ = 30 d
  - **AT50**: Zhang's two-threshold rule
  - **AT-FSC** (full surveillance cycle): treat for the whole 30-day cycle whenever `N(nT) > N_T`
  - **AT-PSC** (partial surveillance cycle): treat only for `T₁ ∈ [0,30]` days of each cycle when `N(nT) > N_T` — **the paper's proposal**
  - **AT-RL** variants: `RL(0,30)` binary {0,30} days (≡ data-driven AT-FSC); `RL(0,15)`; `RL(0,19)`; and full action space 0–30 days
- **RL setup:** **PPO** with clipping, **100,000 training iterations**. Reward `rₜ = r_base + r_treatment + r_progress` = **+0.1/day survived, +0.05 per non-treatment day, −0.1 on progression** — explicitly *"adapted from the reward designs used in [Gallagher] and simplified."* Weights chosen heuristically to put three clinical objectives on comparable scales. Actions sampled from the learned policy, so repeated rollouts differ even in a deterministic environment.
- **Endpoint:** **TTP = first t with `N(t) ≥ 1.2·N₀`** — RECIST-analogous 20%, with the paper stating plainly that total volume is a simplified surrogate for lesion-diameter sums.
- **Results (Patient 25 parameters):**
  - AT-PSC at its optimal threshold beats **AT-FSC by 190 days (+7.4%)**, **AT50 by 639 days (+30.2%)**, and **MTD by 1,679 days (+156%)**.
  - Optimising the treatment window as well adds a further **402 days**, with a **10.1% reduction in cumulative dose**.
  - TTP rises with the treatment threshold `N_T` up to a critical `N*_T`, then **falls sharply** — because treatment triggered too late cannot control rapid sensitive-cell growth. This is a *different* failure mode from competitive release, and the paper is careful to distinguish them.
  - Once `T₁` exceeds ~9 days, AT-PSC beats MTD, IT and AT50; at `T₁` of 10–20 days it also beats AT-FSC. Optimal window ≈ **13 days** of 30.
  - **Action-space design matters a lot:** Kaplan–Meier and TTP distributions over 100 stochastic simulations differ substantially between `{0,30}`, `{0,15}`, `{0,19}` and the full `0–30` space. **RL with an adaptive treatment window further improves TTP** over the best mechanistic AT-PSC.
  - Robustness: heatmaps over `(rS, rR)` with `T₁` and `N_T` independently re-optimised; a fixed `T₁ = 13` d performs well across the grid.
- **Code:** github.com/Kexin0902/AT-PSC
- **Take for the project:** This paper proves your project design is publishable-adjacent and shows exactly how to structure it. Two openings it leaves you: (i) it uses a **single threshold + window** — a richer state (history, composition) or a continuous dose is unexplored; (ii) it does **no cross-environment transfer test** — combining Ma's protocol suite with Aif's transfer discipline would be a genuine contribution. Also note it independently converged on **Gallagher's reward weights**, which makes those weights a de facto standard you can adopt without justifying from scratch.

---

### [23] Ghoreyshi, Debnath, Olawuni, Armstrong, Somarelli & George (2025) — *A memory-driven RL model of phenotypic adaptation for anticipating therapeutic resistance in prostate cancer*
*bioRxiv 2025.11.03.686133.*

- **The distinctive idea: cellular memory.** Phenotypic transitions depend not on the current environment but on a **running average `πₙ` of the last N environmental states**, compared against a critical value `π_crit`. Resistance is **reversible and non-genetic**.
- **Experimental system:** LNCaP prostate cancer cells, **enzalutamide** vs DMSO, fluorescently labelled sensitive (green, `X_SG`) and resistant (red, `X_RR`) populations, 24-day time series at **24-hour resolution**, 3-day treatment cycles. Resistant phenotype stability confirmed by >1 year of chronic culture.
- **Aggregate model:** growth rates `R_S^DMSO = r_S^DMSO(1−L) − c_S^enza·L`, `R_S^enza = r_S^enza(1−L)` with `L ∈ {0,1}` the environment and `c_S^enza` the cost of the sensitive phenotype under drug.
- **⭐ Result on memory length:** varying memory capacity as the single fitting parameter, the best fit to experimental growth-rate trajectories was a **6-day memory window — exactly twice the 3-day treatment cycle.** A 2-day window adapted too crudely; a 12-day window gained no accuracy in environmental estimation but converged too slowly. The best-fitting memory is also the one that **maximises aggregate cellular fitness** over the 15-day window — i.e. the cells' memory length is itself near-optimal.
- **RL model:** **Soft Actor–Critic (SAC)** with a **time-dependent mixed-effect generalised Lotka–Volterra (tM-GLV)** environment, including two intermediate transitioning states beyond S and R, a phenotype transition matrix `Φ(I(t))` driven by the memory integral, logistic transformations F/G governing switching asymmetry, density-dependence strength `η`, and treatment-induced death `R̃(D)` with `B = diag(β₁, β₂)` the phenotype-specific drug-effect coefficients.
  - **State:** cell population counts **and the ratio of enza-resistant to enza-sensitive cells**.
  - **Action:** continuous treatment intensity.
  - **Reward:** integrates therapeutic effectiveness, phenotype transitions and **cumulative dose penalties**, plus an explicit **progression-free-survival component in the step reward** — added because reducing dose to suppress resistance can let sensitive cells grow to a critical burden. Stated objective: "maintain a minimal yet stable population of resistant cells."
  - SAC chosen for **entropy regularisation** → better exploration of the phenotypic landscape and better capture of coexistence; required careful per-patient hyperparameter tuning to avoid divergence.
- **Clinical application:** calibrated on the **first 10–15%** of each patient's PSA trajectory (through one decline-and-rebound), then used to predict the rest. **SAC-derived ADT policies maintain >50% progression-free beyond 120 months vs 100% progression by 60 months under the clinical intermittent schedule** (for patients who actually progressed). Inferred parameters discriminate outcome: progressors show earlier memory-driven sensitive→resistant shifts (γ), faster PSA rebound (ϕ), and weaker drug effects on both phenotypes (non-progressors have β₁ > β₂ and both larger).
- **Take for the project:** (1) If you want a genuinely novel angle, **reversible phenotypic plasticity with memory** is it — every other paper in this corpus assumes pre-existing, irreversible resistance, and West 2023 §1.3 names plasticity as an open question. (2) The result that **optimal memory ≈ 2× the treatment cycle** is a beautiful, citable bridge between cell biology and scheduling. (3) Note their state includes the **R:S ratio** explicitly — evidence for the composition-in-state argument from [13]. (4) SAC + continuous dose is the natural upgrade path from binary PPO.

---

### [24] Tavakoli, Mohammadpur, Salimi Sartakhti & Manshaei (2025) — *RL-Driven Evolutionary Stackelberg Game Model for Adaptive Breast Cancer Therapy*
*Math Comput Appl 30(6):134.* doi:10.3390/mca30060134

- **The framing:** a **Stackelberg game** — the physician is a *rational leader* who moves first and anticipates the tumour's response; the heterogeneous cell population is the follower. Contrasted with the **Nash** strategy, which optimises dose against a *fixed* resistance level. This is Stein et al.'s "ecologically enlightened" (Nash) vs "evolutionarily enlightened" (Stackelberg) distinction.
- **Data:** tumour volumes from breast-cancer-bearing mice treated with **Capecitabine and Gemcitabine**. Parameters estimated per-subject with the **GEKKO** optimisation package in Python. Example fits: Mouse 2 `r=0.20, u₀=1.5478, σ=0.0499`; Mouse 25 `r=0.1907, u₀=0.5102, σ=0.0499`.
- **Benchmarking:** against Exponential, Logistic and Gompertz growth models. Those capture monotonic growth but **fail to reproduce treatment-induced regression, rebound, and phenotypic switching** — the game-theoretic model matches across MTD, dose-modulation AT, and intermittent AT protocols.
- **Physician objective function:** `Q` weighted by `C₁` (tumour burden), `C₂` (resistance), `C₃` (toxicity), evaluated at the ecological equilibrium `x*(D, u_R)`. Parameterisation `Qmax=1, C₁=0.51, C₂=0.08, C₃=0.41`; transition rates `α_RS=0.60, α_RR=α_SS=1.0, α_SR=0.25`. **Progression defined as burden exceeding `δ·Kmax` with `δ = 0.7`.** Outcomes classified into elimination / stabilisation / progression regions.
- **Result:** all three strategies land in the stabilisation region, but **Stackelberg > Nash > MTD** on the objective. Stackelberg gives lower dose, lower toxicity **and** lower treatment-induced resistance. The authors honestly flag that Stackelberg's near-zero dosing is driven by the heavy toxicity weight `C₃` — "highlights the sensitivity of the framework to parameter selection."
- **RL component:** **Q-learning**, single-agent and two-drug. Reward:
  ```
  reward = (x_prev − x) − λ_dose·(D_G + D_C) − β_final·u
  ```
  with `λ_dose = 0.08`, `β_final = 0.15`; `u` = current resistance level; constraint `D_G + D_C < 1` (normalised total dose cap per step). Trained then applied to an unseen test subject (Mouse 25). The agent **intensifies treatment as burden rises and reduces it when resistance is low** — behaviour the authors attribute directly to the reward structure. Combination therapy exploits **dose diversification** to improve control without exceeding the total dose budget. Quantitative comparison (Table 4) on final burden, cumulative dose and final resistance level.
- **Take for the project:** (1) The reward function is notable for **penalising the resistance level directly** (`−β_final·u`) — most papers penalise only dose and burden. If your state includes R, this is a one-line experiment: does adding a resistance penalty change the learned policy? (2) The Stackelberg-vs-Nash distinction is a good conceptual frame for "why anticipate evolution instead of reacting to it" — useful in your Introduction. (3) The per-step **total dose cap across drugs** is the right way to model toxicity in a multi-drug action space. (4) Caveat when citing: the sensitivity of the result to `C₃` is acknowledged by the authors themselves.

---

### [25] Giles & Newton (2026) — *Reinforcement learning for chemotherapy scheduling in a stochastic tumor evolution model*
*Phys Rev E 113:064404.* doi:10.1103/cbkw-5s4b — **The most interesting reward function in the corpus.**

- **Model:** a **stochastic finite-cell** (weighted **Moran process**) model. Three subpopulations: chemosensitive **S**, and two **single-drug-resistant** lineages **R₁** and **R₂**, resistant to cytotoxic agents C₁ and C₂ respectively. **Fixed total population N**, so the state is a composition `(S, R₁, R₂)` on a simplex (they work in `S₅₀`, i.e. N=50). Fitness from a payoff matrix M; tumour **volume** evolves in a separate equation with expansion rate ∝ (population-averaged fitness − a fixed microenvironmental baseline).
- **MDP:** `S` = valid subpopulation distributions; `A` = allowable discrete chemotherapy doses of C₁/C₂; `T` = birth–death transition probabilities from the Moran process; `R(s,s',a) = R(s')`, a **table over reached states**. Tabular **Q-learning**, `Q ← Q + α(r + γ·maxₐ Q(s',a') − Q)`. Explicitly **model-free**: the agent never sees the payoff matrix or transition probabilities — the simulator is an opaque experience generator.
- **⭐ Reward — maximise population ENTROPY, i.e. reward coexistence:**
  ```
  H(x) = −Σᵢ (xᵢ/N)·log₁₀(xᵢ/N)
  ```
  Rationale, and it's a good one: without drug, S outcompetes R₁ and R₂ and fixes; with high-dose drug, S is eliminated and the resistant lineages invade. **The goal is that all three coexist with none reaching fixation.** In a stochastic finite population, small subpopulations are lost to drift, so **maximising diversity minimises extinction risk** for the subpopulation you need (S) — and Shannon entropy is maximised exactly at equal coexistence. (They cite the literature on entropy as a measure of tumour heterogeneity.)
  - Implementation: a **piecewise-constant** reward over the entropy landscape. Binary version: **+1** for states within 0.05% of maximum entropy (the "target region"), **−1** near the simplex boundaries. Three- and four-region variants also tested, producing visibly different learned policies (Fig 16).
- **Analyses performed:** (i) inferring dominant evolutionary trajectories under treatment from the policy structure; (ii) quantifying **robustness to partial observability** of both initial conditions and state updates; (iii) constructing simplified **symmetry-informed heuristics** that approximate the full learned policy.
- **Take for the project:** (1) **Reward coexistence, not burden reduction.** This is the most direct translation of "adaptive therapy's goal is containment, not cure" into a reward function, and it is a legitimately different design choice from the Gallagher/Ma survival-plus-holiday scheme. **Running both on the same environment and comparing the learned policies would be a strong, self-contained result for your project.** (2) Distilling the learned policy into a symmetry-informed heuristic is the same interpretability move as Gallagher's sigmoid — both are worth doing. (3) A Moran process on a simplex is cheap to implement and gives you genuine stochasticity plus drift-driven extinction, which ODEs cannot.

---

### [25b] Li, Tan & Yu (2025) — *Optimal adaptive cancer therapy based on evolutionary game theory*
*PLOS ONE 20(4):e0320677.* doi:10.1371/journal.pone.0320677

- **Model:** West et al.'s EGT model of cancer progression with **three types: healthy cells, drug-sensitive cancer cells, drug-resistant cancer cells** — note the explicit healthy-cell compartment, which most AT models omit. **Pharmacokinetics added** (drug concentration as a state, not just dose).
- **Method:** optimal control with two hard constraints — **maximum drug concentration `cmax`** and **maximum tumour burden `Tmax`**. Existence of an optimal control proved; structure derived via **Pontryagin's minimum principle**.
- **Comparators:** MTD (25 days continuous), low-dose continuous (`m = 0.352`), metronomic (`m = 0.5`, 12.5-day interval), and the derived optimal adaptive strategy.
- **Optimal strategy structure:** begin with a **short burst at maximum dose** to remove some sensitive cells, then switch to a different dose until the internal drug concentration reaches `cmax`, then **halt administration**. Example: doses `m = (…, 0.8745)` with on/off switch times `t = (1, 6.2, 14.2, 18.6, 27, 31.6, 39.6, 44.8)`.
- **Personalised variants:**
  - *Low tumour burden group*: `x = (0.4, 0.5, 0.1)`, `cmax = 0.6`, `Tmax = 0.4` → doses ≈ (0.75, 0.745, 0.745, 0.745), switch times (0.6, 5.6, 12, 16.4, 22.8, 27.6, 34, 39.6). Healthy and tumour fractions settle into **low-amplitude oscillation**.
  - *High tumour burden group*: `cmax = 0.79`, `Tmax = 0.5`, maximum dose used → switch times (2.4, 4.2, 14.2, 18.6, 28.6, 33.2, 43.2, 48.2); healthy cells fluctuate **50–75%**.
- **Stated result:** the optimal adaptive strategy "effectively preserves a high survival rate of healthy cells" by keeping sensitive and resistant populations in **low-level competition**.
- **Take for the project:** Two things most AT models skip and this one has: **healthy cells in the state** (giving you a principled toxicity term rather than a dose proxy) and **explicit PK** (so the action is dose but the effect is concentration). Either is a defensible extension. The `cmax` / `Tmax` pair is also a clean **constrained-MDP** formulation if you want to go beyond reward shaping for safety.

---

# Section 4 — RL for chemotherapy dosing (control-theory lineage)

*These papers mostly do **not** model resistance evolution — they model tumour–immune–normal-cell dynamics and optimise dose against toxicity. Their value to you is methodological: algorithm choices, robustness protocols, and the RL-vs-optimal-control comparisons.*

### [26] Padmanabhan, Meskin & Haddad (2017) — *Reinforcement learning-based control of drug dosing for cancer chemotherapy treatment*
*Math Biosci 293:11–20.* doi:10.1016/j.mbs.2017.08.004

- **The first RL-for-chemo-dosing paper in this corpus.** Tabular **Q-learning**, model-free, closed-loop.
- **Environment:** the de Pillis–Radunskaya-type 4-state model — `x₁ = N` (normal cells), `x₂ = T` (tumour), `x₃ = I` (immune), `x₄ = C` (drug concentration); logistic growth for normal and tumour cells, per-capita death `d₁`, cell-kill terms `a₁,a₂,a₃` for immune/tumour/normal.
- **State:** discretised on **tumour cell count `x₂` alone**. **Action:** discretised drug infusion rate, `[0, umax]`.
- **Reward — relative error improvement, an unusual and rather elegant choice:**
  ```
  r_{k+1} = (e(kT) − e((k+1)T)) / e(kT)   if e((k+1)T) < e(kT)
          = 0                             otherwise
  ```
  where `e(t)` is a chosen combination of system states. The agent is rewarded for **fractional progress**, never punished — so reward is bounded in [0,1) and scale-free. ε-greedy exploration; Q updated with learning rate `ηₖ`; convergence tested by `|Qₖ − Q_{k−1}| ≤ δ`.
- **Patient-specific customisation is done two ways, and this is the paper's real contribution:** (i) reshape `e(t)` to encode clinical priorities — for a *young* patient, prioritise rapid tumour kill and accept normal-cell damage (regeneration is likely, metastasis is the bigger risk); for a patient with comorbidities or a brain tumour, protect normal cells; (ii) set `umax` to respect patient limits — three cases are worked: young adult (`umax = 10 mg/l/day`), pregnant woman (`umax = 0.5` until delivery at day 90, then 10), elderly with comorbidities (`umax = 10` with a protective error function).
- **Parameters:** `a₁ = 0.2` (immune kill), `a₂ = 0.3` (tumour kill), `a₃ = 0.1` (normal kill) mg⁻¹l day⁻¹; `b₁ = b₂ = 1` cell⁻¹ (reciprocal carrying capacities). **50,000 training scenarios.** Robustness tested at −10% and +15% parameter variation; statistics over **15 simulated patients**.
- **Take for the project:** (1) The reward is a clean, scale-free alternative worth trying: reward **fractional improvement** rather than absolute burden. (2) The idea that **patient-specific priorities live in the reward function, not the model** is a useful framing for your Discussion — it's how you'd let an oncologist tune an RL policy. (3) `umax` as a patient-specific hard cap is the simplest possible toxicity model.

---

### [27] Eastman, Przedborski & Kohandel (2021) — *RL derived chemotherapeutic schedules for robust patient-specific therapy*
*Sci Rep 11:17882.* doi:10.1038/s41598-021-97028-6 — **⭐ The definitive RL-vs-optimal-control robustness study. Use it to justify your method choice.**

- **The question:** *in-silico* dosing schedules depend on a model parameterisation that is prohibitively hard to validate per patient. If the parameters are wrong, is an RL schedule or an optimal-control schedule more forgiving?
- **Environment:** a two-compartment **tumour growth inhibition model** — proliferative `P` and quiescent `Q` cells; `P` self-renews at `γ`, dies at `δ`, becomes quiescent at `α`; `Q` dies at `ε`, re-enters proliferation at `β`; only `P` is killed by the drug `f(t)`. Parameterised for **breast/ovarian solid tumour** *and separately for bone marrow*, so the model carries an explicit **toxicity compartment**.
- **Objective functional:**
  ```
  J_b(f) = ∫₀ᵀ [ P_bm(t) + Q_bm(t) − b·( P_bc(t) + Q_bc(t) ) ] dt
  ```
  maximise surviving bone marrow while minimising cancer cells; `b ≥ 0` sets the trade-off (`b ≫ 1` → maximal dosing regardless of marrow; `b ≪ 1` → preserve marrow at the cost of kill). The authors note explicitly that the exact functional form is a modelling choice, not a result.
- **Crucial design detail:** the RL agent is given **periodic access to relative bone marrow density** — a far more easily measurable quantity than tumour-model parameters. The optimal controller was given the same measurements for fairness.
- **Method:** **double deep Q-learning**, hyperparameter-tuned; compared against the discretised optimal control problem solved with **APOPT via the GEKKO** Python library. Both agents trained/derived on nominal mean-value parameters `ξ₀`, then applied blind to **600 testing virtual patients** (200 each at **15%, 20%, 25% parameter perturbation strength**). Scores scaled by separately re-solving the optimal control problem for each test patient. A stronger baseline — the **nearest-training-neighbour optimal controller (NTNOC)** — was also constructed, giving OC access to distribution-level data.
- **⭐ Results — the numbers to quote:**
  | Perturbation | RL median scaled score | Nominal OC | RL sd | OC sd |
  |---|---|---|---|---|
  | 15% | **0.991** | 0.976 | 0.01 | 0.09 |
  | 20% | **0.992** | 0.962 | 0.01 | 0.14 |
  | 25% | **0.991** | 0.986 | 0.01 | 0.16 |
  
  Versus the stronger NTNOC baseline: NTNOC wins narrowly at 15% (by 0.003 in median), **RL wins at 20% (by 0.077) and 25% (by 0.060)**; Wilcoxon statistics 13023.5 (p ~ 10⁻⁸) at 20% and 14593.5 (p ~ 10⁻⁹) at 25%. NTNOC standard deviations (0.007, 0.06, 0.06) sit between nominal OC and RL.
- **The interpretation that matters:** RL's advantage is not a higher peak — it is a **near-zero variance across patients**. The RL score histogram barely moves under perturbation; the OC histogram diffuses badly. Local sensitivity analysis confirmed the model is genuinely sensitive to parameter perturbation.
- **Take for the project:** This is your citation for "why RL and not optimal control." Even better, **copy the experimental design**: derive/train on nominal parameters, then evaluate on a large cohort of perturbed virtual patients at several perturbation strengths, scale scores by the per-patient optimum, and **report the variance, not just the mean**. Include an optimal-control (or AT50-with-oracle-threshold) comparator. That protocol is achievable in your scope and would make your results considerably more credible than a single-patient TTP number.

---

### [28] Kiram, Youkana, Saouli, Susto & Kahloul (2026) — *Recurrent Deep RL for Chemotherapy Control under Partial Observability*
*arXiv:2605.02552.*

- **The premise, stated bluntly:** most RL work on chemotherapy assumes **full observability of the patient state — "a condition rarely met in clinical practice."** Formulate as a **POMDP** instead.
- **Setup:** `AhnChemoEnv` from the **DTR-Bench** benchmark suite (a ready-made environment — worth knowing about). Partial observability implemented by **hiding the normal-cell population**, adding **observation noise**, and adding **stochastic perturbations to the state dynamics**. PK/PD inter-patient variability was deliberately *removed* so the comparison isolates partial observability and noise.
- **Methods:** **recurrent TD3 (rTD3)** with an **LSTM history encoder** taking `(a_{t−1}, r_t, o_t)`; baselines **TD3**, **SAC**, and **recurrent PPO (rPPO)** from Stable-Baselines3. Target: `y_t = r_t + γ(1−d_t)·minᵢ Q'ᵢ(τ_{t+1}, a'_{t+1})`.
  - **Training detail worth stealing:** continuous episodes stored, **overlapping subsequences randomly sampled**, hidden states **initialised to zero at each subsequence start rather than using a burn-in** — avoids stale hidden states in off-policy updates.
  - 10 shared random seeds; **4×10⁵ environment steps**; evaluation every 5×10³ steps over 25 episodes; RTX 4070 Ti.
- **Results:** under **full** observability, rTD3, TD3 and SAC are all competitive — recurrence adds little. Under **partial** observability the gap opens sharply: rTD3 is strongest and most stable, rPPO also stays positive with low variance, while **TD3 and SAC show large oscillations, very high across-seed variance, and unreliable convergence** — temporary gains are lost later in training. The authors are careful that the huge final standard deviations of TD3/SAC "reflect a robustness issue rather than a small difference in average performance." Behaviourally, the recurrent policy produced **more structured dosing, a more stable immune response, and higher, more consistent early drug concentration followed by a synchronised taper**.
- **Take for the project:** (1) Your project is *inherently* partially observable — in reality you observe PSA or imaging-derived volume, not `(S, R)`. This paper says: if you model that honestly, use **recurrence** (LSTM/GRU) or **frame-stacking** (cf. Aif's 64-step stack), and expect memoryless agents to be unstable rather than merely worse. (2) **DTR-Bench** is a real benchmark you could report on for external comparability. (3) The "10 seeds, report variance" discipline appears again — that is now three papers making the same point.

---

### [29] Tordesillas & Arbelaiz (2019) — *Personalized Cancer Chemotherapy Schedule: a numerical comparison of performance and robustness in model-based and model-free scheduling*
*arXiv:1904.01200.*

- **The cleanest head-to-head:** Optimal Control vs **DQN** vs **DDPG** on the same de Pillis-type model.
- **Hyperparameters (a useful starting grid):**
  | | DQN | DDPG |
  |---|---|---|
  | Replay start size | 10³ | 5×10³ |
  | Layers | 2 | 3 |
  | Hidden units/layer | 100 | 300 |
  | Discount γ | 0.99 | 0.995 |
  
  Agent timestep **0.3 days** (chosen for reasonable training time). DQN tested with action-space discretisations of 7, 10, 20, 30, 40, 50 and 60 nodes.
- **Results, in order:**
  1. **With an accurate model, OC is best** — but DQN and DDPG follow closely, and in the preliminary case all three policies **match** (Fig 3b–d).
  2. **Under parametric uncertainty** (sweeping the growth parameter `r₁` over 1.3–1.7), DRL policies degrade far less than OC.
  3. **Under perturbed initial tumour size** (`T₀` trained at 0.7), **OC fails to exterminate the disease in all cases while both DRL methods succeed** — a striking robustness result.
  4. **Under stochastic forcing** of `T(t)`, DQN and DDPG drive the tumour closer to extermination than OC at final time.
  5. **DDPG > DQN**, attributed to the continuous action space.
  6. **Sample efficiency:** ~**1,500 episodes** for DDPG to reach near-optimal cost. This is the only paper that reports an explicit sample-complexity curve against the optimal-control reference — valuable if you need to argue your compute budget is sufficient.
- **Take for the project:** (1) The "OC wins on the nominal model, RL wins under uncertainty" result, replicated here and in [27], is the two-citation justification for your whole approach. (2) If you move to continuous dose, **DDPG/TD3/SAC over DQN** — the continuous action space is the stated reason. (3) 1,500 episodes as an order-of-magnitude expectation for a 2-state ODE environment.

---

### [30] Shiranthika, Chen, Wang, Yang, Sudantha & Li (2022) — *Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning* (SOCR)
*IEEE J Biomed Health Inform 26(9):4763–4772.* doi:10.1109/JBHI.2022.3183854

- **The only offline-RL paper in the corpus** — i.e. learning from retrospective records with **no simulator and no exploration**. This is the only RL paradigm that could be applied to real patient data without a validated model.
- **Data:** a manually constructed database of **40 Stage-IV colon cancer patients** receiving line-1 chemotherapy (Bevacizumab/Cetuximab + Irinotecan/5FU/Oxaliplatin/Xeloda combinations).
- **Algorithm: Conservative Q-Learning (CQL)**, a SAC-based offline method. The problem offline RL solves: naive Q-learning on offline data produces **wildly over-optimistic Q-values**, because minimising Bellman error pushes Q up for actions never observed. CQL adds a regulariser that simultaneously **minimises action values under the current policy and maximises them under the data distribution**, rebalancing the overestimation. MOPO also compared. Hyperparameters grid-searched over learning rate, α, γ, initial α, and target-sync coefficient τ.
- **The "supervised" part:** the RL-optimal policy is **corrected toward the oncologists' historical decisions** with an adjustment factor (0.4 in their experiments), injecting clinical domain knowledge onto the algorithmic output. Framed as a two-way process.
- **Evaluation:** offline policy-evaluation scorers — value scale (`E[maxₐ Q(sₜ,a)]`, flags overestimation), advantage, observational error. Validated by RMSE against the actual prescriptions of **three different oncologists** for six Bevacizumab patients over six cycles; RL-optimal relative dose intensity closely tracked given doses while achieving higher per-cycle and cumulative reward.
- **Take for the project:** Not directly reusable (you have no patient dataset), but two things matter. (1) If your Discussion proposes clinical translation, **offline RL is the honest path** — online exploration on patients is not an option, and this paper names the specific technical obstacle (Q-value overestimation) and a specific solution (CQL). (2) The "supervise the RL policy with clinician decisions" pattern is a sensible safety mechanism to cite.

---

### [30b] Mashayekhi, Nazari, Jafarinejad & Meskin (2024) — *Deep reinforcement learning-based control of chemo-drug dose in cancer treatment*
*Comput Methods Programs Biomed 243:107884.* doi:10.1016/j.cmpb.2023.107884

- **The contribution:** previous RL work on chemo dosing (including Padmanabhan [26]) **discretised** states and actions, which demands expert-guided binning and misrepresents reality. This paper keeps both in their original **continuous** spaces.
- **Algorithm: TD3** (twin-delayed DDPG). Environment: nonlinear pharmacological cancer model, 4 states `[x₁,x₂,x₃,x₄]`.
- **TD3 mechanics, laid out usefully:** six networks — actor, target actor, two critics, two target critics. **Clipped double Q-learning**: target `y ← r + γ·min_{i=1,2} Q'ᵢ(s', ã)` uses the *minimum* of the twin critics to suppress overestimation bias. Actor and all target networks updated only every `d` iterations; soft updates `θ'ᵢ ← τθᵢ + (1−τ)θ'ᵢ`, `φ' ← τφ + (1−τ)φ'`. Experience replay buffer; ε-decay after warm-up episodes. Sensitivity of convergence to **batch size** and **discount factor γ** reported explicitly (Figs 3 and 6).
- **Robustness:** three patient models — nominal, −10%, +10% parameter variation — with the same trained controller; ~30,000 training steps; error-rate learning curves with standard-deviation bands.
- **Limitation they state:** the four-state model, though experimentally validated and widely used, "may have some deviation from real-world conditions."
- **Take for the project:** If you go continuous-dose, this is the reference TD3 implementation recipe and the reference for *why* continuous beats discretised. The batch-size and γ sensitivity figures also tell you which two hyperparameters to sweep first.

---

# Section 5 — Adjacent domains (radiotherapy, immunotherapy, ABM)

*Read these for transferable methodology, especially reward design under competing clinical endpoints and techniques for small datasets.*

### [31] Tseng, Luo, Cui, Chien, Ten Haken & El Naqa (2017) — *Deep reinforcement learning for automated radiation adaptation in lung cancer*
*Med Phys 44(12):6690–6705.* doi:10.1002/mp.12625 — **The canonical "RL with too little clinical data" solution.**

- **Problem:** learn automated mid-course dose adaptation for NSCLC from **only 114 retrospective patients**. Maximise tumour local control (LC) while limiting radiation pneumonitis grade ≥2 (RP2).
- **Three-component architecture — the reusable idea:**
  1. **GAN** to learn the patient-population distribution and generate **synthetic patients**, overcoming the small sample size.
  2. **DNN** trained on real + synthetic data to build a **radiotherapy artificial environment (RAE)** — i.e. learn the **transition probabilities** so the RL agent has an environment to interact with.
  3. **DQN** operating on the RAE to choose the optimal adaptive dose.
- **State:** large-scale patient characteristics — clinical, genetic, imaging radiomics, plus tumour and lung dosimetric variables. **Action:** dose per fraction, freely ranging **1–5 Gy**, applied at **2/3 of the way through the course**. **Reward:** the **uncomplicated cure probability `P+`** as baseline.
- **Results:** with the baseline `P+` reward, the DQN's adaptive fraction doses tracked clinical decisions at **RMSE = 0.76 Gy**, and self-selected a range of **1.5–3.8 Gy**, similar to the clinical protocol — but were systematically **less aggressive** (one-tailed p = 0.0006). Re-weighting `P+` to emphasise avoiding local failure improved agreement to **RMSE = 0.5 Gy**. Against 34 protocol patients constrained by a **17.2% NTCP limit for RP2**: clinicians made 19 good / 15 bad decisions; the DQN made **17 good / 4 bad / 13 "potentially good"**. Classical **TD(0) gave RMSE = 3.3 Gy** — numerically unstable and unable to generalise to unseen states, which the authors attribute to DQN's function approximation plus replay memory.
- **Honest conclusion:** "development of this framework into a fully credible autonomous system for clinical decision support would require further validation on larger multi-institutional datasets."
- **Take for the project:** (1) The **GAN → learned-transition-model → DQN** pipeline is the template for ever moving off a hand-written ODE and onto data. (2) **The reward function had to be re-weighted to match clinical priorities** — an honest, citable demonstration that reward design *is* the clinical-priority specification step. (3) Tabular/classical TD fails where DQN works, for stated reasons.

---

### [31b] Niraula, Jamaluddin, Matuszak, Ten Haken & El Naqa (2021) — *Quantum deep RL for clinical decision support in oncology: application to adaptive radiotherapy*
*Sci Rep 11:23545.* doi:10.1038/s41598-021-02910-y

- **The premise (interesting even if you ignore the quantum part):** physicians decide **under irreducible uncertainty**, so represent the decision itself as an **indeterministic quantum state** rather than a deterministic choice. Formally a 5-tuple `(S, |D⟩, TF, P_{LC|RP2}, R)` where `|D⟩` is a **superposition over all eigen-doses**; a decision is made by amplitude-amplifying the optimal eigen-dose `|d⟩*` (from the deep Q-net) and then **measuring** the amplified state.
- **Data:** 67 stage-III NSCLC patients on prospective adaptive protocols (training) + **174 NSCLC patients externally, multi-institutionally, for validation**. Three frameworks compared: classical DRL, qDRL in a Qiskit simulator, qDRL on a real **IBM quantum computer**. Double Q-learning for the deep Q-net.
- **⭐ Reward function — the best-documented multi-endpoint reward in the corpus:**
  built on `P+ = P_LC·(1 − P_RP2)`, plus a **tiered bonus structure**:
  - **+10** for a **clinically** desirable outcome: `p_LC > 70%` **and** `p_RP2 < 17.2%`
  - **+5** for a **computationally** desirable outcome: `p_LC > 50%` **and** `p_RP2 < 50%`
  - **−1** for failing to achieve either — explicitly "motivates the AI agent to search for the optimal dose as soon as possible"
- **Small-data solution:** **WGAN-GP** (Wasserstein GAN with gradient penalty) generating **4,000 additional training points**, to let the RL algorithm represent the state space adequately.
- **Evaluation — two metrics, since this is not supervised learning:** (i) **similarity score** = RMSE between retrospective clinical decisions and AI recommendations; (ii) a **self-evaluation scheme** comparing clinical decisions and AI recommendations by the improvement in observed clinical outcomes. Q-values are reported as a proxy for the AI's confidence. Conclusion: potential improvement of **at least ~10% over unaided clinical practice**.
- **State features** include an explicit genetic marker (`Cxcr1-Rs2234671` SNP, IL8RA, associated with radiation toxicity) and lung gEUD with `α/β = 4 Gy`, `a = 1`.
- **Take for the project:** Ignore the quantum layer. Take (1) the **tiered reward with explicit clinical thresholds** — this is how you encode "a good outcome means TTP > X *and* cumulative dose < Y" without hand-tuned weights, and it is far more defensible to a clinician than a weighted sum; (2) the **two-metric evaluation** (similarity to standard of care + outcome improvement) as a model for how to evaluate a policy with no ground-truth optimum; (3) WGAN-GP for data augmentation if you ever fit to real trajectories.

---

### [32] Yao, Chen & Zhang (2024) — *Optimized patient-specific immune checkpoint inhibitor therapy based on tumor immune microenvironment modeling*
*Briefings in Bioinformatics* / medRxiv 2024.04.08.24305526. doi:10.1093/bib/bbae547

- **Model:** an ODE model of the **tumour immune microenvironment (TIME)** built from patients' **genomic and transcriptomic data**, capturing chemotherapy, ICIs, immune cells and cancer cells — including **sensitive and resistant cancer populations**. State `Sₜ = {log T_sens, log T_res, log L, η_CD8, μ, r}` where `η_CD8` is the CD8+T-cell-to-tumour-cell ratio.
- **Algorithm: DDPG** — chosen because chemotherapy dose is continuous and because the TIME simulation is expensive and slow, favouring an **off-policy** method. 10,000 training steps. **200 TCGA SKCM patient simulators for training, 30 held out for evaluation.**
- **⭐ Reward — the most sophisticated terminal-reward design in the corpus:**
  - **Immediate reward** = tumour size reduction rate: `rₜ = [(T_s + T_r)_{t−1} − (T_s + T_r)_t] / (T_s0 + T_r0)`
  - **Terminal reward:**
    - tumour eliminated (final population < 1×10²) → **maxed at 1**, and the ICI cost is **not charged at all**
    - treatment failure (final population > 2.6×10⁸) → **`−1 − λ₁·Σₜ Iₜ`** (ICI cost charged in full)
    - **in between → simulate the TIME with no treatment for another six cycles (18 weeks) and score the tumour's re-growth rate.** This is a genuinely clever device: it charges the agent for the *durability* of the remission rather than its instantaneous depth, and the authors state it exists specifically to capture the delayed response of ICI and the evolutionary dynamics of resistant vs sensitive cells.
    - plus `λ₂·(4 − T)` rewarding **early** elimination (before cycle 4)
- **Results:** DRL achieves the **highest overall average reward (1.079)** versus the best fixed schedule (0.922). Clear phenotype-dependent policies: **"extremely cold" tumours** (very low CD8+ infiltration) → **high-dose chemotherapy alone**, ICI adds nothing; **"hot" tumours** → chemotherapy and ICI **simultaneously**; **"cold" tumours** → a **mid-dose of chemotherapy first to warm the TIME**, then ICI. High chemo doses (4–5 units) *reduce* reward by damaging anti-tumour immunity (0.595 vs 0.692 and 0.399 vs 0.596 with vs without ICI). Some cold tumours with fast resistant growth fail regardless.
- **Limitations stated:** still largely theoretical despite real-data calibration; drug variety not modelled; melanoma only.
- **Code:** github.com/yaoyao0111/ICC_decision
- **Take for the project:** **Steal the "simulate forward without treatment and score re-growth" terminal reward.** In your setting, that translates to: at episode end, run the environment untreated for a fixed horizon and penalise the resistant fraction's growth rate — a direct reward for *having left the tumour in a controllable evolutionary state*, which is precisely what adaptive therapy is supposed to do and which TTP alone does not capture. That is a novel, defensible contribution well within your scope.

---

### [33] Lagzian, Razavi & Kardehi Moghaddam (2026) — *A biologically constrained agent-based model of cancer stem cell dynamics with RL-guided adaptive radiotherapy*
*PLOS ONE 21(2):e0340426.* doi:10.1371/journal.pone.0340426

- **Environment:** a 2-D agent-based model, **350×350 lattice**, unit spacing, no-flux diffusion boundaries; a **single cancer stem cell (CSC)** seeded at a random site at t=0; **Δt = 1 hour**, total **480 hours (20 days)**; seed fixed at 42 for reproducibility; calibration/profiling runs on a 100×100 grid. Each cell is an agent with type (stem **C** / non-stem **P**), position, lifespan, and behaviours (migration, division, death). Includes oxygen diffusion, spatial limits, stochastic migration, cell-cycle dynamics.
- **Ten explicit biological constraints:** symmetric/asymmetric CSC division, non-stem proliferation, spatial and resource competition among neighbours, oxygen/nutrient-limited growth, natural death and phenotype switching. Parameter ranges chosen from Grimes et al. (2016) and Norton et al. (2014) for numerical stability and biological plausibility.
- **RL:** tabular **Q-learning** coupled to the ABM, adjusting radiation dose from real-time CSC localisation and microenvironmental feedback. Reward uses weighting factors `w₁`, `w₂` (tuning procedure in supplementary S2).
- **Validation:** simulated tumour area growth compared against *in vivo* CSC-driven progression data (Norton et al.) — reproduces the characteristic quiet phase → near-exponential growth → terminal plateau (hypoxia, space, nutrient depletion).
- **Result:** **RL-guided adaptive radiotherapy reduced CSC count by 69%** (98 → 30 cells) versus **65%** for a constant-dose **2 Gy × 8** protocol.
- **The authors' own honesty is exemplary and worth imitating:** the RL component is "currently presented as a proof-of-concept to demonstrate feasibility, and its optimization and convergence analysis will be explored in future studies"; 2-D only; no immune cells, angiogenesis or molecular signalling; unlimited CSC proliferation assumed deliberately to isolate stem dynamics.
- **Take for the project:** A modest result (69% vs 65%) reported with complete transparency about scope — a good tonal model for a course project. Also: if you want a cheap second environment for a transfer test, a **small stochastic lattice ABM** like this is very implementable, and it introduces exactly the spatial effects Aif [20] showed break ODE-trained policies.

---

### [33b] Zhu, Zhu, Wang, Xu, Wu, Li, Ma & Cao (2026) — *Deep learning for adaptive chemotherapy: a DDPG-based approach to optimizing tumor-immune dynamics*
*PLOS ONE 21(4):e0345877.* doi:10.1371/journal.pone.0345877

- **Environment:** nonlinear de Pillis-type tumour / normal / immune cell system with drug concentration as the control input; **Gaussian noise added** to simulate physiological oscillation and treatment-response uncertainty. Parameters from de Pillis: `ρ=0.01, c₁=0.3, c₂=0.165, c₃=1.0, c₄=0.3, d=0.2, d₂=1.0, r=1.5, b=0.3, a₁=0.2, a₂=0.3, a₃=0.1`.
- **⭐ Complete DDPG hyperparameters (the most reproducible configuration in the corpus — use as your default starting point):** actor and critic both MLP with **two hidden layers of 256 units**; learning rate **0.001** for both; replay buffer **1,000,000**; batch size **256**; discount **γ = 0.99**; soft-update **τ = 0.005**; Gaussian exploration noise.
- **Reward design:** a **state-aware early-termination** rule — if tumour count exceeds a safety threshold or normal cells fall below a critical value, the episode is **dropped immediately** (no further learning or compute spent on it). The normal-cell term uses a function whose gradient → ∞ as normal cells → 0⁺, giving "a powerful learning signal to avoid lethal toxicity," with diminishing marginal reward as normal-cell counts become healthy — so the agent doesn't over-prioritise normal cells at the expense of tumour suppression. A reward-function optimisation ablation is shown (Fig 7).
- **Benchmarks:** fixed-dose, **A2C, PPO, SAC**. Results (RMAE on tumour cells): **DDPG 0.157**, A2C 0.248, SAC 0.318, PPO 0.381. **A2C failed to converge** on this continuous-control task (uncontrolled tumour proliferation); PPO slowed but did not reduce tumour cells and suppressed normal-cell recovery. SAC matched DDPG on final tumour suppression and converged faster early (entropy-driven exploration) but **suppressed normal cells more** — below 0.7 at the lowest point — so DDPG was preferred on the efficacy/safety trade-off. Monte Carlo sensitivity analysis ranked parameters by Pearson correlation with tumour shrinkage rate; initial tumour burden × treatment duration synergy analysed.
- **Take for the project:** (1) The hyperparameter table is your copy-paste default for any continuous-action agent. (2) **A cross-algorithm comparison (DDPG/SAC/PPO/A2C) on one environment is a self-contained, achievable result** — and this paper shows the outcome is not obvious (A2C fails outright). (3) The **asymptotic-gradient toxicity penalty** is an elegant way to encode "never cross this line" in a shaped reward. (4) **Early episode termination on safety violation** is a cheap and genuinely useful trick.

---

# Part III — Cross-paper synthesis

## Table A — Environments and models

| # | Model class | Populations / state | Drug acts by | Competition | Stochastic? | Key parameters |
|---|---|---|---|---|---|---|
| 03 | LV competition (3-sp) | T+, TP, T− | reducing **carrying capacity** | full 3×3 matrix α, αᵢᵢ=1, 0<αᵢⱼ<1 | no | r=[.278,.355,.665]/d ×0.1; K_TP 10⁴→10²; K_T+=1.5x_TP→0.5x_TP |
| 04 | LV competition (2-sp) | S, R | (as 03) | α_RS = **6** (fitted) | no | rS=.0156, rR=.0091 /d; β_SC=8 |
| 07 | **LV + turnover + Norton–Simon** | S, R | **killing dividing cells** (×2 mitosis factor) | all coefficients = 1 | no | rS=.027/d; d̂D=1.5; r̂R,d̂T∈[0,1]; n₀∈[.1,1]; f_R∈[1e-5,.25] |
| 10, 11 | **Replicator** (EGT, frequency only) | GLY, VOP, DEF | penalising GLY fitness | public-goods payoffs | 11: **yes** (Brownian fitness noise σ=0.15) | ba=2.5, bv=2, c=1, n=4, dmax=3 |
| 11 | LV variant (Carrère) | zS, zR | `−α·zS·d` | explicit **−β·zS·zR** | yes (σS=σR=0.15) | gS, gR, m (size ratio), C, α, β |
| 12 | LV competition (3-sp) | T+, TP, T− | K_TP(Λ)=10⁴−9900Λ; μ(Λ)=1.5−Λ | **α₃₂ = 2 (>1 required)** | no | r ×0.01 scaling; viability Σx ≤ 9000; stabilising Λ=0.4 / 0.4848 |
| 13 | Replicator, frequency-dependent | T+, TP, T− | switching payoff matrix + Kᵢ weights | payoff bᵢⱼ per drug | no | bᵢⱼ listed per treatment; Kmax=1.5e4 |
| 14 | Logistic ODE, 4-pop, 2 drugs | x₁..x₄ by resistance pattern | `−αA·CA·x` terms | shared κ only | no | γᵢ equal or γ₁=γ₂>γ₃=γ₄; threshold κT |
| 15 | **SDE**, d cell types + mutation | x₁..x_d | dose–response ρᵢ,α | implicit | **yes** | P_step,mut=1e-6; P_epis,mut=0.3; Imax=8·max ρ |
| 16,17 | LV + turnover (= 07) | S, R | as 07 | =1 | no (probabilistic policy only) | cost=1−rR/rS; turnover=dS/rS |
| 18 | **Fitness landscape + Markov chain (SSWM)** | 16 (→1024) genotypes | choice among 15 drugs | — | yes (SSWM transitions) | horizon 20 steps; noise σ²∈[0,100] |
| 19 | **Fitness seascape + ABM + PK** | 16 genotypes × concentration | dose–response per genotype | — | yes | t½, Cmax, rm, K; p_miss |
| 20 | LV, **asymmetric**, pop-independent drug | S, R | `−D` (independent of population) | c acts **only on R** | yes (δ) | K=18,000; Bayesian-calibrated to PhysiCell |
| 21 | MSABM → **Fokker–Planck PINN surrogate** | TC, TAM (M1/M2), quiescent TC | CSF1R_I, IGF1R_I | multiscale | yes | 550 core-h → <18 core-h; RMSE<0.06, R²>0.97 |
| 22 | LV, **cytostatic** drug | S, R | `rS(1 − dD·D)` — reduces **proliferation** | α, β (set to 1) | policy-level only | Strobl fits, 7 patients; T=30 d cycle |
| 23 | **tM-GLV + memory** | S, R + 2 intermediate states | `R̃(D)`, B=diag(β₁,β₂) | density-dependence η | yes | **memory window 6 d = 2× cycle**; Φ(I(t)) |
| 24 | Evolutionary **Stackelberg** game | sensitive, resistant + resistance trait u | dose D_G, D_C | α_RS=.60, α_SR=.25 | spatial 100×100 grid | C₁=.51,C₂=.08,C₃=.41; progression δ=0.7·Kmax |
| 25 | **Moran process**, fixed N | S, R₁, R₂ on simplex (N=50) | payoff matrix selection | payoff matrix M | **yes** (finite-population drift) | volume ∝ (mean fitness − baseline) |
| 25b | EGT + **PK**, healthy cells | healthy, sensitive, resistant | concentration-mediated | West payoffs | no | cmax=0.6/0.79; Tmax=0.4/0.5 |
| 26,29,30b,33b | de Pillis tumour–immune–normal | N, T, I, C | `−aᵢ·x·C` | logistic | 33b: **yes** (Gaussian) | a₁=.2,a₂=.3,a₃=.1; b₁=b₂=1 |
| 27 | 2-compartment TGI, ×2 tissues | P, Q (cancer) + P, Q (bone marrow) | kills P only | — | no | γ,δ,α,β,ε; trade-off weight b |
| 31,31b | Learned transition model (GAN/DNN) | clinical+genetic+radiomic+dosimetric | dose/fraction 1–5 Gy | — | yes | NTCP limit 17.2% for RP2 |
| 32 | ODE TIME from -omics | log T_sens, log T_res, log L, η_CD8, μ, r | chemo + ICI | immune-mediated | yes | elimination <1e2, failure >2.6e8 |
| 33 | **ABM lattice** 350×350 | CSC (C), non-stem (P), O₂ field | radiation dose | 10 explicit constraints | yes | Δt=1 h, 480 h, seed 42 |

## Table B — RL algorithms, state, action, reward

| # | Algorithm | State / observation | Action | Reward | Discount & budget |
|---|---|---|---|---|---|
| 15 | **DDPG** | full composition `(x₁..x_d)`, unobserved types = 0 | **continuous** dose, 1 or 2 drugs, [0, Imax] | terminal success scaled by cumulative dose; terminal failure scaled by log residual burden; **potential-based shaping** on growth rate; per-step quadratic dose penalty; drug-specific weights w₁=16, w₂=48 | 50k ep / ~1M samples (1 drug); 130k ep / 2.6M (2 drugs) |
| 16,17 | **A3C** (LSTM→128→64→32→16→10) | current tumour size `V(t)` | **binary**: treat / holiday | **+0.1/step survived; +0.05/step holiday; −0.1 progression; +5 at 30-y survival** | **γ = 0.999**; 100 eval runs |
| 18 | **DQN** (RL-genotype: 2×conv+pool+dense28; RL-fit: dense 64,28) | genotype **or** fitness + one-hot prev action | choice of **15 drugs** | **R = 1 − f(s\|a)** | finite horizon 20 steps; γ swept 0→1 |
| 20 | **PPO** (MLP 2×128) | **stacked past 64 burden values + past 64 actions** | binary treat / pause | custom; total normalised burden in sliding window | 5×10⁶ steps, ~4,900 updates, ~10⁵ episodes, updates every 1,024 |
| 21 | **A3C** + LSTM feature extractor (→4-D) | **survival probability and its change** (2-D time series) | two drugs (softmax, sampled) | per algorithm S2 | on PINN surrogate |
| 22 | **PPO** (clipped) | tumour burden at 30-day decision points | **treatment-window length** per cycle: {0,30}, {0,15}, {0,19}, or full 0–30 days | +0.1/day survived, +0.05/non-treatment day, −0.1 progression (from [16]) | 100,000 iterations |
| 23 | **SAC** | population counts **+ R:S ratio** | continuous treatment intensity | efficacy + transition + cumulative-dose penalty + **PFS component in step reward** | per-patient tuning |
| 24 | **Q-learning** | burden + resistance level u | dose split (D_G, D_C), `D_G+D_C<1` | **(x_prev − x) − 0.08·(D_G+D_C) − 0.15·u** | — |
| 25 | **tabular Q-learning** | `(S,R₁,R₂)` composition on simplex | discrete doses of C₁/C₂ | **Shannon entropy** of composition; +1 within 0.05% of max entropy, −1 near boundaries (2/3/4-region variants) | α, γ tunable |
| 26 | **tabular Q-learning**, ε-greedy | discretised tumour count x₂ | discretised infusion rate, patient-specific umax | **fractional error improvement**, 0 if no improvement | 50,000 scenarios |
| 27 | **double DQN** | tumour states + periodic **bone-marrow density** | dose f(t) | objective `∫[P_bm+Q_bm − b(P_bc+Q_bc)]dt` | 600 test patients @15/20/25% perturbation |
| 28 | **recurrent TD3** (LSTM over (a,r,o)) vs TD3/SAC/rPPO | **noisy, partial** (normal cells hidden) | continuous dose | DTR-Bench AhnChemoEnv | 4×10⁵ steps, 10 seeds |
| 29 | **DQN** (2×100) & **DDPG** (3×300) | de Pillis states | DQN: 7–60 discretisation; DDPG: continuous | model objective | γ 0.99 / 0.995; Δt 0.3 d; ~1,500 ep to near-optimal |
| 30 | **CQL** (offline, SAC-based) | clinical features, 40 real patients | drug combination + dose | clinical reward, then **supervised toward oncologist decisions** (factor 0.4) | grid-searched |
| 30b | **TD3** | continuous 4-state | **continuous** dose | error-based | ~30,000 steps; batch & γ sensitivity reported |
| 31 | **DQN** on GAN+DNN-learned environment | clinical+genetic+radiomics+dosimetry | dose/fraction 1–5 Gy | **P+** (uncomplicated cure probability), re-weighted variant | 114 patients + synthetic |
| 31b | **double DQN** + quantum decision states | as 31 + SNP, gEUD | eigen-dose from superposition | **P+ = p_LC(1−p_RP2)**, **+10** if p_LC>70% & p_RP2<17.2%, **+5** if >50%/<50%, **−1** otherwise | 67 train + 174 external; WGAN-GP 4,000 synthetic |
| 32 | **DDPG** | log tumour (sens/res), log immune, CD8 ratio, μ, r | continuous chemo + ICI dose | immediate = burden reduction rate; terminal = 1 if eliminated (dose free), −1−λ₁ΣI if failed, else **score re-growth over 6 untreated cycles**; +λ₂(4−T) for early elimination | 10,000 steps; 200 train / 30 test simulators |
| 33 | **tabular Q-learning** | CSC localisation + microenvironment | radiation dose | weighted w₁, w₂ | proof-of-concept |
| 33b | **DDPG** (MLP 2×256) vs SAC/PPO/A2C | tumour, normal, immune, drug | continuous dose | asymptotic normal-cell term + **episode dropped on safety violation** | γ=0.99, τ=0.005, lr 1e-3, buffer 1e6, batch 256 |

## Table C — Baselines and metrics

| Baseline | Definition | Used by |
|---|---|---|
| **MTD / CT** | `D(t) = Dmax` ∀t until progression | essentially all |
| **AT50** | `Dmax` until N < 0.5N₀; 0 until N ≥ N₀ | 03, 04, 07, 16, 17, 20, 22 |
| **AT-1 (dose modulation)** | ±50% dose per ±20% volume change; skip below cutoff | 02, 09, 24 |
| **AT-2 (dose skipping)** | fixed dose if volume ↑≥25% vs two measurements prior | 02, 09 |
| **Intermittent (IT)** | fixed on/off periods, not response-triggered | 03, 09, 22, 25b |
| **AT-FSC / AT-PSC** | single-threshold, full vs partial treatment window per 30-d cycle | 22 |
| **Dose titration (upward)** | start low, increase until stabilisation | 12, 09 |
| **CM regimen** | closed-form dose maintaining `Σx = κT` | 14 |
| **Optimal control (HJB / PMP / FBSweep / APOPT)** | model-based optimum — the ceiling | 10, 11, 12, 14, 25b, 27, 29 |
| **MDP by backward induction** | perfect-information optimum, finite horizon | 18 |
| **Random / naive drug cycling** | — | 18 |
| **Other RL algorithms** | A2C, PPO, SAC vs DDPG; TD3 vs rTD3 | 28, 33b |

| Metric | Definition as used | Papers |
|---|---|---|
| **TTP / TTF** | first t with `N(t) ≥ 1.2·N₀` (RECIST-analogous 20%) | 07, 16, 17, 22 — **the field standard** |
| | first t with `N > δ·Kmax`, δ=0.7 | 24 |
| | radiographic progression (PCWG2/RECIST), clinical | 03, 04 |
| | generations to treatment failure | 20 |
| **Overall survival** | months | 04 |
| **Cumulative dose** | % of SOC; `∫d dt` | 02, 03, 04, 10, 22, 24 |
| **Time off treatment** | % of trial duration | 04 |
| **Progression-free survival** | Kaplan–Meier, log-rank | 02, 17, 22, 23, 33 |
| **Cost distribution / CDF** | P(cost ≤ budget) — not just the mean | **11** |
| **Cross-environment performance difference** | Cohen's d between training and reference env | **20** |
| **Scaled score vs per-patient optimum** | RL score ÷ patient-specific optimal-control score | **27** |
| **Similarity to clinician decisions** | RMSE in Gy or dose units | 30, 31, 31b |
| **Population entropy / coexistence** | Shannon H of composition | **25** |
| **Robustness** | ±10–25% parameter perturbation; noise sweeps; measurement delay | 18, 26, 27, 29, 30b |

## Table D — Consensus, disagreement, and gaps

**Where the literature agrees:**
1. MTD is evolutionarily unwise when pre-existing resistance is present; competitive release is the mechanism. (all)
2. Optimal policies under a dose-plus-time objective are **bang-bang**. (10, 11, 14)
3. Adaptive therapy does **not** require a cost of resistance; competition near carrying capacity suffices. (07, 20, + lattice models)
4. **Higher withdrawal thresholds generally extend TTP** — treat less, hold at higher burden. (09 §2.2, 17, 22)
5. RL ≈ optimal control on a known model, RL > optimal control under uncertainty. (27, 29)
6. Reward structures cluster on: survival/TTP + holiday bonus + progression penalty + dose penalty. (16, 17, 22, 24, 15)
7. **Decision/monitoring frequency is a first-order design variable**, not an implementation detail. (16, 17, 22)

**Where it disagrees:**
- **Dose modulation vs dose skipping:** mouse data favours modulation (02); some ABMs favour skipping (09 §2.2); ABM two-drug work finds no strong sequential-vs-joint preference (14 citing Thomas et al.).
- **Is MTD ever better?** Models with size-proportional death rates find MTD superior to dose-skipping AT (09 §2.2 citing ref. 75).
- **Competition coefficient magnitude:** Zhang assumes `0 < αᵢⱼ < 1`; Cunningham shows containment requires `α₃₂ > 1`; co-culture estimates reach **12.6**; Zhang 2022's fitted `α_RS = 6`.
- **Cytotoxic vs cytostatic drug action:** kill dividing cells (07, 16) vs reduce proliferation rate (22) vs reduce carrying capacity (03, 12) vs population-independent kill (20). **This choice changes the answer** — state yours explicitly.
- **Does the clinical evidence support AT at all?** (05a says no; 04, 05b say yes with caveats.)

**Gaps you could occupy:**
| Gap | Evidence it's open | Feasible for your project? |
|---|---|---|
| Cross-environment / transfer validation of RL policies | only 20 does it; 22 does not | **Yes — highest value** |
| Reward-design comparison on a fixed environment (survival-based vs entropy/coexistence vs regrowth-penalised) | 16/22 vs 25 vs 32 use incompatible environments | **Yes — clean and self-contained** |
| Partial observability with a realistic biomarker (PSA, monthly) | 28 does POMDP but not in an evolutionary model | **Yes** |
| Distributional / budget-constrained objectives in RL | 11 does it in optimal control only | Maybe — a CVaR variant |
| Reversible phenotypic plasticity | only 23; named open in 09 §1.3 | Stretch |
| Cost/benefit of maintaining high burden (mutation, metastasis risk) | 09 §3.2: "models have not considered" this | Stretch — needs a mutation term |
| Multi-lesion / spatially distributed disease | 09 §3.2; 20's spatial findings | No — out of scope |

---

# Part IV — Proposed model specification for this project

Every value below is traceable to a paper in Part II. This is a starting configuration, not a constraint — but starting here makes your results directly comparable to Gallagher [16,17], Ma [22], Strobl [07] and Aif [20], which is worth a great deal.

## IV.1 Environment — two-population Lotka–Volterra with turnover

Adopt **Strobl et al. 2021 [07]** in the form Gallagher [16] writes it:

```python
# dS/dt = rS * S * (1 - (S+R)/K) * (1 - dD * D(t)) - dS * S
# dR/dt = rR * R * (1 - (S+R)/K)                   - dR * R
# N(t)  = S(t) + R(t)
```

**Why this model:** it is the minimum structure that contains the phenomenon (two competing types, shared resource, drug acting on one), it has published parameter fits to 67 real patients, and four other RL papers use it — so your numbers mean something to a reader.

**Baseline parameters**

| Parameter | Value | Source |
|---|---|---|
| `rS` | 0.027 /day | [07] (from [03]) — the field's standard time scale |
| `rR` | `rS·(1 − cost)`, cost ∈ [0, 0.5] | [07] sweep; [04] fitted rS=0.0156, rR=0.0091 ⇒ cost ≈ 0.42 |
| `dS = dR = dT` | `turnover·rS`, turnover ∈ [0, 0.5] | [07]; [16] defines turnover as `dS/rS` |
| `K` | 1.0 (normalised) | [07] non-dimensionalisation (K as density scale) |
| `dD` | 1.5 | [07] |
| `n₀ = N₀/K` | 0.75 (near capacity) | [07] sweep [0.1, 1] |
| `f_R = R₀/N₀` | 0.01 | [07] sweep [1e-5, 0.25] |
| Competition | `α = β = 1` initially | [07], [22] — then sweep |
| Integration | RK45 / `solve_ivp`, dt ≤ 1 day | — |

**Progression (episode termination):** `N(t) ≥ 1.2·N₀` — RECIST-analogous, used by [07], [16], [17], [22]. Also terminate at a time cap (e.g. 30 years, per [16]'s survival bonus).

**Virtual patient cohort:** a grid over **(cost of resistance, turnover)** — this is exactly Strobl's and Gallagher's parameter space, so your cohort is interpretable and your robustness heatmap is directly comparable to Gallagher's Fig 5c. Add 7 clinically fitted patients from [22] / [07]'s Bruchovsky fits if you can get them (github.com/Kexin0902/AT-PSC has the parameter table).

## IV.2 MDP formulation

**State.** Run three variants — this comparison is itself a result:
- `s₁ = [N(t)/N₀]` — burden only. Matches [16], [17]. The honest clinical case (you observe a biomarker, not composition).
- `s₂ = [S/K, R/K]` — full composition. The oracle case. [13] and [23] argue composition is the real control target; this tests that claim.
- `s₃ = [N/N₀ history (last k steps), action history (last k steps)]` — frame-stacked. [20] uses k=64. Gives the agent the derivative and phase information that a single burden value hides, without requiring unobservable composition.

Optionally add `t/T_max` (finite-horizon awareness — [18] uses an explicit finite horizon and the policy becomes greedy near the end).

**Action.**
- **Primary — binary:** `D ∈ {0, Dmax}`. Justified by the **bang-bang optimality result** of [10] and [11]; used by [16], [17], [22]; clinically legible.
- **Secondary — treatment-window length:** `D ∈ {0,…,30} days` within a 30-day surveillance cycle, per [22]. This is the more clinically realistic action and where [22] found its gains.
- **Stretch — continuous:** `D ∈ [0, Dmax]`, requiring DDPG/TD3/SAC per [29], [30b], [33b].

**Decision interval.** Make this an explicit experimental axis — it is *not* a detail. [16] found training at a lower frequency produces better policies, and that evaluating below the training frequency fails catastrophically. Suggested values: **{7, 14, 28, 56, 81} days** (81 d is the interval at which AT50 is provably optimal in [16], so it is a meaningful anchor).

**Reward — start from the Gallagher/Ma consensus, then compare alternatives.**

*R1 — baseline, from [16]/[17], replicated by [22]:*
| Term | Value | Circumstance |
|---|---|---|
| Base | +0.1 | per timestep survived |
| Holiday | +0.05 | per timestep with no treatment |
| Progression | −0.1 | on `N ≥ 1.2·N₀` |
| Survival | +5 | on reaching the time cap |

with **γ = 0.999**.

*R2 — dose-and-time cost, from [10]:* `r = −D(t)·Δt − σ·Δt` with a large terminal bonus/penalty. This is the exact negation of Gluzman's provably optimal objective, so **an agent trained on R2 should recover the HJB bang-bang policy** — that is a verifiable correctness check on your whole pipeline.

*R3 — coexistence, from [25]:* `r = +1` if the S:R composition is near maximum Shannon entropy, `−1` near the boundaries. Rewards containment directly rather than via survival time.

*R4 — durability, from [32]:* at episode end, simulate the environment **untreated** for a fixed horizon and penalise the resistant population's growth rate over that window. Rewards leaving the tumour in a *controllable evolutionary state*, which TTP alone does not capture.

Add from [15] if credit assignment proves hard over long episodes: **potential-based reward shaping** `r_shape = γΦ(s') − Φ(s)` with Φ built from the population growth rate — provably policy-invariant, so it cannot change the optimum, only the learning speed. Add from [24] if your state includes R: a direct resistance penalty `−β·R/N`.

## IV.3 Algorithm

**Start with PPO.** Reasons: [20] and [22] both use it on almost exactly this problem; it is stable with binary/discrete actions; Stable-Baselines3 gives you a trustworthy implementation.

**Architecture (from [20]):** policy and value networks, MLP, **2 hidden layers × 128 units**. If using the history-stacked state `s₃`, either frame-stack into the MLP (as [20] does) or use an LSTM encoder (as [16], [28]).

**Budget (from [20], [22]):** 10⁵–10⁶ environment steps; policy updates every 1,024 steps. [22] used 100,000 PPO iterations. [29] suggests ~1,500 episodes may already reach near-optimal on a simple ODE environment — so start small and check the learning curve before committing compute.

**Non-negotiable: train ≥10 agents with different seeds and report the spread.** [20], [28] and [33b] all show that single-seed results in this domain are misleading, and [28] shows memoryless agents can be *unstable* rather than merely worse.

**If you extend to continuous dose** — DDPG or TD3, with [33b]'s configuration as the default: MLP 2×256, lr 1e-3, buffer 1e6, batch 256, γ=0.99, τ=0.005, Gaussian exploration noise. [30b] documents the TD3 mechanics (clipped double-Q, delayed actor updates) and reports batch size and γ as the two most convergence-sensitive hyperparameters.

## IV.4 Baselines (all of these, not a subset)

1. **No treatment** — the floor.
2. **MTD / CT** — `D = Dmax` always.
3. **AT50** — the clinical protocol from [03].
4. **AT-x threshold sweep** — withdrawal threshold swept over {25%, 50%, 75%, 90%}, since [09] §2.2 and [17] both predict higher thresholds do better. Report the best-performing threshold as the *tuned* heuristic ceiling — this is a much stronger baseline than AT50 alone, and beating it is a real result.
5. **Intermittent therapy** — fixed 30-on / 30-off, per [22].
6. **AT-PSC** — partial-cycle, `T₁ ≈ 13` days of 30, per [22].
7. **Optimality ceiling** — for the binary-action, low-dimensional case, compute the optimal policy directly by **value iteration on a discretised (S,R) grid**, per [18]'s backward-induction design. Reporting "the agent reached 96% of the computable optimum" is far more informative than "the agent beat AT50."

## IV.5 Metrics

**Primary:** TTP (days to `N ≥ 1.2N₀`). Report **median and full distribution** across seeds and across the virtual cohort — [11] and [27] both show the mean hides what matters.

**Secondary:** cumulative dose (and as % of MTD, per [03], [04]); fraction of time off treatment ([04] reports 46%); Kaplan–Meier PFS curves with log-rank tests across the cohort ([17], [22]); maximum tumour burden reached (a quality-of-life and mutation-risk proxy, per [09] §3.2); final resistant fraction.

**Robustness (this is where your project earns its credibility):**
- Parameter perturbation at ±15/20/25%, evaluated on a held-out virtual cohort, scored against the per-patient optimum — the [27] protocol.
- Observation noise sweep on the biomarker — the [18] protocol.
- Measurement delay — the [18] protocol; clinically, PSA reported days after the draw.
- Decision-frequency mismatch between training and evaluation — the [16]/[17] protocol.
- **Cross-environment transfer** — see IV.6.

**Interpretability (do not skip this):**
- Plot `P(treat)` against tumour burden and **fit a sigmoid**, per [16]/[17]. Report the extracted threshold `N*` and compare it to AT50's implied threshold and to the analytically optimal threshold for your decision interval.
- Plot the policy over the `(S, R)` plane and check for a **bang-bang partition** into drugs-on and drugs-off regions, as [10] and [11] predict.
- Check for the two non-obvious behaviours that optimal control predicts: **delayed treatment initiation** and **upward dose titration** ([12]). If your agent discovers either unprompted, that is a figure.

## IV.6 The transfer test — build this in from week one

[20] is unambiguous: a policy that is excellent in its training environment can be mediocre or worse in a more realistic one, and the *ranking* of agents does not survive the move. Pick at least one second environment:

| Option | Cost | What it tests |
|---|---|---|
| Exponential-growth variant with resistant-only progression criterion (Lu et al., used by [16]) | trivial | growth-law and endpoint misspecification |
| [22]'s **cytostatic** drug term `rS(1 − dD·D)` instead of Strobl's cytotoxic kill | trivial | drug-mechanism misspecification — a genuine open disagreement (Table D) |
| [11]'s Carrère SR model with explicit `−β·zS·zR` | small | competition-structure misspecification |
| Add a mutation term `S → R` at rate μ | small | [07]'s no-mutation assumption |
| **Stochastic Moran process** on a fixed-N simplex, per [25] | moderate | demographic stochasticity and drift-driven extinction |
| Small lattice ABM, per [33] | large | spatial effects — the ones [20] found actually break policies |

Report train-environment and reference-environment performance for **every** agent, and the correlation between them — the [20] protocol. A negative or weak correlation is not a failed project; it is the finding.

## IV.7 Suggested experiment sequence

| # | Experiment | Answers | Effort |
|---|---|---|---|
| 0 | Implement [07]'s ODE; reproduce AT50 = 672 d, CT = 451 d from [16] | Is my environment correct? | S |
| 1 | Baseline suite across the (cost, turnover) grid | Where is there room to improve at all? | S |
| 2 | Value iteration on a discretised (S,R) grid → optimality ceiling | What is the best achievable? | M |
| 3 | PPO with R1 on one patient, 10 seeds; vs all baselines | Does RL beat tuned heuristics? | M |
| 4 | Extract the sigmoid; compare `N*` to AT50 and to the analytic optimum | Is the policy interpretable and does it match theory? | S |
| 5 | State ablation: `s₁` vs `s₂` vs `s₃` | Is burden alone sufficient? (tests [13]/[23]) | M |
| 6 | Decision-frequency sweep, train × evaluate matrix | Does [16]'s asymmetry replicate? | M |
| 7 | Cohort training (generalist) vs per-patient (specialist) | Does [17]'s trade-off replicate? | M |
| 8 | Reward comparison: R1 vs R2 vs R3 vs R4 on a fixed environment | **Does the objective change the policy? (a genuine gap)** | M |
| 9 | Robustness: parameter perturbation, noise, delay | Does [27]'s variance result replicate? | M |
| 10 | **Cross-environment transfer** | Is any of this real? | M–L |
| 11 | *(stretch)* continuous dose with TD3/SAC; or add mutation; or POMDP with a PSA-like observation | novelty | L |

Experiments 0–4 are a complete, defensible project. 5–10 are what make it good. 11 is publishable-adjacent.

## IV.8 Things to get right that are easy to get wrong

- **Normalise the state.** Feed `N/N₀` and `S/K, R/K`, not raw cell counts. Neural networks do not handle 10⁴-scale inputs well.
- **Non-dimensionalise time** using `rS` as the scale, per [07]. Then your results transfer across parameter sets.
- **Watch the discount factor.** γ=0.999 at a 28-day step means an effective horizon of ~1,000 steps ≈ 76 years. If your episodes are 100 steps, γ=0.999 is effectively undiscounted — which may be what you want (it is what [16] wants), but know it.
- **Probabilistic policies give stochastic outcomes even in a deterministic environment** — [16] averages 100 evaluations, [22] runs 100 stochastic simulations. Always average.
- **Progression must be checked continuously, not only at decision points**, or a fast-growing tumour will progress and recover between observations and you will silently overestimate TTP. [16] traces its low-turnover failures to exactly this.
- **Fix and report your seeds.** [33] fixes SEED = 42 and says so.
- **Log the whole trajectory**, not just TTP — you need `S(t)`, `R(t)`, `D(t)` to produce the interpretability figures, and re-running is expensive.

---

# Part V — Limitations to state in your report

Write these yourself, but these are the ones a reader in this field will look for:

1. **In-silico only, in an environment you chose.** [20] demonstrated concretely that policy quality is a property of the training environment, not of the disease. Say so, and show your transfer test.
2. **Two-population, well-mixed, no space.** Real tumours are spatially structured; spatial structure can *increase* the effective cost of resistance by trapping resistant cells in the core ([09] §2.2), and growth-driven collective motion accelerates resistance escape ([20]).
3. **Resistance is pre-existing, binary and irreversible.** Justified by [07] (citing Viossat & Noble), but [23] shows reversible, memory-driven plasticity produces different dynamics, and [09] §1.3 names plasticity as an open question.
4. **No pharmacokinetics.** Dose is treated as instantaneous effect. [19] shows PK and adherence matter a great deal; [25b] includes PK explicitly and finds the optimal policy is structured around reaching `cmax`.
5. **Toxicity is a dose proxy, not a model.** [27] models bone marrow explicitly; [25b] and [26] model normal cells; [31b] uses a real NTCP constraint. A cumulative-dose penalty is a crude stand-in and should be named as one.
6. **Total burden is a surrogate for a surrogate.** Trials measure PSA/LDH, PSA is a proxy for burden, burden is summed across lesions that need not respond proportionately ([09] §3.2, [13]).
7. **The clinical evidence base is contested.** [05a] — no randomised trial has demonstrated AT superiority; the pilot's comparisons have acknowledged weaknesses ([05b]). Do not write as though AT is established.
8. **Maintaining high burden has costs this model does not represent** — mutation opportunity, possible evolution away from the cost of resistance, metastasis, quality of life. [09] §3.2 states plainly that models have not addressed this.
9. **Reward design encodes clinical priorities and is a modelling assumption**, not a neutral technical choice. [31] had to re-weight `P+` to match clinical behaviour; [24] notes its near-zero-dose result is driven by the toxicity weight `C₃`.
10. **No claim of clinical applicability.** [33]'s framing is the right register: a proof-of-concept demonstrating feasibility, with optimisation, convergence analysis and validation left to future work.

---

# Part VI — Reading order and next actions

**Read in this order (about 3 focused sessions):**

1. **[07] Strobl 2021** — your environment. Read Methods and Figure 4 carefully.
2. **[16] or [17] Gallagher** — your method. [17] is the peer-reviewed version; [16] has more detail in places. Read both Methods sections.
3. **[22] Ma 2026** — your closest competitor. Read to see exactly what has already been done, and to identify what has not.
4. **[03] Zhang 2017** — the clinical trial and the parameter source.
5. **[10] Gluzman 2020** — why bang-bang, and why it matters that adaptive can cure states MTD cannot.
6. **[20] Aif 2025** — read the Discussion before you write any claims.
7. **[05a] Mistry + [05b] response** — five minutes, and it will keep your report honest.
8. **[09] West 2023 §2.2 and §3.2** — for your Discussion.

Skim the rest as needed; use Part II as the index.

**Immediate next actions:**
1. Re-download the three broken files (Part 0.1) — Gatenby 2009, Viossat & Noble 2021, Cunningham 2018. Fix `fetch_papers.py` to verify the downloaded PDF's title against the expected title, and to fail loudly rather than substituting.
2. Fill the blank fields in `rl_adaptive_cancer_therapy.bib` from the publisher records (several entries lack authors, volumes and pages; `king_seascapes` is missing a year — it is 2025, Sci Adv 11:eadv1268).
3. Clone the four code repositories — they will save you days:
   - `github.com/MathOnco/AT_costOfResistance_LVModel` — the ODE model and fitting code
   - `github.com/Kexin0902/AT-PSC` — the RL + protocol suite, and the 7-patient parameter table
   - `github.com/cunninghamjj/Evolution-based-mathematical-models-...-in-mCRPC` — trial PSA data
   - `github.com/DavisWeaver/evo_dm` — the MDP/DQN framework
4. Implement Experiment 0 and check you reproduce **AT50 = 672 days, CT = 451 days**. Until that matches, nothing downstream is trustworthy.

---

## Appendix — Full citation list

**Section 1 — Evolutionary foundations**
1. Gatenby RA, Brown JS. The Evolution and Ecology of Resistance in Cancer Therapy. *Cold Spring Harb Perspect Med.* 2020;10:a040972. ⚠️ *(file present; Gatenby RA, Silva AS, Gillies RJ, Frieden BR. Adaptive Therapy. Cancer Res. 2009;69(11):4894–4903. doi:10.1158/0008-5472.CAN-08-3658 — missing)*
2. Enriquez-Navas PM, et al. Exploiting evolutionary principles to prolong tumor control in preclinical models of breast cancer. *Sci Transl Med.* 2016;8(327):327ra24. doi:10.1126/scitranslmed.aad7842
3. Zhang J, Cunningham JJ, Brown JS, Gatenby RA. Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer. *Nat Commun.* 2017;8:1816. doi:10.1038/s41467-017-01968-5
4. Zhang J, Cunningham J, Brown J, Gatenby R. Evolution-based mathematical models significantly prolong response to abiraterone in mCRPC and identify strategies to further improve outcomes. *eLife.* 2022;11:e76284. doi:10.7554/eLife.76284
5. Mistry HB. On the reporting and analysis of a cancer evolutionary adaptive dosing trial. *Nat Commun.* 2021;12:316. doi:10.1038/s41467-020-20174-4
6. Zhang J, Cunningham JJ, Brown JS, Gatenby RA. Response to Mistry. *Nat Commun.* 2021;12:329. doi:10.1038/s41467-020-20175-3
7. Strobl MAR, Gallaher J, Robertson-Tessi M, West J, Anderson ARA. Treatment of evolving cancers will require dynamic decision support. *Ann Oncol.* 2023;34(10):867–884. doi:10.1016/j.annonc.2023.08.008 ⚠️ *(file present; Viossat Y, Noble R. A theoretical analysis of tumour containment. Nat Ecol Evol. 2021;5(6):826–835. doi:10.1038/s41559-021-01428-w — missing)*
8. Strobl MAR, West J, Viossat Y, Damaghi M, Robertson-Tessi M, Brown JS, Gatenby RA, Maini PK, Anderson ARA. Turnover modulates the need for a cost of resistance in adaptive therapy. *Cancer Res.* 2021;81(4):1135–1147. doi:10.1158/0008-5472.CAN-20-0806
9. Kaznatcheev A, Peacock J, Basanta D, Marusyk A, Scott JG. Fibroblasts and alectinib switch the evolutionary games played by non-small cell lung cancer. *Nat Ecol Evol.* 2019;3(3):450–456. doi:10.1038/s41559-018-0768-z
10. West J, Adler F, Gallaher J, Strobl M, Brady-Nicholls R, Brown J, Robertson-Tessi M, Kim E, Noble R, Viossat Y, Basanta D, Anderson ARA. A survey of open questions in adaptive therapy. *eLife.* 2023;12:e84263. doi:10.7554/eLife.84263

**Section 2 — Optimal control**

11. Gluzman M, Scott JG, Vladimirsky A. Optimizing adaptive cancer therapy: dynamic programming and evolutionary game theory. *Proc R Soc B.* 2020;287:20192454. doi:10.1098/rspb.2019.2454
12. Wang M, Scott JG, Vladimirsky A. Threshold-awareness in adaptive cancer therapy. *PLOS Comput Biol.* 2024;20(6):e1012165. doi:10.1371/journal.pcbi.1012165
13. Cunningham J, Thuijsman F, Peeters R, Viossat Y, Brown J, Gatenby R, Staňková K. Optimal control to reach eco-evolutionary stability in mCRPC. *PLOS ONE.* 2020;15(12):e0243386. doi:10.1371/journal.pone.0243386 ⚠️ *(duplicated as both 12a and 12b; Cunningham JJ, et al. Optimal control to develop therapeutic strategies for mCRPC. J Theor Biol. 2018;459:67–78. doi:10.1016/j.jtbi.2018.09.022 — missing)*
14. West J, You L, Zhang J, Gatenby RA, Brown JS, Newton PK, Anderson ARA. Towards multidrug adaptive therapy. *Cancer Res.* 2020;80(7):1578–1589. doi:10.1158/0008-5472.CAN-19-2669
15. Widdershins A, Hansen E, Read A, Hohl R. Optimal control theory as a method for designing multidrug adaptive therapy regimens. *npj Syst Biol Appl.* 2026. doi:10.1038/s41540-025-00613-y

**Section 3 — Core RL**

16. Engelhardt D. Dynamic Control of Stochastic Evolution: A Deep Reinforcement Learning Approach to Adaptively Targeting Emergent Drug Resistance. *J Mach Learn Res.* 2020;21:1–30.
17. Gallagher K, Strobl M, Gatenby R, Maini P, Anderson A. Learning to Adapt — Deep Reinforcement Learning in Treatment-Resistant Prostate Cancer. *bioRxiv.* 2023. doi:10.1101/2023.04.28.538766
18. Gallagher K, Strobl MAR, Park DS, Spoendlin FC, Gatenby RA, Maini PK, Anderson ARA. Mathematical Model-Driven Deep Learning Enables Personalized Adaptive Therapy. *Cancer Res.* 2024;84(11):1929–1941. doi:10.1158/0008-5472.CAN-23-2040
19. Weaver DT, King ES, Maltas J, Scott JG. Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance. *PNAS.* 2024;121(6):e2303165121. doi:10.1073/pnas.2303165121
20. King ES, Stacy AE, Weaver DT, Maltas J, Barker-Clarke R, Dolson E, Scott JG. Fitness seascapes are necessary for realistic modeling of the evolutionary response to drug therapy. *Sci Adv.* 2025;11:eadv1268. doi:10.1126/sciadv.adv1268
21. Aif S, Eiche M, Appold N, Fischer E, Citak T, Kayser J. Reinforcement Failing guides the discovery of emergent physical dynamics in adaptive tumor therapy. *bioRxiv.* 2025. doi:10.1101/2025.04.08.647768
22. Liu Z, Zhang J, Hong L, Nie Q, Sun X. Multiscale mathematical model-informed reinforcement learning optimizes combination treatment scheduling in glioblastoma evolution. *Sci Adv.* 2025;11:eadv3316. doi:10.1126/sciadv.adv3316
23. Ma K, Wang N, Yang Z, Cheke RA, Tang B. Single-threshold-guided adaptive cancer therapy with partial-cycle treatment: a mechanistic and reinforcement learning analysis. *PLOS Comput Biol.* 2026;22(6):e1014457. doi:10.1371/journal.pcbi.1014457
24. Ghoreyshi ZS, Debnath S, Olawuni PD, Armstrong AJ, Somarelli JA, George JT. A memory-driven reinforcement learning model of phenotypic adaptation for anticipating therapeutic resistance in prostate cancer. *bioRxiv.* 2025. doi:10.1101/2025.11.03.686133
25. Tavakoli F, Mohammadpur D, Salimi Sartakhti J, Manshaei MH. Reinforcement Learning-Driven Evolutionary Stackelberg Game Model for Adaptive Breast Cancer Therapy. *Math Comput Appl.* 2025;30(6):134. doi:10.3390/mca30060134
26. Giles M, Newton PK. Reinforcement learning for chemotherapy scheduling in a stochastic tumor evolution model. *Phys Rev E.* 2026;113:064404. doi:10.1103/cbkw-5s4b
27. Li Z, Tan X, Yu Y. Optimal adaptive cancer therapy based on evolutionary game theory. *PLOS ONE.* 2025;20(4):e0320677. doi:10.1371/journal.pone.0320677

**Section 4 — RL for dosing**

28. Padmanabhan R, Meskin N, Haddad WM. Reinforcement learning-based control of drug dosing for cancer chemotherapy treatment. *Math Biosci.* 2017;293:11–20. doi:10.1016/j.mbs.2017.08.004
29. Eastman B, Przedborski M, Kohandel M. Reinforcement learning derived chemotherapeutic schedules for robust patient-specific therapy. *Sci Rep.* 2021;11:17882. doi:10.1038/s41598-021-97028-6
30. Kiram FME, Youkana I, Saouli R, Susto GA, Kahloul L. Recurrent Deep Reinforcement Learning for Chemotherapy Control under Partial Observability. *arXiv:2605.02552.* 2026.
31. Tordesillas J, Arbelaiz J. Personalized Cancer Chemotherapy Schedule: a numerical comparison of performance and robustness in model-based and model-free scheduling methodologies. *arXiv:1904.01200.* 2019.
32. Shiranthika C, Chen KW, Wang CY, Yang CY, Sudantha BH, Li WF. Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning. *IEEE J Biomed Health Inform.* 2022;26(9):4763–4772. doi:10.1109/JBHI.2022.3183854
33. Mashayekhi H, Nazari M, Jafarinejad F, Meskin N. Deep reinforcement learning-based control of chemo-drug dose in cancer treatment. *Comput Methods Programs Biomed.* 2024;243:107884. doi:10.1016/j.cmpb.2023.107884

**Section 5 — Adjacent domains**

34. Tseng HH, Luo Y, Cui S, Chien JT, Ten Haken RK, El Naqa I. Deep reinforcement learning for automated radiation adaptation in lung cancer. *Med Phys.* 2017;44(12):6690–6705. doi:10.1002/mp.12625
35. Niraula D, Jamaluddin J, Matuszak MM, Ten Haken RK, El Naqa I. Quantum deep reinforcement learning for clinical decision support in oncology: application to adaptive radiotherapy. *Sci Rep.* 2021;11:23545. doi:10.1038/s41598-021-02910-y
36. Yao Y, Chen FY, Zhang Q. Optimized patient-specific immune checkpoint inhibitor therapy for cancer treatment based on tumor immune microenvironment modeling. *Brief Bioinform.* 2024. doi:10.1093/bib/bbae547
37. Lagzian M, Razavi SE, Kardehi Moghaddam R. A biologically constrained agent-based model of cancer stem cell dynamics with reinforcement learning-guided adaptive radiotherapy. *PLOS ONE.* 2026;21(2):e0340426. doi:10.1371/journal.pone.0340426
38. Zhu W, Zhu M, Wang W, Xu L, Wu J, Li W, Ma R, Cao L. Deep learning for adaptive chemotherapy: a DDPG-based approach to optimizing tumor-immune dynamics. *PLOS ONE.* 2026;21(4):e0345877. doi:10.1371/journal.pone.0345877

---

*38 unique papers read in full text; 39 PDFs on disk (12a/12b identical); 2 files mismatched against the manifest and 2 papers missing, as detailed in §0.1.*
