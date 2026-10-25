"""
Strobl (Cancer Res 2021) / Gallagher (bioRxiv 2023) two-population
Lotka-Volterra tumour model with turnover, in the form Gallagher writes it.

    dS/dt = r_S S (1 - (S+R)/K) (1 - dhat_D D(t)) - d_T S
    dR/dt = r_R R (1 - (S+R)/K)                   - d_T R
    N(t)  = S(t) + R(t)

Progression: N >= 1.2 * N_0   (field standard: Strobl, Gallagher, Ma)
"""
import numpy as np

# --- Parameters (Phase 2 verification table) ------------------------------
R_S      = 0.027    # /day, Strobl 2021
K        = 1.0      # normalised carrying capacity
DHAT_D   = 1.5      # Strobl 2021 kill factor, dhat_D := 2*d_D
N0_FRAC  = 0.75     # n_0 = N_0/K
FR0      = 0.01     # f_R = R_0/N_0
PROG     = 1.2      # progression at N >= 1.2 N_0

DT   = 0.1          # integration step (days)
TMAX = 3000.0       # cap (days)


def rates(cost, turnover):
    r_R = R_S * (1.0 - cost)
    d_T = turnover * R_S
    return r_R, d_T


def simulate(policy, cost=0.42, turnover=0.35, tmax=TMAX, dt=DT,
             decision_interval=None, record=True):
    """Integrate with RK4. `policy` maps (t, S, R, N0, state) -> D in [0,1].

    decision_interval: if set (days), the dose is only revised at those
    instants and held constant between them - the clinically realistic case.
    Progression is checked at EVERY integration step, never only at decision
    points (Phase 2 implementation trap #1).
    """
    r_R, d_T = rates(cost, turnover)
    N0 = N0_FRAC * K
    R = FR0 * N0
    S = N0 - R

    def deriv(S, R, D):
        comp = 1.0 - (S + R) / K
        dS = R_S * S * comp * (1.0 - DHAT_D * D) - d_T * S
        dR = r_R * R * comp - d_T * R
        return dS, dR

    ts, Ss, Rs, Ds = [0.0], [S], [R], []
    state = {}
    t = 0.0
    D = policy(0.0, S, R, N0, state)
    ttp = None
    next_decision = decision_interval if decision_interval else dt

    while t < tmax:
        if t + 1e-9 >= next_decision:
            D = policy(t, S, R, N0, state)
            next_decision += decision_interval if decision_interval else dt

        k1S, k1R = deriv(S, R, D)
        k2S, k2R = deriv(S + 0.5*dt*k1S, R + 0.5*dt*k1R, D)
        k3S, k3R = deriv(S + 0.5*dt*k2S, R + 0.5*dt*k2R, D)
        k4S, k4R = deriv(S + dt*k3S, R + dt*k3R, D)
        S += dt/6.0*(k1S + 2*k2S + 2*k3S + k4S)
        R += dt/6.0*(k1R + 2*k2R + 2*k3R + k4R)
        S = max(S, 0.0); R = max(R, 0.0)
        t += dt

        if record:
            ts.append(t); Ss.append(S); Rs.append(R); Ds.append(D)

        if ttp is None and (S + R) >= PROG * N0:      # continuous check
            ttp = t
            if not record:
                break

    Ds.append(D)
    return dict(t=np.array(ts), S=np.array(Ss), R=np.array(Rs),
                N=np.array(Ss) + np.array(Rs), D=np.array(Ds),
                N0=N0, ttp=ttp if ttp is not None else tmax,
                progressed=ttp is not None)


# --- Protocols -------------------------------------------------------------
def p_none(t, S, R, N0, st):
    return 0.0


def p_mtd(t, S, R, N0, st):
    return 1.0


def make_at(frac):
    """Adaptive therapy: dose until N < frac*N_0, withdraw until N >= N_0."""
    def pol(t, S, R, N0, st):
        N = S + R
        on = st.get('on', True)
        if on and N < frac * N0:
            on = False
        elif (not on) and N >= N0:
            on = True
        st['on'] = on
        return 1.0 if on else 0.0
    return pol


def make_at_graded(frac, levels=(0.0, 0.5, 1.0)):
    """Graded-dose adaptive therapy - the extended action space."""
    def pol(t, S, R, N0, st):
        N = S + R
        on = st.get('on', True)
        if on and N < frac * N0:
            on = False
        elif (not on) and N >= N0:
            on = True
        st['on'] = on
        if not on:
            return levels[0]
        return levels[-1] if N > 0.85 * N0 else levels[len(levels)//2]
    return pol


def p_intermittent(t, S, R, N0, st):
    return 1.0 if (int(t // 30) % 2 == 0) else 0.0
