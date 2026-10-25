"""
fx - the Phase 3 follow-up experiments of `further expts.txt`: the extra patients, every PPO job, and the
switching-penalty value iteration (D4). The model itself lives in ../src/atrl.py (new options default to the
original model, see decisions.md F0).

    python further_expts/fx.py [workers]   # trains every missing job (cached in results/agents/), default 6 workers

then execute further_expts_experiments.ipynb, which only loads the cached checkpoints.
Kept light on imports on purpose: spawned workers re-import this file.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
from atrl import *  # noqa: E402,F403  (also brings np, sparse, spsolve)

TURNP = patient(0.0, 0.04)        # A1: PoP with turnover 0.04 (E7 cohort edge, below Strobl's bound)
MU = 3e-5                         # E1: smallest swept mu where VI-R4's dose departs from VI-R1's by >= 0.5 pp
CONVP = patient(mu=MU)            # E1: PoP with sensitive -> resistant conversion
DOSES3 = (0.0, 0.5, 1.0)          # D1
DOSES5 = (0.0, 0.25, 0.5, 0.75, 1.0)  # D2
B2_TARGETS = (0.05, 0.1, 0.25, 0.5)   # B2 smooth-target f*, the E6 grid


def fjobs(exp):
    """Training jobs per experiment, one definition shared by the notebook and the launcher below."""
    a = os.path.join(HERE, 'results', 'agents')
    ten = lambda tag, **kw: [dict(obj=o, seed=s, out=f'{a}/{tag}_{o}_s{s}.pkl', **kw) for o in OBJECTIVES for s in range(10)]
    if exp == 'B2':   # smooth coexistence rewards: entropy (10 seeds, replaces R3) and target distance (5 seeds per f*, as E6)
        return ([dict(obj='R3s', sampler=Fixed(POP), seed=s, out=f'{a}/B2_R3s_s{s}.pkl') for s in range(10)] +
                [dict(obj=f'R3d{f}', sampler=Fixed(POP), seed=s, out=f'{a}/B2_R3d{f}_s{s}.pkl') for f in B2_TARGETS for s in range(5)])
    if exp == 'A1':   # E3 on a patient with turnover > 0
        return ten('A1', sampler=Fixed(TURNP))
    if exp == 'A2':   # E3 (full state) and E11b (burden actor + full critic) at the E11d budget
        return (ten('A2full', sampler=Fixed(POP), steps=3_000_000) +
                ten('A2burden', sampler=Fixed(POP), obs='burden', critic_obs='full', steps=3_000_000))
    if exp == 'D1':   # three doses
        return ten('D1', sampler=Fixed(POP), actions=DOSES3)
    if exp == 'D2':   # five doses
        return ten('D2', sampler=Fixed(POP), actions=DOSES5)
    if exp == 'E1':   # sensitive -> resistant conversion
        return ten('E1', sampler=Fixed(CONVP))
    raise ValueError(exp)


EXPS = ('B2', 'D1', 'A1', 'E1', 'D2', 'A2')  # training order: cheap and central first, the 3x-budget reruns last


# --------------------------------------------------------------------------
# D4: value iteration with a dose-switching penalty. The previous dose joins the state, so the
# (N, log f) grid is replicated once per previous action (still cheap).
# --------------------------------------------------------------------------
def solve_vi_switch(obj, p, lam, actions=(0.0, 1.0), max_iter=200):
    """Policy iteration on (previous dose, N, log f) with reward r - lam * 1[D_t != D_{t-1}].
    Block j of V holds the states whose previous dose was actions[j]. lam = 0 reproduces solve_vi."""
    interval = CFG['interval']
    gamma = gamma_for(interval)
    g = Grid(p['N0'])
    S, R = g.states()
    n, k = S.size, len(actions)
    r, P = [], []
    for i, a in enumerate(actions):
        S1, R1, prog, _ = advance(S, R, a, p, interval)
        base = reward(obj, S1, R1, np.full(n, a), prog, np.zeros(n, bool), p, gamma)
        r.append(np.concatenate([base - lam * (i != j) for j in range(k)]))
        idx, w = g.weights(S1, R1)
        W = sparse.csr_matrix(((w * (~prog)[:, None]).ravel(), (np.repeat(np.arange(n), 4), idx.ravel())), shape=(n, n))
        M = np.zeros((k, k)); M[:, i] = 1  # from any previous dose, taking action i lands in block i
        P.append(sparse.kron(sparse.csr_matrix(M), W, format='csr'))
    m = k * n
    pol, I = np.full(m, k - 1), sparse.identity(m, format='csr')
    for it in range(max_iter):
        Ppi = sparse.diags((pol == 0).astype(float)) @ P[0]
        for j in range(1, k):
            Ppi = Ppi + sparse.diags((pol == j).astype(float)) @ P[j]
        V = spsolve((I - gamma * Ppi).tocsc(), np.choose(pol, r))
        Q = np.stack([r[j] + gamma * (P[j] @ V) for j in range(k)], -1)
        new = np.where(Q.max(1) > Q[np.arange(m), pol] + 1e-12, Q.argmax(1), pol)
        if (new == pol).all():
            break
        pol = new
    return dict(V=V.reshape(k, n), grid=g, obj=obj, p=p, lam=lam, actions=tuple(actions), iters=it + 1)


class SwitchLookahead(Policy):
    """One-step lookahead on solve_vi_switch's value. The first decision is free (no previous dose)."""
    def __init__(self, vi):
        self.vi, self.gamma = vi, gamma_for(CFG['interval'])

    def reset(self, B):
        self.prev = None

    def __call__(self, S, R, t, p):
        vi, acts, q = self.vi, self.vi['actions'], []
        for i, a in enumerate(acts):
            S1, R1, prog, _ = advance(S, R, a, vi['p'], CFG['interval'])
            pen = 0.0 if self.prev is None else vi['lam'] * (self.prev != i)
            r = reward(vi['obj'], S1, R1, np.full(S.shape, a), prog, np.zeros(S.shape, bool), vi['p'], self.gamma)
            q.append(r - pen + self.gamma * ~prog * vi['grid'].interp(vi['V'][i], S1, R1))
        self.prev = np.argmax(q, 0)
        return np.asarray(acts, float)[self.prev]


if __name__ == '__main__':
    jobs_ = [j for e in EXPS for j in fjobs(e)]
    todo = [j for j in jobs_ if not os.path.exists(j['out'])]
    print(f'{len(todo)} of {len(jobs_)} jobs to train', flush=True)
    run_many(jobs_, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 6)  # ~0.5 GB RAM per worker
    print('done', flush=True)
