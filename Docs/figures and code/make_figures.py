"""Figures for the AI in Medicine proposal + deck.
All trajectory/heatmap data are produced by simulating the model in lv_model.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from lv_model import *

# ---- Design tokens (deck identity + validated categorical palette) --------
NAVY   = "#1B2A4A"
TEAL   = "#0F6E5F"
INK    = "#2A2A2A"
MUTED  = "#898781"
GRID   = "#E1E0D9"
AXIS   = "#C3C2B7"
CARD   = "#F4F6F7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated: all checks pass
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "Carlito", "font.size": 11,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 220,
})

COST, TURN = 0.10, 0.06          # re-implementation benchmark point
OUT = "/home/claude/figs"
import os; os.makedirs(OUT, exist_ok=True)


def tidy(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


# =========================================================================
# FIG 1 - Why adaptive therapy exists: burden trajectories
# =========================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    runs = [
        ("No treatment",    p_none,       S3, (-8,  0.055), "right"),
        ("Continuous MTD",  p_mtd,        S2, (-14, 0.055), "right"),
        ("Adaptive (AT50)", make_at(0.5), S1, (10,  0.055), "left"),
    ]
    for label, pol, c, (dx, dy), ha in runs:
        r = simulate(pol, COST, TURN, tmax=900)
        ttp = r["ttp"]
        m = r["t"] <= ttp                      # stop each arm at its own progression
        ax.plot(r["t"][m], (r["N"] / r["N0"])[m], color=c, lw=2.2,
                solid_capstyle="round", zorder=3)
        ax.plot([ttp], [PROG], "o", ms=8, color=c, mec="white", mew=1.8, zorder=5)
        ax.annotate(f"{label}\n{ttp:.0f} days", xy=(ttp, PROG),
                    xytext=(ttp + dx, PROG + dy), ha=ha, va="bottom",
                    fontsize=11, color=c, weight="bold", linespacing=1.35)
    ax.axhline(PROG, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.text(875, PROG - 0.045, "progression threshold  N = 1.2 N₀",
            fontsize=10, color=MUTED, ha="right", va="top")
    ax.set_xlim(0, 880); ax.set_ylim(0.0, 1.38)
    ax.set_xlabel("Days from treatment start")
    ax.set_ylabel("Tumour burden  N(t) / N₀")
    ax.set_title("Withholding drug delays progression by preserving competition",
                 color=NAVY, weight="bold", fontsize=13.5, loc="left", pad=12)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_trajectories.png", bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# FIG 2 - The mechanism: competitive release under MTD vs containment
# =========================================================================
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    cases = [("Continuous MTD", p_mtd, axes[0]),
             ("Adaptive therapy (AT50)", make_at(0.5), axes[1])]
    for title, pol, ax in cases:
        r = simulate(pol, COST, TURN, tmax=760)
        m = r["t"] <= 740
        t = r["t"][m]
        ax.fill_between(t, 0, r["S"][m] / r["N0"], color=S1, alpha=0.85, lw=0,
                        label="Sensitive")
        ax.fill_between(t, r["S"][m] / r["N0"],
                        (r["S"][m] + r["R"][m]) / r["N0"], color=S2, alpha=0.9,
                        lw=0, label="Resistant")
        ax.plot(t, (r["S"][m] + r["R"][m]) / r["N0"], color="white", lw=2.0)
        ax.axhline(PROG, color=MUTED, lw=1.1, ls=(0, (5, 4)))
        ax.set_title(f"{title}\nprogression at {r['ttp']:.0f} d", color=NAVY,
                     weight="bold", fontsize=12, loc="left", pad=10,
                     linespacing=1.4)
        ax.set_xlabel("Days"); ax.set_xlim(0, 740); ax.set_ylim(0, 1.42)
        tidy(ax)
    axes[0].set_ylabel("Population / N₀")
    axes[0].annotate("sensitive cells eradicated\n→ competitive release",
                     xy=(170, 0.28), xytext=(250, 0.85), fontsize=10,
                     color=INK, linespacing=1.35,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                                     connectionstyle="arc3,rad=-0.25"))
    axes[1].annotate("sensitive pool repeatedly rebuilt\n→ resistance suppressed for longer",
                     xy=(255, 0.42), xytext=(330, 1.05), fontsize=10,
                     color=INK, linespacing=1.35, ha="left",
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                                     connectionstyle="arc3,rad=0.3"))
    axes[1].legend(loc="lower right", frameon=True, facecolor="white",
                   edgecolor=AXIS, framealpha=0.95, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_mechanism.png", bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# FIG 3 - Cohort heatmap: the answer depends on where the patient sits
# =========================================================================
def fig3():
    costs = np.linspace(0.02, 0.50, 40)
    turns = np.linspace(0.00, 0.09, 36)
    gain = np.full((len(turns), len(costs)), np.nan)
    for i, tu in enumerate(turns):
        for j, co in enumerate(costs):
            ct = simulate(p_mtd,        co, tu, tmax=2500, record=False)
            at = simulate(make_at(0.5), co, tu, tmax=2500, record=False)
            if ct["progressed"] and at["progressed"]:
                gain[i, j] = at["ttp"] / ct["ttp"]
    np.save(f"{OUT}/gain.npy", gain)

    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP[1:])
    cmap.set_bad("#EDEDEA")                      # censored region reads as absent
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    im = ax.imshow(gain, origin="lower", aspect="auto", cmap=cmap,
                   extent=[costs[0], costs[-1], turns[0], turns[-1]],
                   vmin=np.nanmin(gain), vmax=np.nanmax(gain), interpolation="nearest")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("TTP gain of AT50 over continuous MTD (×)", fontsize=10.5,
                 color=INK)
    cb.outline.set_edgecolor(AXIS)

    ax.text(0.375, 0.0715, "tumour never reaches the progression threshold\n"
                           "(model equilibrium sits below 1.2 N₀)",
            fontsize=10, color=MUTED, ha="center", va="center", linespacing=1.4)

    ax.plot([COST], [TURN], "o", ms=12, mfc="none", mec="white", mew=2.6, zorder=5)
    ax.plot([COST], [TURN], "o", ms=12, mfc="none", mec=S2, mew=2.0, zorder=6)
    ax.annotate("a single MVP patient\nreports one number from here",
                xy=(COST - 0.006, TURN + 0.002), xytext=(0.055, 0.0825),
                fontsize=10.5, color=S2, weight="bold", linespacing=1.35,
                ha="left", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=S2, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=S2, lw=1.5,
                                connectionstyle="arc3,rad=0.2"))

    gx, gy = np.meshgrid([0.10, 0.26, 0.42], [0.010, 0.030, 0.050])
    ax.plot(gx.ravel(), gy.ravel(), "s", ms=8, mfc="none", mec="white", mew=2.0,
            zorder=5)
    ax.annotate("3 × 3 virtual cohort", xy=(0.42, 0.010), xytext=(0.455, 0.0045),
                fontsize=10.5, color="white", weight="bold", ha="right")

    ax.set_xlabel("Cost of resistance"); ax.set_ylabel("Cell turnover")
    ax.set_xlim(costs[0], costs[-1]); ax.set_ylim(turns[0], turns[-1])
    ax.set_title("The size of the effect depends on where the patient sits",
                 color=NAVY, weight="bold", fontsize=13.5, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_cohort.png", bbox_inches="tight")
    plt.close(fig)
    return gain


# =========================================================================
# FIG 4 - Cross-evaluation matrix: structure, not fabricated numbers
# =========================================================================
def fig4():
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")
    cols = ["Scored on\nTTP  (R1 metric)", "Scored on\ndurability  (R4 metric)"]
    rows = ["Agent trained\non R1  (survival)", "Agent trained\non R4  (durability)"]
    x0, y0, w, h = 3.3, 1.1, 2.9, 1.55
    for j, c in enumerate(cols):
        ax.text(x0 + w*j + w/2, y0 + 2*h + 0.28, c, ha="center", va="bottom",
                fontsize=11, color=NAVY, weight="bold", linespacing=1.4)
    for i, r in enumerate(rows):
        ax.text(x0 - 0.28, y0 + h*(1-i) + h/2, r, ha="right", va="center",
                fontsize=11, color=NAVY, weight="bold", linespacing=1.4)
    for i in range(2):
        for j in range(2):
            diag = (i == j)
            rect = Rectangle((x0 + w*j, y0 + h*(1-i)), w - 0.09, h - 0.09,
                             facecolor=CARD if diag else "#E8F1FB",
                             edgecolor=AXIS if diag else S1,
                             lw=1.0 if diag else 1.8, zorder=2)
            ax.add_patch(rect)
            msg = ("circular\nnot evidence" if diag
                   else "the finding\nlives here")
            ax.text(x0 + w*j + (w-0.09)/2, y0 + h*(1-i) + (h-0.09)/2, msg,
                    ha="center", va="center", fontsize=11,
                    color=MUTED if diag else S1,
                    weight="normal" if diag else "bold", linespacing=1.5, zorder=3)
    ax.text(x0, 0.45, "Off-diagonal = what a clinician loses by optimising the wrong objective",
            fontsize=10.5, color=INK, ha="left")
    ax.text(0.0, 6.12, "Every agent is scored on every metric",
            fontsize=13.5, color=NAVY, weight="bold", ha="left")
    ax.text(0.0, 5.72, "Training on an objective and scoring it on its own metric proves nothing.",
            fontsize=10.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_crosseval.png", bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# FIG 5 - Why RL and not value iteration: where the state space goes
# =========================================================================
def fig5():
    fig, ax = plt.subplots(figsize=(9.4, 3.15))
    ax.set_xlim(0, 12); ax.set_ylim(0.3, 4.85); ax.axis("off")
    panels = [
        (0.15, "Course-scope environment", ["state: (S, R)", "action: treat / withhold",
                                            "fully observed, deterministic"],
         "Value iteration solves it exactly\n≈ 10⁴ grid states", CARD, AXIS, TEAL),
        (6.15, "Extended environment", ["state: (S, R) + toxicity", "action: dose 0, ½ or 1",
                                        "observed every 28 d, with noise",
                                        "patient (cost, turnover) unknown"],
         "Belief state — value iteration\nis no longer tractable", "#E8F1FB", S1, S1),
    ]
    for x, title, bullets, foot, fc, ec, accent in panels:
        ax.add_patch(Rectangle((x, 0.35), 5.55, 4.4, facecolor=fc, edgecolor=ec,
                               lw=1.4, zorder=2))
        ax.text(x + 0.38, 4.15, title, fontsize=12.5, color=NAVY, weight="bold")
        for k, b in enumerate(bullets):
            ax.text(x + 0.38, 3.55 - k*0.44, "•  " + b, fontsize=11, color=INK)
        ax.text(x + 0.38, 0.95, foot, fontsize=11, color=accent, weight="bold",
                linespacing=1.45)
    ax.add_patch(FancyArrowPatch((5.78, 2.55), (6.02, 2.55), arrowstyle="-|>",
                                 mutation_scale=22, color=NAVY, lw=2.2))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_why_rl.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); g = fig3(); fig4(); fig5()
    print("gain range: %.2f - %.2f" % (np.nanmin(g), np.nanmax(g)))
    for f in sorted(os.listdir(OUT)):
        print(" ", f, os.path.getsize(f"{OUT}/{f}")//1024, "KB")
