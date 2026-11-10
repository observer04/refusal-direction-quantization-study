"""Proposed figures for the refusal-ablation / quantization report.

Every number plotted here is copied from measured, already-verified sources
(report Tables 4.1-4.4, research/model_audit_v1/semantic_summary.json,
research/model_audit_v1/classifier_comparison_all.csv,
reproduction/qwen35_9b/published_parameters.json, base_model_config.json).
No hypothetical data. Re-run with any python that has matplotlib:

    python3 report/figures_proposed/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.edgecolor": "#8a8984",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    }
)

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3de"
# Ordinal one-hue ramp (validated: dataviz validate_palette.js --ordinal, light)
CLEAR, PARTIAL, NOFUL = "#184f95", "#3987e5", "#86b6ef"
# Categorical slots 1/2 (validated reference palette)
S1, S2 = "#2a78d6", "#eb6834"


def box(ax, xy, w, h, text, fc="white", ec=INK, lw=1.2, fs=11, ls="-", weight="normal"):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, ls=ls)
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight, linespacing=1.3)


def arrow(ax, a, b, text=None, ls="-", color=INK, off=(0, 0), fs=10, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        mx, my = (a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=INK2,
                bbox=dict(fc="white", ec="none", pad=1.5))


# ---------------------------------------------------------------- Figure 1
def fig_question_design():
    """The frozen research question as a commutative square with measured pilot counts."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.75, 6.2)
    ax.axis("off")
    W, D = (2.0, 4.9), (7.4, 4.9)
    QW, QD = (2.0, 1.7), (7.4, 1.7)
    QWd = (4.7, 0.25)
    bw, bh = 2.6, 0.95
    box(ax, W, bw, bh, "$W$\noriginal model, FP16")
    box(ax, D, bw, bh, "$W+\\Delta$\nedited model, FP16")
    box(ax, QW, bw, bh, "$Q(W)$\noriginal model, NF4")
    box(ax, QD, bw, bh, "$Q(W+\\Delta)$\nedited, then NF4", fc="#eef4fc", ec=S1, lw=1.6)
    box(ax, QWd, 3.9, 0.8, "$Q(W)+\\Delta$\nNF4 base with FP32 edit factors", fs=10)

    arrow(ax, (W[0] + bw / 2, W[1]), (D[0] - bw / 2, D[1]), "apply edit $\\Delta$ (abliterate)", off=(0, 0.32))
    arrow(ax, (W[0], W[1] - bh / 2), (QW[0], QW[1] + bh / 2), "quantize $Q$", off=(-0.62, 0))
    arrow(ax, (D[0], D[1] - bh / 2), (QD[0], QD[1] + bh / 2), "quantize $Q$\n(deployment path)", off=(0.85, 0))
    arrow(ax, (QW[0] + bw / 2, QW[1] + 0.12), (QD[0] - bw / 2, QD[1] + 0.12),
          "does the edit's effect survive?", ls=(0, (4, 3)), color=S1, off=(0, -0.35))
    arrow(ax, (QW[0] + 0.3, QW[1] - bh / 2), (QWd[0] - 1.95, QWd[1]), "keep edit\nin floating point", rad=0.25, off=(-0.55, -0.1), fs=9)

    ax.text(5.0, 6.05,
            "Edit effect in FP16:  $E_{16}=H(W+\\Delta)-H(W)$        "
            "Edit effect after NF4:  $E_{4}=H(Q(W+\\Delta))-H(Q(W))$",
            ha="center", fontsize=10.5, color=INK)
    ax.text(5.0, 5.62, "Question: is $E_4 \\approx E_{16}$, and if not, is the loss edit-specific or generic quantization damage?",
            ha="center", fontsize=10.5, color=INK, style="italic")
    ax.text(9.95, -0.72, "Not run in this study: re-extracting the direction from $Q(W)$.",
            ha="right", va="bottom", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "fig1_question_design.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig_projection_geometry():
    """Why the omitted harmless-mean projection mattered (2-D schematic + measured outcome)."""
    fig = plt.figure(figsize=(8.2, 3.7))
    ax = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    ax.set_xlim(-0.2, 3.6)
    ax.set_ylim(-0.45, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    o = (0, 0)
    mh = (2.4, 0.4)       # harmless mean (schematic, not to scale)
    mf = (3.0, 1.9)       # harmful mean
    r = (mf[0] - mh[0], mf[1] - mh[1])
    import math

    n = math.hypot(*mh)
    u = (mh[0] / n, mh[1] / n)
    dot = r[0] * u[0] + r[1] * u[1]
    rp = (r[0] - dot * u[0], r[1] - dot * u[1])

    def vec(a, b, c, lw=1.6, ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=lw, color=c, ls=ls, shrinkA=0, shrinkB=0))

    vec(o, mh, MUTED)
    vec(o, mf, MUTED)
    ax.text(0.05, 1.95, "schematic (2-D, not to scale)", fontsize=8.5, color=INK2)
    ax.text(mh[0] + 0.05, mh[1] - 0.2, "$\\mu_{\\mathrm{harmless}}$", color=INK2, fontsize=12)
    ax.text(mf[0] + 0.06, mf[1] - 0.02, "$\\mu_{\\mathrm{harmful}}$", color=INK2, fontsize=12)
    vec(mh, mf, S2, lw=2)
    ax.text(mh[0] + r[0] / 2 + 0.12, mh[1] + r[1] / 2 - 0.05, "raw $d$", color=INK, fontsize=11)
    vec(mh, (mh[0] + rp[0], mh[1] + rp[1]), S1, lw=2.2)
    ax.text(mh[0] + rp[0] - 1.05, mh[1] + rp[1] - 0.1, "projected $r$", color=INK, fontsize=11)
    # component along harmless mean
    ax.plot([mh[0] + rp[0], mf[0]], [mh[1] + rp[1], mf[1]], ls=":", color=MUTED, lw=1.2)
    ax.text(-0.1, -0.42, "$r \\propto d - (d\\cdot u)\\,u$,  $u=\\hat\\mu_{\\mathrm{harmless}}$;  then  "
                        "$M' = M - \\lambda\\, r\\, r^{\\top} M$",
            fontsize=10, color=INK)

    # measured outcome, two one-axis panels
    labels = ["raw $d$", "projected $r$"]
    for i, (title, vals, xmax, fmt) in enumerate(
        [
            ("Marker matches on the 100-prompt test", (72, 8), 100, "{:d}"),
            ("Benign first-token KL vs. original", (0.434047, 0.078022), 0.5, "{:.3f}"),
        ]
    ):
        a = fig.add_axes([0.62, 0.62 - i * 0.45, 0.34, 0.22])
        a.barh([1, 0], vals, color=[S2, S1], height=0.55)
        a.set_yticks([1, 0], labels)
        a.set_xlim(0, xmax)
        a.set_title(title, fontsize=10, loc="left", color=INK)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for y, v in zip([1, 0], vals):
            a.text(v + xmax * 0.02, y, fmt.format(v), va="center", fontsize=10, color=INK)
    fig.text(0.62, 0.0, "Same fixed strengths and layers; only the projection differs.\n"
                        "Controlled check on 32 development prompts: 20/32 → 0/32 markers, KL 0.438 → 0.066.",
             fontsize=8.5, color=INK2)
    fig.savefig(OUT / "fig2_projection_geometry.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig_edit_map():
    """Where the edit lands in Qwen3.5-9B: layer types and per-layer ablation strength."""
    params = {
        "Attention output projection (19 layers)": (2.0, 18.99, 1.43, 9.65, S1),
        "MLP down projection (23 layers)": (1.67, 26.06, 0.84, 18.04, S2),
    }
    full_attn = {i for i in range(32) if i % 4 == 3}  # from base_model_config.json layer_types
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(8.0, 3.9), sharex=True,
                                  gridspec_kw={"height_ratios": [0.55, 3]})
    for i in range(32):
        ax0.add_patch(Rectangle((i - 0.45, 0), 0.9, 1, color=INK if i in full_attn else "#d6d5cf", lw=0))
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.spines[:].set_visible(False)
    ax0.tick_params(bottom=False)
    from matplotlib.patches import Patch
    ax0.legend(handles=[Patch(color=INK, label="full attention (8 layers)"),
                        Patch(color="#d6d5cf", label="Gated DeltaNet linear attention (24 layers)")],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    for k, (name, (mx, pos, mn, dist, c)) in enumerate(params.items()):
        xs, ys = [], []
        for i in range(32):
            d = abs(i - pos)
            ys.append(0.0 if d > dist else mx + (d / dist) * (mn - mx))
            xs.append(i + (k - 0.5) * 0.38)
        ax.bar(xs, ys, width=0.36, color=c, label=name, edgecolor="white", lw=0.5)
    ax.axvline(18.49, color=MUTED, ls="--", lw=1)
    ax.text(18.3, 2.18, "refusal direction taken at index 18.49 (slots 19/20, mix 0.49)", fontsize=9, color=INK2, va="top", ha="right")
    ax.set_ylabel("ablation strength $\\lambda$")
    ax.set_xlabel("layer index")
    ax.set_xlim(-0.8, 31.8)
    ax.set_ylim(0, 2.25)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.savefig(OUT / "fig3_edit_map.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def fig_semantic_vs_markers():
    """Same outputs, three measures: semantic fulfillment vs. two refusal detectors."""
    rows = [  # cohort, arm, clear, partial, no, marker_refusals, v3_refusals, n
        ("Matched (N=6)", "Original W", 0, 0, 6, 6, 6, 6),
        ("Matched (N=6)", "Reconstruction D", 2, 3, 1, 2, 1, 6),
        ("Matched (N=6)", "Publisher P", 2, 3, 1, 2, 1, 6),
        ("Fresh (N=8)", "Original W", 1, 1, 6, 8, 7, 8),
        ("Fresh (N=8)", "Reconstruction D", 6, 2, 0, 1, 0, 8),
        ("Fresh (N=8)", "Publisher P", 6, 2, 0, 1, 0, 8),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 3.9), sharex=True)
    for ax, cohort in zip(axes, ["Matched (N=6)", "Fresh (N=8)"]):
        sub = [r for r in rows if r[0] == cohort]
        for j, (_, arm, c, p, nf, mk, v3, n) in enumerate(sub):
            y = len(sub) - 1 - j
            left = 0
            for val, col, lab in [(c, CLEAR, "clear"), (p, PARTIAL, "partial"), (nf, NOFUL, "no")]:
                if val:
                    ax.barh(y, val, left=left, color=col, height=0.62, edgecolor="white", lw=2)
                    ax.text(left + val / 2, y, str(val), ha="center", va="center", fontsize=9,
                            color="white" if col != NOFUL else INK)
                left += val
            ax.text(8.6, y, f"{mk}", ha="center", va="center", fontsize=10, color=INK)
            ax.text(9.8, y, f"{v3}", ha="center", va="center", fontsize=10, color=INK)
        ax.set_yticks(range(len(sub)), [r[1] for r in reversed(sub)])
        ax.set_title(cohort, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, 10.4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    axes[0].text(8.6, 3.0, "marker\nrefusals", ha="center", fontsize=8.5, color=INK2)
    axes[0].text(9.8, 3.0, "public v3\nrefusals", ha="center", fontsize=8.5, color=INK2)
    axes[1].set_xticks(range(0, 9))
    axes[1].set_xlabel("safety prompts, by semantic fulfillment judgment")
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=CLEAR, label="clear fulfillment"), Patch(color=PARTIAL, label="partial"),
                        Patch(color=NOFUL, label="no fulfillment")],
               loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.42, 1.04))
    fig.savefig(OUT / "fig4_semantic_vs_markers.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig_nf4_pilot():
    arms = [("$W$", 0, 1, 0, 5), ("$W+\\Delta$", 2, 1, 1, 2), ("$Q(W)$", 0, 1, 0, 5),
            ("$Q(W+\\Delta)$", 2, 2, 1, 1), ("$Q(W)+\\Delta$", 2, 3, 1, 0)]
    capped = [1, 5, 1, 5, 5]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for j, (name, c, p, u, nf) in enumerate(arms):
        y = len(arms) - 1 - j
        left = 0
        for val, col, hatch in [(c, CLEAR, None), (p, PARTIAL, None), (u, "white", "////"), (nf, NOFUL, None)]:
            if val:
                ax.barh(y, val, left=left, color=col, height=0.62, edgecolor="white" if not hatch else MUTED,
                        lw=2 if not hatch else 0.8, hatch=hatch)
                ax.text(left + val / 2, y, str(val), ha="center", va="center", fontsize=9,
                        color="white" if col in (CLEAR, PARTIAL) else INK)
            left += val
        ax.text(6.55, y, f"{capped[j]}/6", ha="center", va="center", fontsize=10, color=INK)
    ax.text(6.55, len(arms) - 0.35, "reached\nlimit", ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(arms)), [a[0] for a in reversed(arms)])
    ax.set_xlim(0, 7.0)
    ax.set_xticks(range(0, 7))
    ax.set_xlabel("safety prompts (N=6 per arm)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=CLEAR, label="clear"), Patch(color=PARTIAL, label="partial"),
                       Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="unknown"),
                       Patch(color=NOFUL, label="no")],
              loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.45, 1.2))
    fig.savefig(OUT / "fig5_nf4_pilot.png")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6
def fig_claim_ladder():
    """Four claims that are routinely conflated, and which evidence reaches which rung."""
    rungs = [
        ("1  The edit is implemented", "all 42 edited matrices verified;\nsave and reload give 100 of 100 identical outputs", "established"),
        ("2  Refusal wording disappears", "marker matches fall from 100 to 8 of 100;\nv3 heuristic finds 0 refusals on 8 fresh prompts", "established (wording only)"),
        ("3  Requested content is supplied", "semantic review: 6 of 8 fresh prompts clearly\nfulfilled, against 1 of 8 for the original", "selected cases, 1 reviewer"),
        ("4  Content is correct / usable", "deliberately not tested", "open"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    fills = ["#cde2fb", "#cde2fb", "#eef4fc", "white"]
    for i, (claim, ev, status) in enumerate(rungs):
        y = 0.5 + i * 1.0
        box(ax, (2.1, y), 3.9, 0.78, claim, fc=fills[i], weight="bold", fs=10.5,
            ls="-" if i < 3 else (0, (4, 3)))
        ax.text(4.35, y, ev, va="center", fontsize=9.5, color=INK, linespacing=1.25)
        ax.text(12.15, y, status, va="center", ha="right", fontsize=9, color=INK2, style="italic")
    ax.add_patch(FancyArrowPatch((0.05, 0.3), (0.05, 3.85), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=MUTED))
    ax.text(0.0, 4.15, "stronger claim", fontsize=9, color=INK2)
    fig.savefig(OUT / "fig6_claim_ladder.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_question_design()
    fig_projection_geometry()
    fig_edit_map()
    fig_semantic_vs_markers()
    fig_nf4_pilot()
    fig_claim_ladder()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
