"""
Refined figure generators for Chapter 1 (openbook-chapter01).

Goal: produce book-scale readable figures (no cramped text, no accidental
artifacts like empty legends, and sufficient contrast on warm backgrounds).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIG_FACECOLOR = "#fff9f3"  # outside-axes background (warm paper)
AX_FACECOLOR = "#fbf4ec"  # inside-axes background

COLOR_BLUE = "#2e5d7a"
COLOR_ORANGE = "#d08b3c"
COLOR_GREEN = "#2a7f6f"
COLOR_RED = "#b04a4a"
COLOR_DARK = "#2f2e2b"
COLOR_MID = "#6d6a65"
GRID_COLOR = "#d9d0c8"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": FIG_FACECOLOR,
            "axes.facecolor": AX_FACECOLOR,
            "axes.edgecolor": "#8a827a",
            "axes.labelcolor": COLOR_DARK,
            "text.color": COLOR_DARK,
            "xtick.color": COLOR_MID,
            "ytick.color": COLOR_MID,
            "font.family": "serif",
            # Keep figures legible at book scale without inflating vertical
            # size (aspect ratio changes can disrupt LaTeX page breaks).
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_pdf: Path, out_png: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)


def figure_1_1_probability_codelength(out_dir: Path) -> None:
    _style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.0), gridspec_kw={"wspace": 0.28}
    )

    p = np.linspace(0.01, 1.0, 600)
    bits = -np.log2(p)
    nats = -np.log(p)

    ax1.plot(p, bits, color=COLOR_BLUE, linewidth=3.0, label=r"Bits ($\log_2(1/p)$)")
    ax1.plot(p, nats, color=COLOR_ORANGE, linewidth=3.0, label=r"Nats ($\ln(1/p)$)")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 7.6)
    ax1.set_xlabel(r"Probability $p$")
    ax1.set_ylabel("Optimal code length")
    ax1.set_title(r"(a) Probability vs code length")
    ax1.grid(True, color=GRID_COLOR, alpha=0.35, linewidth=0.8)
    leg = ax1.legend(loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_alpha(0.9)

    probabilities = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
    lengths = [1, 2, 3, 4]
    labels = [r"$p=\frac{1}{2}$", r"$p=\frac{1}{4}$", r"$p=\frac{1}{8}$", r"$p=\frac{1}{16}$"]
    # Prefix-free example codes with matching lengths.
    codes = ["0", "10", "110", "1110"]

    x = np.arange(len(probabilities))
    bars = ax2.bar(
        x,
        lengths,
        color=COLOR_BLUE,
        alpha=0.78,
        edgecolor="#2b2a28",
        linewidth=1.0,
        width=0.62,
    )
    for bar, code in zip(bars, codes, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            code,
            ha="center",
            va="bottom",
            fontsize=13,
            family="monospace",
            color=COLOR_DARK,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0.0, 5.6)
    ax2.set_xlabel("Probability")
    ax2.set_ylabel("Code length (bits)")
    ax2.set_title(r"(b) Concrete examples")
    ax2.grid(True, color=GRID_COLOR, alpha=0.35, linewidth=0.8, axis="y")
    # Intentionally no legend: avoid the accidental empty legend artifact.

    out_pdf = out_dir / "fig_1_1_probability_codelength.pdf"
    out_png = out_dir / "fig_1_1_probability_codelength.png"
    _save(fig, out_pdf, out_png)
    plt.close(fig)


def figure_1_2_mdl_polynomial(out_dir: Path) -> None:
    _style()

    rng = np.random.default_rng(7)
    x_data = np.linspace(0.0, 1.0, 20)

    def f_true(x: np.ndarray) -> np.ndarray:
        base = 0.40 + 0.62 * (1.0 - np.exp(-6.0 * x)) - 0.11 * x
        bump = 0.06 * np.exp(-((x - 0.5) ** 2) / 0.02)
        return base + bump

    y_true = f_true(x_data)
    y_data = y_true + rng.normal(0.0, 0.085, size=x_data.shape)
    x_smooth = np.linspace(0.0, 1.0, 600)

    def poly_fit(degree: int) -> np.ndarray:
        # High-degree fits are intentionally ill-conditioned to illustrate overfit wiggles.
        with np.errstate(all="ignore"):
            coeff = np.polyfit(x_data, y_data, degree)
        return np.polyval(coeff, x_smooth)

    y_under = poly_fit(1)
    y_good = poly_fit(3)
    y_over = poly_fit(15)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11.0, 4.35), gridspec_kw={"wspace": 0.28}
    )

    # ---- (a) Fit quality vs flexibility
    ax_left.scatter(
        x_data,
        y_data,
        s=52,
        color="#2b2a28",
        alpha=0.88,
        edgecolor="#ffffff",
        linewidth=0.8,
        label="data",
        zorder=6,
    )
    ax_left.plot(
        x_smooth, f_true(x_smooth), color="#b6ada5", linewidth=4.0, label="true", zorder=3
    )
    ax_left.plot(
        x_smooth, y_under, color=COLOR_ORANGE, linewidth=3.2, label=r"underfit ($d=1$)", zorder=4
    )
    ax_left.plot(
        x_smooth, y_good, color=COLOR_BLUE, linewidth=4.0, label=r"good fit ($d=3$)", zorder=5
    )
    ax_left.plot(
        x_smooth, y_over, color=COLOR_RED, linewidth=3.2, label=r"overfit ($d=15$)", zorder=4
    )

    ax_left.set_title(r"(a) Fit quality vs flexibility", pad=8)
    ax_left.set_xlabel(r"$x$")
    ax_left.set_ylabel(r"$y$")
    ax_left.grid(True, color=GRID_COLOR, alpha=0.30, linewidth=0.8)
    ax_left.set_xlim(-0.02, 1.02)
    ax_left.set_ylim(0.26, 1.14)
    leg = ax_left.legend(loc="upper left", frameon=True, borderaxespad=0.9)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_alpha(0.92)

    # ---- (b) MDL as a geometric tradeoff
    k = np.arange(1, 19)

    base = 18 / (1.0 + 0.25 * k) + 7.0 * np.exp(-(k - 1) / 3.0) + 0.2 * (18 - k) / 18.0
    data_codelength = base - base[-1] * (k / 18.0) ** 8

    lam = 1.6
    total = data_codelength + lam * k
    k_opt = int(k[np.argmin(total)])
    y_opt = float(data_codelength[k_opt - 1])

    ax_right.plot(
        k,
        data_codelength,
        color="#3a3733",
        linewidth=3.0,
        marker="o",
        markersize=4.5,
        alpha=0.92,
        label=r"data codelength (relative NLL)",
        zorder=4,
    )

    # Supporting line of slope -lambda through the optimum.
    k_line = np.linspace(k.min(), k.max(), 200)
    y_line = y_opt - lam * (k_line - k_opt)
    ax_right.plot(
        k_line,
        y_line,
        color=COLOR_GREEN,
        linewidth=3.0,
        alpha=0.9,
        label=r"supporting line (slope $-\lambda$)",
        zorder=3,
    )

    # Mark underfit / good fit / overfit example degrees (parameters ~ degree+1).
    examples = [
        (2, COLOR_ORANGE, r"$d=1$"),
        (4, COLOR_BLUE, r"$d=3$"),
        (16, COLOR_RED, r"$d=15$"),
    ]
    for kk, color, label in examples:
        yy = float(data_codelength[kk - 1])
        ax_right.scatter([kk], [yy], s=155, color=color, edgecolor="white", linewidth=1.6, zorder=6)
        ax_right.text(kk + 0.25, yy + 0.8, label, color=color, fontsize=14)

    ax_right.scatter(
        [k_opt],
        [y_opt],
        s=210,
        facecolor="none",
        edgecolor=COLOR_GREEN,
        linewidth=2.8,
        zorder=7,
    )
    ax_right.annotate(
        "MDL optimum",
        xy=(k_opt, y_opt),
        xytext=(k_opt + 3.2, y_opt + 2.4),
        arrowprops=dict(arrowstyle="->", color=COLOR_DARK, lw=1.6),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", alpha=0.88, edgecolor="#aaa39b"),
        fontsize=14,
        color=COLOR_GREEN,
        ha="left",
        va="center",
    )

    ax_right.set_title(r"(b) MDL as a geometric tradeoff", pad=8)
    ax_right.set_xlabel(r"model parameters $k$")
    ax_right.set_ylabel(r"data codelength (relative nats)")
    ax_right.grid(True, color=GRID_COLOR, alpha=0.30, linewidth=0.8)
    ax_right.set_xlim(k.min() - 0.5, k.max() + 0.5)
    ax_right.set_ylim(0.0, max(24.0, float(data_codelength.max()) + 1.0))
    leg = ax_right.legend(loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_alpha(0.92)

    out_pdf = out_dir / "fig_1_2_mdl_polynomial.pdf"
    out_png = out_dir / "fig_1_2_mdl_polynomial.png"
    _save(fig, out_pdf, out_png)
    plt.close(fig)


def _simplex_xy(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, math.sqrt(3) / 2.0])
    xy = p1[:, None] * v1 + p2[:, None] * v2 + p3[:, None] * v3
    return xy[:, 0], xy[:, 1]


def figure_1_3_cross_entropy_kl(out_dir: Path) -> None:
    _style()

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.35),
        gridspec_kw={"wspace": 0.3, "width_ratios": [1.1, 1.0]},
    )

    # ---- (a) 2-simplex with negative-entropy level sets
    grid = 140
    pts = []
    vals = []
    eps = 1e-9
    for i in range(grid + 1):
        for j in range(grid + 1 - i):
            p1 = i / grid
            p2 = j / grid
            p3 = 1.0 - p1 - p2
            pts.append((p1, p2, p3))
            f = p1 * math.log(p1 + eps) + p2 * math.log(p2 + eps) + p3 * math.log(p3 + eps)
            vals.append(f)

    pts = np.asarray(pts)
    vals = np.asarray(vals)
    x, y = _simplex_xy(pts[:, 0], pts[:, 1], pts[:, 2])

    levels = np.quantile(vals, [0.08, 0.18, 0.30, 0.44, 0.60, 0.75, 0.87])
    ax_left.tricontour(
        x,
        y,
        vals,
        levels=levels,
        colors=["#8f8880"],
        linestyles="--",
        linewidths=1.6,
        alpha=0.55,
    )

    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2.0], [0.0, 0.0]])
    ax_left.plot(triangle[:, 0], triangle[:, 1], color="#4a4641", linewidth=2.2)

    p = np.array([0.62, 0.28, 0.10])
    q = np.array([0.38, 0.44, 0.18])
    px, py = _simplex_xy(p[0:1], p[1:2], p[2:3])
    qx, qy = _simplex_xy(q[0:1], q[1:2], q[2:3])

    ax_left.scatter(px, py, s=150, color=COLOR_BLUE, edgecolor="white", linewidth=1.6, zorder=5)
    ax_left.scatter(qx, qy, s=150, color=COLOR_ORANGE, edgecolor="white", linewidth=1.6, zorder=5)
    ax_left.annotate(
        "",
        xy=(qx[0], qy[0]),
        xytext=(px[0], py[0]),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=3.0),
    )
    ax_left.text(px[0] - 0.04, py[0] - 0.03, r"$p$", fontsize=16, color=COLOR_BLUE)
    ax_left.text(qx[0] + 0.02, qy[0] - 0.01, r"$q$", fontsize=16, color=COLOR_ORANGE)

    ax_left.text(-0.04, -0.06, r"$p_1=1$", fontsize=14, color=COLOR_DARK)
    ax_left.text(1.01, -0.06, r"$p_2=1$", fontsize=14, color=COLOR_DARK, ha="right")
    ax_left.text(0.5, math.sqrt(3) / 2.0 + 0.03, r"$p_3=1$", fontsize=14, color=COLOR_DARK, ha="center")

    ax_left.set_title(r"(a) Distributions on the simplex", pad=8)
    ax_left.set_aspect("equal")
    ax_left.set_xlim(-0.06, 1.06)
    ax_left.set_ylim(-0.08, math.sqrt(3) / 2.0 + 0.08)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    # ---- (b) Tangent-gap view for Bernoulli KL
    t = np.linspace(0.02, 0.98, 800)
    f = t * np.log(t) + (1.0 - t) * np.log(1.0 - t)
    ax_right.plot(t, f, color="#3a3733", linewidth=3.0, label=r"$F(t)=t\log t+(1-t)\log(1-t)$")

    q_t = 0.25
    p_t = 0.82
    f_q = q_t * math.log(q_t) + (1.0 - q_t) * math.log(1.0 - q_t)
    slope = math.log(q_t) - math.log(1.0 - q_t)
    tangent = f_q + slope * (t - q_t)
    ax_right.plot(t, tangent, color=COLOR_GREEN, linewidth=3.0, label=r"Tangent at $q$")

    f_p = p_t * math.log(p_t) + (1.0 - p_t) * math.log(1.0 - p_t)
    t_p = f_q + slope * (p_t - q_t)
    ax_right.vlines(p_t, t_p, f_p, colors=COLOR_RED, linewidth=3.2, zorder=5)
    ax_right.scatter([p_t], [f_p], s=90, color=COLOR_RED, edgecolor="white", linewidth=1.2, zorder=6)
    ax_right.scatter([q_t], [f_q], s=90, color=COLOR_ORANGE, edgecolor="white", linewidth=1.2, zorder=6)

    ax_right.annotate(
        r"tangent gap $= \mathrm{KL}(p\|q)$",
        xy=(p_t, (t_p + f_p) / 2.0),
        xytext=(0.52, -0.75),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color=COLOR_DARK, lw=1.6),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", alpha=0.85, edgecolor="#aaa39b"),
        fontsize=14,
        ha="center",
    )

    ax_right.set_title(r"(b) Tangent gap at $p$", pad=8)
    ax_right.set_xlabel(r"Bernoulli parameter $t$")
    ax_right.set_ylabel(r"Value")
    ax_right.grid(True, color=GRID_COLOR, alpha=0.35, linewidth=0.8)
    leg = ax_right.legend(loc="lower left", frameon=True)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_alpha(0.9)
    ax_right.set_xlim(0.0, 1.0)

    out_pdf = out_dir / "fig_1_3_cross_entropy_kl.pdf"
    out_png = out_dir / "fig_1_3_cross_entropy_kl.png"
    _save(fig, out_pdf, out_png)
    plt.close(fig)


def figure_1_9_flow_warp(out_dir: Path) -> None:
    _style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.35), gridspec_kw={"wspace": 0.18}
    )

    def warp(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x2 = x + 0.22 * np.sin(math.pi * y) + 0.10 * np.sin(2.0 * math.pi * x) * np.cos(
            math.pi * y
        )
        y2 = y + 0.18 * np.sin(math.pi * x) - 0.06 * np.sin(2.0 * math.pi * y)
        return x2, y2

    grid_min, grid_max = -1.2, 1.2
    n_lines = 13
    xs = np.linspace(grid_min, grid_max, 420)
    ys = np.linspace(grid_min, grid_max, 420)
    grid_vals = np.linspace(-1.0, 1.0, n_lines)

    minor = dict(color=COLOR_ORANGE, alpha=0.45, linewidth=1.15)
    major = dict(color="#9b5b2e", alpha=0.80, linewidth=1.8)

    for v in grid_vals:
        lw = major if abs(v) < 1e-9 or abs(v - 1.0) < 1e-9 or abs(v + 1.0) < 1e-9 else minor
        ax1.plot(xs, np.full_like(xs, v), **lw)
        ax1.plot(np.full_like(ys, v), ys, **lw)

        xw, yw = warp(xs, np.full_like(xs, v))
        ax2.plot(xw, yw, **lw)
        xw, yw = warp(np.full_like(ys, v), ys)
        ax2.plot(xw, yw, **lw)

    for ax in (ax1, ax2):
        ax.axhline(0.0, color="#77726c", alpha=0.55, linewidth=1.2)
        ax.axvline(0.0, color="#77726c", alpha=0.55, linewidth=1.2)
        ax.set_aspect("equal")
        ax.set_xlim(grid_min, grid_max)
        ax.set_ylim(grid_min, grid_max)
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.set_title(r"(a) base space $z$", pad=8)
    ax2.set_title(r"(b) warped data space $x=f(z)$", pad=8)

    out_pdf = out_dir / "fig_1_9_flow_warp.pdf"
    out_png = out_dir / "fig_1_9_flow_warp.png"
    _save(fig, out_pdf, out_png)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine Chapter 1 figures.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: code/figures/ch01_information_compression).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Subset of figures: 1.1 1.2 1.3 1.9 (default: all in this script).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "figures" / "ch01_information_compression"

    only = {s.strip() for s in args.only}
    if not only or "1.1" in only:
        figure_1_1_probability_codelength(out_dir)
    if not only or "1.2" in only:
        figure_1_2_mdl_polynomial(out_dir)
    if not only or "1.3" in only:
        figure_1_3_cross_entropy_kl(out_dir)
    if not only or "1.9" in only:
        figure_1_9_flow_warp(out_dir)


if __name__ == "__main__":
    main()
