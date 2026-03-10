"""
Figure generation utilities for Chapter 7 (Diffusion and Score-Based Models).
Produces the chapter figures using the shared visual style.

Run:
  python ch7.py
  python ch7.py --fig 7.1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle


# =============================================================================
# Style constants (mirror Chapter 1 palette)
# =============================================================================

@dataclass(frozen=True)
class Palette:
    p: str = "#2e5877"
    q: str = "#d58a2f"
    teal: str = "#2f8b7b"
    green: str = "#6d8b4f"
    magenta: str = "#b4574a"
    gray: str = "#4a4038"
    light_gray: str = "#b7a89a"

PALETTE = Palette()
STRUCTURE_COLORS = [PALETTE.p, PALETTE.teal, PALETTE.magenta, PALETTE.green]
FIG_W = 6.4
FIG_H = 3.6
PAPER = "#fff9f3"
PANEL = "#fbf4ec"
GRID = "#dccdc2"
INK = "#2b231e"

WARM_CMAP = LinearSegmentedColormap.from_list(
    "warm_sci",
    ["#fff5e8", "#f3d3a6", "#e3a05f", "#b86a3c", "#6b2f2a"],
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10.4,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.6,
            "figure.facecolor": PAPER,
            "axes.facecolor": PANEL,
            "axes.edgecolor": PALETTE.gray,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": PALETTE.gray,
            "ytick.color": PALETTE.gray,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "savefig.bbox": "tight",
            # Slightly larger pad prevents tight-bbox from cropping callouts at the edges.
            "savefig.pad_inches": 0.04,
            "savefig.facecolor": PAPER,
        }
    )


def output_dir() -> Path:
    base = Path(__file__).resolve().parent
    out = base / "figures" / "ch07_diffusion_score"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = output_dir()
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    print(f"✓ Saved {stem} (pdf+png)")


def _glow(line, color: str, lw: float) -> None:
    line.set_path_effects(
        [
            patheffects.Stroke(linewidth=lw * 2.3, foreground=color, alpha=0.18),
            patheffects.Normal(),
        ]
    )


def _setup_axes(*axes: plt.Axes) -> None:
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.spines["top"].set_color(PALETTE.gray)
        ax.spines["right"].set_color(PALETTE.gray)
        ax.spines["bottom"].set_color(PALETTE.gray)
        ax.spines["left"].set_color(PALETTE.gray)

def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> float | np.ndarray:
    trap = getattr(np, "trapezoid", None)
    if trap is not None:
        return trap(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)  # type: ignore[attr-defined]


# =============================================================================
# Figure implementations
# =============================================================================

def figure_7_1_forward_reverse() -> None:
    fig = plt.figure(figsize=(FIG_W * 1.45, FIG_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax_forward = fig.add_subplot(gs[0, 0])
    ax_reverse = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor(PAPER)
    _setup_axes(ax_forward, ax_reverse)

    # Panel A: forward corruption of a single image stage-by-stage
    size = 64
    base_x, base_y = np.linspace(-1, 1, size), np.linspace(-1, 1, size)
    Xc, Yc = np.meshgrid(base_x, base_y)
    base_image = np.exp(-3 * (Xc ** 2 + 0.4 * Yc ** 2)) * 0.9 + 0.1 * np.exp(-5 * ((Xc + 0.3) ** 2 + (Yc - 0.4) ** 2))
    n_steps = 6
    beta_schedule = np.linspace(0.01, 0.4, n_steps)
    alpha_products = np.cumprod(1.0 - beta_schedule)
    rng = np.random.RandomState(0)
    gap = 4
    image_row = np.ones((size, n_steps * size + (n_steps - 1) * gap)) * 0.5
    for idx in range(n_steps):
        alpha = alpha_products[idx]
        noise = rng.normal(0, 1, (size, size))
        corrupted = np.sqrt(alpha) * base_image + np.sqrt(1 - alpha) * noise
        corrupted = (corrupted - corrupted.min()) / (corrupted.max() - corrupted.min())
        start = idx * (size + gap)
        image_row[:, start : start + size] = corrupted
    # Leave a small blank band at the bottom for annotations/equation (avoids clipping).
    image_y0 = 0.18
    ax_forward.imshow(image_row, cmap="magma", origin="lower", extent=(-0.05, 1.05, image_y0, 1))
    stage_positions = np.linspace(0, 1, n_steps)
    for pos in stage_positions:
        ax_forward.vlines(pos, image_y0, 1, color="#ffffff", alpha=0.4, linewidth=0.6)
    ax_forward.set_xlim(-0.05, 1.05)
    ax_forward.set_ylim(0, 1)
    ax_forward.set_xticks([0.0, 0.5, 1.0])
    ax_forward.set_xticklabels(["clean", "", "noisy"])
    ax_forward.set_yticks([])
    ax_forward.set_xlabel("timestep $t$")
    ax_forward.set_title("Forward corruption timeline", pad=16)
    ax_forward.text(
        0.5,
        1.02,
        r"Noise $\beta_t$ gradually swallows structure",
        transform=ax_forward.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=PALETTE.magenta,
        clip_on=False,
    )
    ax_forward.text(
        0.5,
        0.07,
        r"$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$",
        ha="center",
        va="center",
        fontsize=9,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec=GRID, alpha=0.92, linewidth=1.0),
    )
    ax_forward.arrow(
        0.08,
        0.03,
        0.75,
        0,
        head_width=0.03,
        head_length=0.04,
        fc=PALETTE.gray,
        ec=PALETTE.gray,
        linewidth=1.0,
        length_includes_head=True,
    )
    ax_forward.text(
        0.5,
        0.30,
        "Envelopes of increasing noise",
        ha="center",
        fontsize=8.5,
        color=PALETTE.gray,
        bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec=GRID, alpha=0.80, linewidth=0.9),
    )

    # Panel B: reverse denoising as before

    # Panel B: reverse denoising
    theta = np.linspace(0, 2 * math.pi, 450)
    radius = 1.0 + 0.3 * np.sin(3 * theta)
    x_manifold = radius * np.cos(theta)
    y_manifold = 0.8 * np.sin(theta)
    manifold = np.stack([x_manifold, y_manifold], axis=1)
    ax_reverse.fill(x_manifold, y_manifold, color=PALETTE.q, alpha=0.08, zorder=1)
    ax_reverse.plot(x_manifold, y_manifold, color=PALETTE.q, linewidth=2.6, zorder=2)

    grid_x = np.linspace(-2.4, 2.4, 30)
    grid_y = np.linspace(-1.9, 1.9, 25)
    X, Y = np.meshgrid(grid_x, grid_y)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    diffs = manifold[None, :, :] - points[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)
    nearest = np.argmin(dists, axis=1)
    delta = manifold[nearest] - points
    norms = np.linalg.norm(delta, axis=1)
    directions = np.zeros_like(delta)
    mask = norms > 1e-6
    directions[mask] = delta[mask] / norms[mask, None]
    magnitude = np.clip(0.9 - norms * 0.2, 0.05, 0.9)
    norm = mcolors.Normalize(vmin=0.0, vmax=0.9)
    ax_reverse.quiver(
        points[:, 0],
        points[:, 1],
        directions[:, 0],
        directions[:, 1],
        magnitude,
        cmap=LinearSegmentedColormap.from_list("score_field", [PALETTE.magenta, PALETTE.teal]),
        norm=norm,
        scale=28,
        width=0.003,
        headwidth=4,
        alpha=0.66,
        zorder=3,
    )

    reverse_path = []
    current = np.array([2.1, -1.2])
    reverse_path.append(current.copy())
    for idx in range(10):
        diffs = manifold - current
        dist = np.linalg.norm(diffs, axis=1)
        best = int(np.argmin(dist))
        direction = diffs[best]
        length = np.linalg.norm(direction)
        if length < 1e-6:
            break
        step = 0.55 * max(0.3, 1.0 - idx * 0.05)
        current = current + (direction / length) * step
        reverse_path.append(current.copy())
    reverse_path = np.array(reverse_path)
    ax_reverse.plot(
        reverse_path[:, 0],
        reverse_path[:, 1],
        color=PALETTE.teal,
        linewidth=2.6,
        solid_capstyle="round",
        zorder=5,
    )
    ax_reverse.scatter(reverse_path[0, 0], reverse_path[0, 1], color=PALETTE.magenta, s=60, zorder=6, label="Noisy start")
    ax_reverse.scatter(reverse_path[-1, 0], reverse_path[-1, 1], color=PALETTE.p, s=90, zorder=6, label="Reconstructed point")
    if reverse_path.shape[0] >= 2:
        arrow = FancyArrowPatch(
            reverse_path[-2],
            reverse_path[-1],
            arrowstyle="->",
            color=PALETTE.teal,
            linewidth=1.6,
            mutation_scale=16,
        )
        ax_reverse.add_patch(arrow)
    ax_reverse.set_xlim(-2.5, 2.5)
    ax_reverse.set_ylim(-2.1, 2.0)
    ax_reverse.set_xticks([])
    ax_reverse.set_yticks([])
    ax_reverse.set_aspect("equal")
    ax_reverse.set_title("Reverse denoising via the score field", pad=6)
    ax_reverse.text(
        0.04,
        0.86,
        "Score\n" + r"$s_t(x)=\nabla_x\log q_t(x)$",
        transform=ax_reverse.transAxes,
        fontsize=9.3,
        weight="bold",
        color=INK,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec=GRID, alpha=0.88, linewidth=0.9),
    )
    ax_reverse.text(
        0.04,
        0.70,
        "points uphill toward\nhigher density",
        transform=ax_reverse.transAxes,
        fontsize=8.6,
        color=PALETTE.gray,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.18", fc=PANEL, ec=GRID, alpha=0.78, linewidth=0.8),
    )

    ax_reverse.legend(
        framealpha=0.92,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        fontsize=8.6,
        borderaxespad=0.25,
        handlelength=1.1,
        handletextpad=0.45,
        labelspacing=0.35,
        markerscale=0.9,
    )

    fig.suptitle(
        "Forward noise addition vs. score-guided reverse reconstruction",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(pad=0.6, rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig_7_1_forward_reverse")

def figure_7_2_score_field() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=120)
    fig.patch.set_facecolor(PAPER)
    x = np.linspace(-3, 3, 45)
    y = np.linspace(-3, 3, 45)
    X, Y = np.meshgrid(x, y)
    mixture = np.exp(-0.5 * ((X + 1) ** 2 + (Y + 1) ** 2)) + 0.7 * np.exp(-0.5 * ((X - 1.2) ** 2 + (Y - 0.5) ** 2))
    mixture = mixture / _trapz(_trapz(mixture, x, axis=1), y, axis=0)
    levels = np.linspace(mixture.min(), mixture.max(), 8)[1:]
    ax.contour(
        X,
        Y,
        mixture,
        levels=levels,
        colors=GRID,
        linewidths=0.8,
        alpha=0.45,
        zorder=1,
    )
    grad_y, grad_x = np.gradient(np.log(mixture + 1e-12))
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    norm = Normalize(vmin=0, vmax=magnitude.max())
    score_cmap = LinearSegmentedColormap.from_list(
        "score_field_warm",
        ["#f3d3a6", "#e3a05f", "#b86a3c", "#6b2f2a"],
    )
    sub_slice = (slice(None, None, 2), slice(None, None, 2))
    X_sparse = X[sub_slice]
    Y_sparse = Y[sub_slice]
    grad_x_sparse = grad_x[sub_slice]
    grad_y_sparse = grad_y[sub_slice]
    magnitude_sparse = magnitude[sub_slice]
    ax.quiver(
        X_sparse,
        Y_sparse,
        grad_x_sparse,
        grad_y_sparse,
        magnitude_sparse,
        cmap=score_cmap,
        scale=30,
        width=0.0042,
        headwidth=4,
        norm=norm,
        alpha=0.60,
        zorder=2,
    )
    start = np.array([-2.4, -2.3])
    path = [start]
    current = start.copy()
    for _ in range(10):
        ix = np.argmin(abs(x - current[0]))
        iy = np.argmin(abs(y - current[1]))
        direction = np.array([grad_x[iy, ix], grad_y[iy, ix]])
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        current = current + 0.25 * direction
        path.append(current.copy())
    path = np.array(path)
    final_direction = direction
    final_point = path[-1].copy()
    extended_end = final_point + 0.45 * final_direction
    trajectory = np.vstack([path, extended_end])
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=PALETTE.magenta, linewidth=2.3, zorder=4)
    ax.scatter(final_point[0], final_point[1], color=PALETTE.magenta, zorder=6)
    arrow = FancyArrowPatch(path[0], extended_end, arrowstyle="->", color=PALETTE.magenta,
                           linewidth=1.6, mutation_scale=12, alpha=0.95)
    ax.add_patch(arrow)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Score field + sample trajectory")
    _setup_axes(ax)
    plt.tight_layout(pad=0.6)
    fig.subplots_adjust(top=0.88)
    save_figure(fig, "fig_7_2_score_field")

def figure_7_3_noise_schedule() -> None:
    t = np.linspace(0, 1, 160)
    beta = 1e-5 + 0.018 * t ** 1.8
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    snr = alpha_bar / (1 - alpha_bar + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.9))
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        _setup_axes(ax)

    fig.suptitle("SNR control through the noise schedule", fontsize=13, y=0.98)
    axes[0].plot(t, beta, color=PALETTE.magenta, label="$\\beta_t$ (noise rate)")
    axes[0].plot(t, alpha_bar, color=PALETTE.teal, label="$\\bar{\\alpha}_t$ (signal)")
    axes[0].set_xlabel("timestep")
    axes[0].set_ylabel("magnitude")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Signal and noise over time", pad=8)

    axes[1].fill_between(t, 1e-1, 10, color=PALETTE.teal, alpha=0.12, label="Coarse-scale regime", zorder=0)
    axes[1].fill_between(t, 10, 1e2, color=PALETTE.magenta, alpha=0.12, label="Fine-scale regime", zorder=1)
    axes[1].plot(t, snr, color=PALETTE.p, linewidth=2.3, zorder=3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("SNR (log scale)")
    axes[1].set_title("Different scales: fine (low t) vs coarse (high t)", pad=8)
    axes[1].legend(loc="upper right")

    fig.subplots_adjust(top=0.86, wspace=0.35)
    plt.tight_layout(pad=0.6)
    save_figure(fig, "fig_7_3_noise_schedule")

def figure_7_4_tweedie() -> None:
    x = np.linspace(-5, 5, 400)
    prior = 0.6 * np.exp(-0.5 * ((x + 1) / 0.7) ** 2) + 0.4 * np.exp(-0.5 * ((x - 1.2) / 0.5) ** 2)
    prior /= _trapz(prior, x)
    x_t = 1.4
    sigma = 0.8
    posterior = prior * np.exp(-0.5 * ((x - x_t) / sigma) ** 2)
    posterior /= _trapz(posterior, x)
    posterior_mean = _trapz(x * posterior, x)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.9))
    fig.patch.set_facecolor(PAPER)
    ax.plot(x, prior, color=PALETTE.gray, label="$p_0(x)$")
    ax.plot(x, posterior, color=PALETTE.teal, label=r"$p(x_0\mid x_t)$")
    ax.axvline(x_t, color=PALETTE.magenta, linestyle="--", label="$x_t$")
    ax.axvline(posterior_mean, color=PALETTE.p, linestyle="-", label="Posterior mean")
    arrow = FancyArrowPatch((x_t, posterior.max() * 0.5), (posterior_mean, posterior.max() * 0.5),
                           arrowstyle="->", color=PALETTE.q, linewidth=2.2)
    ax.add_patch(arrow)
    ax.text(
        0.98,
        0.55,
        r"Adjustment $(1-\bar{\alpha}_t)\,s_t(x_t)$",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=INK,
        fontsize=10.0,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff0ef", ec=PALETTE.magenta, alpha=0.95, linewidth=1.2),
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("Density")
    ax.set_title(r"Tweedie: the score gives the optimal correction to $x_t$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.95), framealpha=0.95)
    _setup_axes(ax)
    plt.tight_layout(pad=0.6)
    save_figure(fig, "fig_7_4_tweedie")

def figure_7_5_guidance() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=120)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 4.5)
    # Move rows slightly down to avoid crowding the title/formula at the top.
    offsets = [3.55, 2.35, 1.15]
    labels = ["Cond. (w=1)", "Uncond. (w=0)", "Guided (w=7)"]
    colors = [PALETTE.teal, PALETTE.gray, PALETTE.magenta]
    for row_idx, (row, label, color) in enumerate(zip(offsets, labels, colors)):
        ax.text(0.25, row + 0.32, label, fontsize=10, fontweight="bold", color=color)
        for col_idx, col in enumerate(np.linspace(1.2, 6.8, 6)):
            jitter = 0.05 * (col_idx - 2.5)
            vertical_shift = 0.04 * (row_idx - 1)
            circle_center = (col + 0.02 * row_idx, row + vertical_shift + jitter * 0.02)
            circle = plt.Circle(circle_center, 0.3, color=color, alpha=0.35 + 0.05 * row_idx, linewidth=0.9, ec=PALETTE.gray)
            ax.add_patch(circle)
            segment = Line2D(
                [circle_center[0] - 0.25, circle_center[0] + 0.25],
                [circle_center[1] - 0.12, circle_center[1] + 0.12],
                color=color,
                linewidth=2.2,
            )
            ax.add_line(segment)
    ax.text(
        0.5,
        0.985,
        r"$\tilde{s}_t = s_\theta(\cdot,\emptyset) + w\,(s_\theta(\cdot,y) - s_\theta(\cdot,\emptyset))$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.34", fc="#fffaf0", ec=PALETTE.gray, alpha=0.95, linewidth=1.3),
    )
    ax.set_title("Classifier-free guidance interpolates between scores", pad=12)
    _setup_axes(ax)
    plt.tight_layout(pad=0.6, rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig_7_5_guidance")

def figure_7_6_sampling_speed() -> None:
    methods = ["DDPM\n1000", "DDIM\n50", "DPM\nSolver", "Consistency"]
    nfe = [1000, 50, 25, 5]
    fid = [3.5, 4.1, 3.9, 6.5]
    time_eval = [620, 25, 12, 3]
    time_mem = [60, 5, 1.5, 0.5]
    time_other = [20, 3, 1, 0.2]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 1.15, FIG_H), gridspec_kw={"width_ratios": [1.05, 1.15]})
    fig.patch.set_facecolor(PAPER)
    ax0, ax1 = axes
    _setup_axes(ax0, ax1)
    order = np.argsort(nfe)
    nfe_sorted = np.asarray(nfe, dtype=float)[order]
    fid_sorted = np.asarray(fid, dtype=float)[order]
    ax0.semilogx(nfe_sorted, fid_sorted, marker="o", linestyle="-", color=PALETTE.p, linewidth=2, markersize=7)
    ax0.set_xlabel("NFE")
    ax0.set_ylabel("FID (lower is better)")
    ax0.set_title("Quality vs compute")
    ax0.grid(True, which="both", alpha=0.08)
    ax0.axhline(
        4.5,
        color=PALETTE.magenta,
        linestyle="--",
        linewidth=2.8,
        alpha=0.9,
        label="Quality threshold",
        zorder=2,
    )
    ax0.legend(loc="upper right", framealpha=0.9)

    bottom = np.zeros(len(methods))
    components = [
        ("Model eval", time_eval, PALETTE.teal),
        ("Memory transfer", time_mem, PALETTE.magenta),
        ("Other overhead", time_other, PALETTE.light_gray),
    ]
    for label, values, color in components:
        ax1.bar(methods, values, bottom=bottom, label=label, color=color, edgecolor=PALETTE.gray)
        bottom += np.array(values)
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Time breakdown per sampler")
    ax1.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.92)
    ax1.tick_params(axis="x", labelsize=8.8)
    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center")
    plt.tight_layout(pad=0.6)
    save_figure(fig, "fig_7_6_sampling_speed")


def main() -> None:
    configure_matplotlib()
    parser = argparse.ArgumentParser(description="Generate Chapter 7 figures")
    parser.add_argument("--fig", help="Figure to render (e.g., 7.1)", default="all")
    args = parser.parse_args()
    figures = {
        "7.1": figure_7_1_forward_reverse,
        "7.2": figure_7_2_score_field,
        "7.3": figure_7_3_noise_schedule,
        "7.4": figure_7_4_tweedie,
        "7.5": figure_7_5_guidance,
        "7.6": figure_7_6_sampling_speed,
    }
    if args.fig == "all":
        for func in figures.values():
            func()
    elif args.fig in figures:
        figures[args.fig]()
    else:
        raise SystemExit(f"Unknown figure {args.fig}")


if __name__ == "__main__":
    main()
