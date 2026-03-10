"""
Figure generation utilities for Chapter 5 (Transformers as Conditional Compressors).
Produces the eight figures for the chapter.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Warm scientific palette (match Chapters 1-4)
COLOR_PRIMARY = '#2e5877'
COLOR_SECONDARY = '#d58a2f'
COLOR_TERTIARY = '#2f8b7b'
COLOR_QUATERNARY = '#b4574a'
COLOR_GRAY = '#6f6258'
COLOR_LIGHT_BLUE = '#c9d8e2'
COLOR_LIGHT_ORANGE = '#f0cfa6'
PAPER = '#fcf6ee'
PANEL = '#f5eadf'
GRID = '#d5c6b8'
INK = '#2b231e'
HIGHLIGHT = '#efd2a0'

WARM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'warm_sci',
    ['#fff5e8', '#f3d3a6', '#e3a05f', '#b86a3c', '#6b2f2a'],
)
WARM_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    'warm_div',
    ['#2e5877', '#f7efe6', '#b86a3c'],
)

FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0
DPI = 300

OUTPUT_DIR = Path(__file__).resolve().parent / 'figures' / 'ch05_transformers_compressors'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    use_tex = os.environ.get('USE_TEX', '0') == '1'
    if use_tex and shutil.which('dvipng') is None:
        use_tex = False

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
            'text.usetex': use_tex,
            'mathtext.fontset': 'cm',
            'mathtext.rm': 'serif',
            'mathtext.it': 'serif:italic',
            'mathtext.bf': 'serif:bold',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.labelsize': 10.5,
            'axes.titlesize': 11,
            'xtick.labelsize': 9.5,
            'ytick.labelsize': 9.5,
            'legend.fontsize': 9.0,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.6,
            'figure.facecolor': PAPER,
            'axes.facecolor': PANEL,
            'axes.edgecolor': GRID,
            'axes.labelcolor': INK,
            'axes.titlecolor': INK,
            'xtick.color': COLOR_GRAY,
            'ytick.color': COLOR_GRAY,
            'grid.color': GRID,
            'grid.alpha': 0.55,
            'legend.framealpha': 0.92,
            'legend.facecolor': PAPER,
            'legend.edgecolor': GRID,
            'lines.solid_capstyle': 'round',
            'lines.solid_joinstyle': 'round',
            'savefig.facecolor': PAPER,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
        }
    )
    if use_tex:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'


def _glow(line, color: str, lw: float) -> None:
    line.set_path_effects(
        [
            patheffects.Stroke(linewidth=lw * 2.1, foreground=color, alpha=0.18),
            patheffects.Normal(),
        ]
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.9)
    ax.tick_params(color=GRID, labelcolor=COLOR_GRAY)


def callout_box(edge: str = COLOR_GRAY, face: str = PAPER) -> dict:
    return dict(boxstyle='round,pad=0.28', facecolor=face, edgecolor=edge, linewidth=1.1, alpha=0.9)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', dpi=DPI)
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=DPI)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = INK,
    fontsize: float = 9.5,
    weight: str = 'bold',
    text_color: str = INK,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha='center',
        va='center',
        fontsize=fontsize,
        color=text_color,
        weight=weight,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = INK,
          lw: float = 1.6, style: str = '-|>', alpha: float = 1.0, linestyle: str = '-') -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            lw=lw,
            color=color,
            alpha=alpha,
            linestyle=linestyle,
        )
    )

# ============================================================================
# FIGURE 5.1: Autoregressive Factorization
# ============================================================================

def figure_5_1() -> None:
    """Visualize autoregressive chain rule factorization."""

    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.35, FIGURE_HEIGHT * 1.35))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.15], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    tokens = ['The', 'cat', 'sat']
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY]
    x_positions = [0.2, 1.9, 3.6]
    conditionals = [
        r'$p(x_1)$',
        r'$p(x_2 \mid x_{<2})$',
        r'$p(x_3 \mid x_{<3})$',
    ]

    for i, (token, color, x) in enumerate(zip(tokens, colors, x_positions)):
        draw_box(ax1, (x, 0.2), 1.1, 0.35, token, COLOR_LIGHT_BLUE, fontsize=9)
        draw_box(ax1, (x, 0.7), 1.1, 0.28, conditionals[i], color, fontsize=9, text_color=PAPER)
        if i > 0:
            arrow(ax1, (x - 0.45, 0.37), (x, 0.37), color=COLOR_GRAY, lw=1.4)

    ax1.text(2.15, 1.02, 'Context grows left to right', ha='center', fontsize=9,
             bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT))

    ax1.set_xlim([-0.1, 5.0])
    ax1.set_ylim([0.05, 1.15])
    ax1.axis('off')
    ax1.set_title('(a) Chain rule as conditional factors', fontsize=11)

    probs = [0.15, 0.08, 0.25]
    codelengths = -np.log2(probs)
    cumulative = np.cumsum(codelengths)
    x_pos = np.arange(len(tokens))

    bars = ax2.bar(
        x_pos,
        codelengths,
        width=0.65,
        color=colors,
        edgecolor=INK,
        linewidth=1.1,
        alpha=0.85,
    )

    line = ax2.plot(
        x_pos,
        cumulative,
        'o-',
        color=COLOR_QUATERNARY,
        linewidth=2.6,
        markersize=7,
        markeredgecolor=INK,
        markeredgewidth=1.2,
        label='Cumulative codelength',
    )[0]
    _glow(line, COLOR_QUATERNARY, 2.6)

    for bar, cl in zip(bars, codelengths):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.7,
            f'{cl:.2f} bits',
            ha='center',
            fontsize=9,
            color='white',
            weight='bold',
            path_effects=[patheffects.withStroke(linewidth=2, foreground=INK, alpha=0.2)],
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tokens, fontstyle='italic')
    ax2.set_xlabel('Token', fontsize=10.5)
    ax2.set_ylabel('Codelength (bits)', fontsize=10.5)
    ax2.set_title('(b) Incremental codelength accumulation', fontsize=11)
    ax2.grid(True, alpha=0.28, axis='y')
    style_axes(ax2)

    ax2.annotate(
        f'Total: {cumulative[-1]:.2f} bits',
        xy=(x_pos[-1], cumulative[-1]),
        xytext=(x_pos[-1] + 0.75, cumulative[-1] + 0.55),
        arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.4),
        fontsize=9.5,
        ha='right',
        bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT),
    )

    ax2.set_xlim([-0.5, len(tokens) - 0.1])
    ax2.set_ylim([0, max(cumulative) + 1.2])
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    save_figure(fig, 'fig_5_1_autoregressive_factorization')
    print('✓ Figure 5.1 saved')
    plt.close(fig)

# ============================================================================
# FIGURE 5.2: Teacher Forcing
# ============================================================================

def figure_5_2() -> None:
    """Illustrate teacher forcing vs autoregressive generation."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.7, FIGURE_HEIGHT))

    tokens = ['The', 'cat', 'sat']
    n = len(tokens)

    # Panel (a): Teacher forcing
    for i in range(n):
        x = i * 1.2
        draw_box(ax1, (x, 0.0), 1.0, 0.45,
                 r'$\langle \mathrm{BOS} \rangle$' if i == 0 else f'${tokens[i-1]}$',
                 facecolor=COLOR_LIGHT_BLUE, fontsize=9)
        draw_box(ax1, (x, 0.85), 1.0, 0.45, 'Model', facecolor=PANEL, fontsize=8, weight='normal')
        draw_box(ax1, (x, 1.7), 1.0, 0.45, f'${tokens[i]}$', facecolor=COLOR_TERTIARY, fontsize=9)

        arrow(ax1, (x + 0.5, 0.45), (x + 0.5, 0.85), lw=1.6)
        arrow(ax1, (x + 0.5, 1.3), (x + 0.5, 1.7), lw=1.6)

    arrow(ax1, (0.1, 2.45), (3.1, 2.45), color=COLOR_GRAY, lw=1.6)
    ax1.text(1.6, 2.62, 'All positions computed in parallel', ha='center', fontsize=9.5,
             bbox=callout_box(edge=GRID, face=COLOR_LIGHT_BLUE))

    ax1.set_xlim([-0.2, 3.4])
    ax1.set_ylim([-0.2, 3.0])
    ax1.axis('off')
    ax1.set_title('(a) Teacher Forcing (Training)', fontsize=11)

    # Panel (b): Autoregressive generation
    y_positions = [0.0, 1.1, 2.2]
    for i, y in enumerate(y_positions):
        draw_box(ax2, (0.0, y), 0.9, 0.4,
                 r'$\langle \mathrm{BOS} \rangle$' if i == 0 else f'${tokens[i-1]}$',
                 facecolor=COLOR_LIGHT_BLUE, fontsize=8.5)
        draw_box(ax2, (1.2, y), 0.9, 0.4, 'Model', facecolor=PANEL, fontsize=8, weight='normal')
        draw_box(ax2, (2.4, y), 0.9, 0.4, f'${tokens[i]}$', facecolor=COLOR_TERTIARY, fontsize=8.5)

        arrow(ax2, (0.9, y + 0.2), (1.2, y + 0.2), lw=1.5)
        arrow(ax2, (2.1, y + 0.2), (2.4, y + 0.2), lw=1.5)

        if i < n - 1:
            ax2.add_patch(FancyArrowPatch((2.85, y + 0.35), (0.45, y + 1.05),
                                          arrowstyle='-|>', mutation_scale=11,
                                          lw=2.0, color=COLOR_QUATERNARY,
                                          linestyle='--', alpha=0.75,
                                          connectionstyle='arc3,rad=-0.2'))

        ax2.text(-0.25, y + 0.2, f't={i + 1}', ha='center', va='center', fontsize=9, weight='bold')

    ax2.text(1.6, 2.85, 'Sequential generation', ha='center', fontsize=9.5,
             bbox=callout_box(edge=COLOR_SECONDARY, face=COLOR_LIGHT_ORANGE))

    ax2.set_xlim([-0.5, 3.4])
    ax2.set_ylim([-0.2, 3.1])
    ax2.axis('off')
    ax2.set_title('(b) Autoregressive Generation (Inference)', fontsize=11)

    plt.tight_layout()
    save_figure(fig, 'fig_5_2_teacher_forcing')
    print('✓ Figure 5.2 saved')
    plt.close(fig)

# ============================================================================
# FIGURE 5.3: Attention Mechanism (Query-Key-Value)
# ============================================================================

def figure_5_3() -> None:
    """Visualize the attention mechanism."""

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.5, FIGURE_HEIGHT * 1.2))

    tokens = ['The', 'cat', 'sat', 'on', 'mat']
    n = len(tokens)
    current_pos = 3
    x_positions = np.arange(n) * 1.1

    y_tokens = 2.7
    y_query = 2.1
    y_keys = 1.5
    y_vals = 1.05

    for i, token in enumerate(tokens):
        if i < current_pos:
            color = COLOR_LIGHT_BLUE
            alpha = 0.45
            edgecolor = INK
            linestyle = '-'
            text_color = INK
        elif i == current_pos:
            color = COLOR_PRIMARY
            alpha = 0.85
            edgecolor = INK
            linestyle = '-'
            text_color = PAPER
        else:
            color = PAPER
            alpha = 0.55
            edgecolor = GRID
            linestyle = '--'
            text_color = COLOR_GRAY
        rect = FancyBboxPatch(
            (x_positions[i], y_tokens),
            0.9,
            0.35,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=1.2,
            alpha=alpha,
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        ax.text(x_positions[i] + 0.45, y_tokens + 0.18, token,
                ha='center', va='center', fontsize=10, weight='bold', color=text_color)

    query_x = x_positions[current_pos] + 0.45
    draw_box(ax, (query_x - 0.45, y_query), 0.9, 0.3, 'Query $q_t$', COLOR_PRIMARY, fontsize=9, text_color=PAPER)

    scores = [0.1, 0.5, 0.3, 0.1]
    for i in range(current_pos + 1):
        x = x_positions[i] + 0.45
        draw_box(ax, (x - 0.22, y_keys), 0.44, 0.24, f'$k_{{{i + 1}}}$', COLOR_SECONDARY, fontsize=8.5, weight='normal')
        draw_box(ax, (x - 0.22, y_vals), 0.44, 0.24, f'$v_{{{i + 1}}}$', COLOR_TERTIARY, fontsize=8.5, weight='normal')

        width = 1.2 + 7 * scores[i]
        line = ax.plot([query_x, x], [y_query, y_keys + 0.24],
                       color=COLOR_QUATERNARY, linewidth=width, alpha=0.55)[0]
        _glow(line, COLOR_QUATERNARY, width)

        ax.text(x, y_keys - 0.18, f'{scores[i]:.1f}', ha='center', fontsize=8.5,
                bbox=callout_box(edge=COLOR_QUATERNARY, face=HIGHLIGHT))

    future_pos = current_pos + 1
    if future_pos < n:
        x_future = x_positions[future_pos] + 0.45
        draw_box(ax, (x_future - 0.22, y_keys), 0.44, 0.24, f'$k_{{{future_pos + 1}}}$', PAPER,
                 edgecolor=GRID, fontsize=8.5, weight='normal')
        draw_box(ax, (x_future - 0.22, y_vals), 0.44, 0.24, f'$v_{{{future_pos + 1}}}$', PAPER,
                 edgecolor=GRID, fontsize=8.5, weight='normal')
        masked = ax.plot([query_x, x_future], [y_query, y_keys + 0.24],
                         color=GRID, linewidth=1.6, alpha=0.9, linestyle='--')[0]
        _glow(masked, GRID, 1.6)
        mx = (query_x + x_future) / 2
        my = (y_query + y_keys + 0.24) / 2
        ax.plot([mx], [my], marker='x', color=COLOR_QUATERNARY, markersize=7, mew=2.0)
        ax.text(mx, my + 0.17, 'masked', ha='center', fontsize=8.5, color=COLOR_GRAY,
                bbox=callout_box(edge=GRID, face=PAPER))

    draw_box(ax, (query_x + 0.65, 1.0), 1.5, 0.35,
             'Output = weighted sum', COLOR_TERTIARY, fontsize=9, weight='normal', text_color=PAPER)

    arrow(ax, (query_x + 0.25, y_vals + 0.12), (query_x + 0.65, 1.17), lw=1.5)

    ax.text(1.5, 0.35, r'$\alpha_i = \mathrm{softmax}(q_t^\top k_i / \sqrt{d_k})$',
            ha='center', fontsize=9.5,
            bbox=callout_box(edge=COLOR_SECONDARY, face=COLOR_LIGHT_ORANGE))

    ax.set_xlim([-0.5, x_positions[-1] + 2.6])
    ax.set_ylim([0.1, 3.2])
    ax.axis('off')
    ax.set_title('Attention: Soft Retrieval by Query-Key Similarity', fontsize=12)

    plt.tight_layout()
    save_figure(fig, 'fig_5_3_attention_mechanism')
    print('✓ Figure 5.3 saved')
    plt.close(fig)

# ============================================================================
# FIGURE 5.4: Scaled Dot Product Attention
# ============================================================================

def figure_5_4() -> None:
    """Show why scaling by sqrt(d) prevents saturation."""

    fig = plt.figure(figsize=(FIGURE_WIDTH * 2.5, FIGURE_HEIGHT))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.9, 0.06], wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    dimensions = [1, 10, 100, 500]
    weights_unscaled = []
    weights_scaled = []
    entropies_unscaled = []
    entropies_scaled = []

    for d in dimensions:
        np.random.seed(42)
        scores = np.random.randn(3) * np.sqrt(d)
        unscaled_probs = np.exp(scores) / np.sum(np.exp(scores))
        scaled_scores = scores / np.sqrt(d)
        scaled_probs = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores))
        weights_unscaled.append(unscaled_probs)
        weights_scaled.append(scaled_probs)
        entropies_unscaled.append(-np.sum(unscaled_probs * np.log2(unscaled_probs + 1e-10)))
        entropies_scaled.append(-np.sum(scaled_probs * np.log2(scaled_probs + 1e-10)))

    weights_unscaled = np.array(weights_unscaled)
    weights_scaled = np.array(weights_scaled)

    im1 = ax1.imshow(weights_unscaled, cmap=WARM_CMAP, vmin=0, vmax=1)
    ax1.set_title('(a) Without scaling: saturation', fontsize=11)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['key 1', 'key 2', 'key 3'])
    ax1.set_yticks(range(len(dimensions)))
    ax1.set_yticklabels([f'$d_k={d}$' for d in dimensions])
    ax1.set_xlabel('Key position', fontsize=10.5)
    ax1.set_ylabel('Dimension $d_k$', fontsize=10.5)

    im2 = ax2.imshow(weights_scaled, cmap=WARM_CMAP, vmin=0, vmax=1)
    ax2.set_title('(b) With scaling $1/\\sqrt{d_k}$: stable', fontsize=11)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['key 1', 'key 2', 'key 3'])
    ax2.set_yticks(range(len(dimensions)))
    ax2.set_yticklabels([f'$d_k={d}$' for d in dimensions])
    ax2.set_xlabel('Key position', fontsize=10.5)

    entropies_unscaled = np.array(entropies_unscaled)
    entropies_scaled = np.array(entropies_scaled)
    ax3.plot(dimensions, entropies_unscaled, label='Without scaling', color=COLOR_PRIMARY, linewidth=2.4)
    ax3.plot(dimensions, entropies_scaled, label='With $1/\\sqrt{d_k}$', color=COLOR_SECONDARY, linewidth=2.4)
    ax3.set_xscale('log')
    ax3.set_xticks(dimensions)
    ax3.set_xticklabels([str(d) for d in dimensions])
    ax3.set_xlabel('Dimension $d_k$', fontsize=10.5)
    ax3.set_ylabel('Entropy (bits)', fontsize=10.5)
    ax3.set_title('(c) Entropy vs. dimension', fontsize=11)
    ax3.grid(True, alpha=0.25)
    ax3.set_ylim([0.0, float(entropies_scaled.max() + 0.25)])
    ax3.legend(frameon=False, loc='lower left')
    ax3.annotate('Entropy collapses', xy=(500, entropies_unscaled[-1]), xytext=(60, entropies_unscaled[-1] + 0.4),
                 arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.1),
                 fontsize=9, color=INK)
    ax3.annotate('Stable entropy', xy=(500, entropies_scaled[-1]), xytext=(120, entropies_scaled[-1] + 0.08),
                 arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.1),
                 fontsize=9, color=COLOR_SECONDARY)

    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label('Attention weight', fontsize=9.5)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    ax1.text(0.1, -0.35, 'Color = weight mass', transform=ax1.transAxes,
             fontsize=9, color=COLOR_GRAY)

    plt.tight_layout()
    save_figure(fig, 'fig_5_4_scaled_attention')
    print('✓ Figure 5.4 saved')
    plt.close(fig)

# ============================================================================
# FIGURE 5.5: Softmax Temperature
# ============================================================================

def figure_5_5() -> None:
    """Show effect of temperature on softmax distribution."""

    scores = np.array([2, 1, 0, -0.5])
    labels = ['Token A', 'Token B', 'Token C', 'Token D']
    temperatures = np.array([0.5, 1.0, 2.0, 5.0])

    probs = []
    entropies = []
    for tau in temperatures:
        scaled = scores / tau
        p = np.exp(scaled) / np.sum(np.exp(scaled))
        probs.append(p)
        entropies.append(-np.sum(p * np.log2(p + 1e-10)))

    probs = np.array(probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    im = ax1.imshow(probs, cmap=WARM_CMAP, vmin=0, vmax=1, aspect='auto')
    ax1.set_yticks(range(len(temperatures)))
    ax1.set_yticklabels([f'$\\tau={t}$' for t in temperatures])
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=10)
    ax1.set_title('(a) Probability mass shifts with temperature', fontsize=11)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Probability', rotation=270, labelpad=12, fontsize=9.5)

    temp_grid = np.linspace(0.4, 5.0, 50)
    entropy_curve = []
    for tau in temp_grid:
        scaled = scores / tau
        p = np.exp(scaled) / np.sum(np.exp(scaled))
        entropy_curve.append(-np.sum(p * np.log2(p + 1e-10)))

    line = ax2.plot(temp_grid, entropy_curve, color=COLOR_PRIMARY, linewidth=2.6)[0]
    _glow(line, COLOR_PRIMARY, 2.6)
    ax2.scatter(temperatures, entropies, color=COLOR_SECONDARY, edgecolor=INK, s=40, zorder=5)

    ax2.set_xlabel('Temperature $\\tau$', fontsize=10.5)
    ax2.set_ylabel('Entropy (bits)', fontsize=10.5)
    ax2.set_title('(b) Higher temperature increases entropy', fontsize=11)
    ax2.grid(True, alpha=0.25)
    style_axes(ax2)

    ax2.annotate('Sharper', xy=(0.6, entropies[0]), xytext=(1.2, entropies[0] + 0.3),
                 arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.2),
                 fontsize=9, bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT))
    ax2.annotate('Smoother', xy=(4.5, entropies[-1]), xytext=(3.2, entropies[-1] - 0.25),
                 arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.2),
                 fontsize=9, bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT))

    plt.tight_layout()
    save_figure(fig, 'fig_5_5_temperature')
    print('✓ Figure 5.5 saved')
    plt.close(fig)

# ============================================================================
# FIGURE 5.6: Positional Encodings
# ============================================================================

def figure_5_6() -> None:
    """Visualize sinusoidal positional encodings."""

    max_len = 50
    d_model = 64

    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.7, FIGURE_HEIGHT))

    im = ax1.imshow(pe.T, cmap=WARM_DIVERGING, aspect='auto', vmin=-1, vmax=1)
    ax1.set_xlabel('Position in sequence', fontsize=10.5)
    ax1.set_ylabel('Embedding dimension', fontsize=10.5)
    ax1.set_title('(a) Sinusoidal positional encodings', fontsize=11)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Value', rotation=270, labelpad=12, fontsize=9.5)

    dims_to_plot = [0, 4, 16, 32]
    line_colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_QUATERNARY]

    for dim, color in zip(dims_to_plot, line_colors):
        line = ax2.plot(position, pe[:, dim], color=color, linewidth=2.2, alpha=0.85, label=f'dim {dim}')[0]
        _glow(line, color, 2.2)

    ax2.set_xlabel('Position in sequence', fontsize=10.5)
    ax2.set_ylabel('Encoding value', fontsize=10.5)
    ax2.set_title('(b) Different dimensions = different frequencies', fontsize=11)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim([0, max_len - 1])
    ax2.set_ylim([-1.2, 1.2])
    ax2.legend(loc='upper left', frameon=True)
    style_axes(ax2)

    ax2.annotate(
        'High freq (dim 0)',
        xy=(9, pe[9, 0]),
        xytext=(12, -0.95),
        arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=1.4),
        fontsize=9,
        bbox=callout_box(edge=COLOR_PRIMARY, face=COLOR_LIGHT_BLUE),
    )
    ax2.annotate(
        'Low freq (dim 32)',
        xy=(26, pe[26, 32]),
        xytext=(30, 0.92),
        arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=1.4),
        fontsize=9,
        bbox=callout_box(edge=COLOR_QUATERNARY, face=COLOR_LIGHT_ORANGE),
    )

    plt.tight_layout()
    save_figure(fig, 'fig_5_6_positional_encodings')
    print('✓ Figure 5.6 saved')
    plt.close(fig)

# ============================================================================
# FIGURE 5.7: Perplexity and Bits Per Token
# ============================================================================

def figure_5_7() -> None:
    """Visualize relationship between perplexity, bits, and compression."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    perplexities = np.array([2, 5, 10, 50, 100, 500, 1000])
    bits = np.log2(perplexities)

    line = ax1.plot(perplexities, bits, 'o-', color=COLOR_PRIMARY,
                    linewidth=2.6, markersize=7, markeredgecolor=INK,
                    markeredgewidth=1.1)[0]
    _glow(line, COLOR_PRIMARY, 2.6)

    annotations = [
        (2, bits[0], 'Binary choice', (3, bits[0] + 0.6)),
        (10, bits[2], 'Focused', (16, bits[2] + 0.8)),
        (100, bits[4], 'Uncertain', (150, bits[4] + 0.6)),
        (1000, bits[6], 'Near uniform', (650, bits[6] - 0.9)),
    ]

    for ppx, b, label, text_pos in annotations:
        ax1.annotate(
            label,
            xy=(ppx, b),
            xytext=text_pos,
            arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.3),
            fontsize=9,
            bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT),
        )

    ax1.set_xlabel('Perplexity', fontsize=10.5)
    ax1.set_ylabel('Bits per token', fontsize=10.5)
    ax1.set_title('(a) $\\mathrm{bpt} = \\log_2(\\mathrm{PPL})$', fontsize=11)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.25, which='major')
    style_axes(ax1)

    vocab_size = 50000
    baseline_bits = np.log2(vocab_size)

    model_perplexities = np.array([10, 50, 100, 500])
    model_bits = np.log2(model_perplexities)
    compression_ratio = baseline_bits / model_bits

    bars = ax2.bar(
        range(len(model_perplexities)),
        compression_ratio,
        color=COLOR_TERTIARY,
        alpha=0.85,
        edgecolor=INK,
        linewidth=1.1,
    )

    for i, (ppx, ratio) in enumerate(zip(model_perplexities, compression_ratio)):
        ax2.text(i, ratio + 0.1, f'{ratio:.1f}x', ha='center', fontsize=9.5, weight='bold')
        ax2.text(i, -0.25, f'PPL={ppx}', ha='center', fontsize=9)

    ax2.axhline(1, color=COLOR_GRAY, linestyle='--', linewidth=1.8, alpha=0.7)
    ax2.text(2.7, 1.08, 'Baseline (uniform)', fontsize=9,
             bbox=callout_box(edge=COLOR_GRAY, face=PAPER))

    ax2.set_ylabel('Compression ratio', fontsize=10.5)
    ax2.set_title(f'(b) Compression vs baseline ({baseline_bits:.1f} bits)', fontsize=11)
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.25, axis='y')
    ax2.set_ylim([0, max(compression_ratio) + 0.6])
    style_axes(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig_5_7_perplexity_compression')
    print('✓ Figure 5.7 saved')
    plt.close(fig)

# ============================================================================
# FIGURE 5.8: Sampling Schemes Comparison
# ============================================================================

def figure_5_8() -> None:
    """Compare different sampling schemes."""

    np.random.seed(42)
    vocab_size = 20
    logits = np.random.randn(vocab_size) * 2
    logits[0] = 5
    logits[1] = 3
    logits[2] = 2

    probs = np.exp(logits) / np.sum(np.exp(logits))
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    max_tokens = 12
    ranks = np.arange(1, max_tokens + 1)
    base = sorted_probs[:max_tokens]

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH * 1.7, FIGURE_HEIGHT * 1.5), sharex=True, sharey=True)
    axes = axes.flatten()

    def base_bars(ax: plt.Axes) -> None:
        ax.bar(ranks, base, color=COLOR_GRAY, alpha=0.25, edgecolor=GRID, linewidth=0.6)
        ax.grid(True, alpha=0.2, axis='y')
        style_axes(ax)
        ax.set_xlim([0.5, max_tokens + 0.5])
        ax.set_ylim([0, 1])

    # (a) Greedy
    base_bars(axes[0])
    greedy = np.zeros_like(base)
    greedy[0] = 1.0
    axes[0].bar(ranks, greedy, color=COLOR_QUATERNARY, alpha=0.8, edgecolor=INK, linewidth=1.1)
    axes[0].set_title('(a) Greedy: picks the max', fontsize=10)
    axes[0].text(0.75, 0.85, 'only top-1', fontsize=9,
                 bbox=callout_box(edge=COLOR_QUATERNARY, face=HIGHLIGHT))

    # (b) Temperature
    tau = 0.7
    temp_probs = base ** (1 / tau)
    temp_probs = temp_probs / temp_probs.sum()
    base_bars(axes[1])
    axes[1].plot(ranks, temp_probs, color=COLOR_PRIMARY, linewidth=2.4, marker='o')
    _glow(axes[1].lines[-1], COLOR_PRIMARY, 2.4)
    axes[1].fill_between(ranks, temp_probs, color=COLOR_PRIMARY, alpha=0.12)
    axes[1].set_title('(b) Temperature: sharper', fontsize=10)
    axes[1].text(6.4, 0.78, f'$\\tau={tau}$', fontsize=9,
                 bbox=callout_box(edge=COLOR_PRIMARY, face=COLOR_LIGHT_BLUE))

    # (c) Top-k
    k = 5
    base_bars(axes[2])
    topk = np.zeros_like(base)
    topk[:k] = base[:k]
    topk = topk / topk.sum()
    axes[2].bar(ranks[:k], topk[:k], color=COLOR_SECONDARY, alpha=0.8, edgecolor=INK, linewidth=1.1)
    axes[2].axvline(k + 0.5, color=COLOR_SECONDARY, linestyle='--', linewidth=1.5, alpha=0.7)
    axes[2].set_title(f'(c) Top-k: cutoff k={k}', fontsize=10)
    axes[2].text(7.2, 0.75, 'hard cutoff', fontsize=9,
                 bbox=callout_box(edge=COLOR_SECONDARY, face=COLOR_LIGHT_ORANGE))

    # (d) Nucleus
    p = 0.9
    cumsum = np.cumsum(base)
    nucleus_size = int(np.searchsorted(cumsum, p) + 1)
    nucleus = np.zeros_like(base)
    nucleus[:nucleus_size] = base[:nucleus_size]
    nucleus = nucleus / nucleus.sum()
    base_bars(axes[3])
    axes[3].bar(ranks[:nucleus_size], nucleus[:nucleus_size],
                color=COLOR_TERTIARY, alpha=0.8, edgecolor=INK, linewidth=1.1)
    axes[3].axvline(nucleus_size + 0.5, color=COLOR_TERTIARY, linestyle='--', linewidth=1.5, alpha=0.7)
    axes[3].set_title(f'(d) Nucleus: p={p}', fontsize=10)
    axes[3].text(6.7, 0.75, f'cut at {nucleus_size} tokens', fontsize=9,
                 bbox=callout_box(edge=COLOR_TERTIARY, face=HIGHLIGHT))

    for ax in axes:
        ax.set_xticks([1, 4, 7, 10, 12])
        ax.set_xticklabels(['1', '4', '7', '10', '12'])

    axes[2].set_xlabel('Ranked token', fontsize=10)
    axes[3].set_xlabel('Ranked token', fontsize=10)
    axes[0].set_ylabel('Probability', fontsize=10)
    axes[2].set_ylabel('Probability', fontsize=10)

    fig.suptitle('Sampling Schemes: Quality vs Diversity Tradeoffs', fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig_5_8_sampling_schemes')
    print('✓ Figure 5.8 saved')
    plt.close(fig)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    configure_matplotlib()
    print('=' * 70)
    print('Generating Chapter 5 Figures: Transformers as Conditional Compressors')
    print('=' * 70)
    print()

    figure_5_1()
    figure_5_2()
    figure_5_3()
    figure_5_4()
    figure_5_5()
    figure_5_6()
    figure_5_7()
    figure_5_8()

    print()
    print('=' * 70)
    print('All figures generated successfully!')
    print(f'Output location: {OUTPUT_DIR}')
    print('Formats: PDF (vector) and PNG (high-res raster)')
    print('=' * 70)
