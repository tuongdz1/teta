"""
Figure generation utilities for Chapter 6 (Representation Learning).
Produces the figures for the chapter.
"""

import os
import numpy as np

# Configure Matplotlib for headless, sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import rc
from matplotlib.patches import (FancyArrowPatch, Circle, Rectangle,
                                FancyBboxPatch, Ellipse, Polygon, Wedge, FancyArrow)
from matplotlib.collections import LineCollection
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy import stats
from scipy.ndimage import gaussian_filter

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Font configuration for publication quality (avoid external LaTeX dependency)
rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif', 'Computer Modern Roman']})
rc('text', usetex=False)
rc('axes', labelsize=11)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)

# Color palette - sophisticated academic colors
COLOR_PRIMARY = '#2E86AB'      # Deep blue
COLOR_SECONDARY = '#A23B72'    # Purple
COLOR_TERTIARY = '#F18F01'     # Orange
COLOR_QUATERNARY = '#C73E1D'   # Red
COLOR_QUINARY = '#6A994E'      # Green
COLOR_GRAY = '#6C757D'
COLOR_LIGHT_BLUE = '#A8DADC'
COLOR_LIGHT_PURPLE = '#D4A5C7'
COLOR_LIGHT_ORANGE = '#FFD56B'
COLOR_LIGHT_GREEN = '#B8D4A8'

# Standard figure size for book
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0

# DPI for high quality
DPI = 300

# Warm scientific palette (align with Chapters 3-5)
WARM_PRIMARY = '#2e5877'
WARM_SECONDARY = '#d58a2f'
WARM_TERTIARY = '#2f8b7b'
WARM_QUATERNARY = '#b4574a'
WARM_GRAY = '#6f6258'
WARM_LIGHT_BLUE = '#c9d8e2'
WARM_LIGHT_ORANGE = '#f0cfa6'
WARM_LIGHT_TEAL = '#d7e6e0'
WARM_LIGHT_PURPLE = '#e4d6e5'
WARM_LIGHT_GREEN = '#d6e4c8'
PAPER = '#fcf6ee'
PANEL = '#f5eadf'
GRID = '#d5c6b8'
INK = '#2b231e'
HIGHLIGHT = '#efd2a0'

CH6_WARM_RC = {
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
    'text.usetex': False,
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
    'xtick.color': WARM_GRAY,
    'ytick.color': WARM_GRAY,
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

WARM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'warm_sci',
    ['#fdf5eb', '#e9caa3', '#d58a2f', '#a65c34', '#5b2b26'],
)

# Create output directory
os.makedirs('figures/ch06_representation_learning', exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def draw_arrow(ax, start, end, color='black', width=0.02, head_width=0.05,
               head_length=0.03, **kwargs):
    """Draw a fancy arrow."""
    arrow = FancyArrow(start[0], start[1],
                      end[0] - start[0], end[1] - start[1],
                      width=width, head_width=head_width,
                      head_length=head_length,
                      fc=color, ec=color, **kwargs)
    ax.add_patch(arrow)

def add_text_box(ax, x, y, text, fontsize=9, color='black',
                 bg_color='white', border_color='black', **kwargs):
    """Add text with a rounded box."""
    ax.text(x, y, text, fontsize=fontsize, color=color,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color,
                    edgecolor=border_color, linewidth=1.5, alpha=0.9),
           ha='center', va='center', **kwargs)


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.9)
    ax.tick_params(color=GRID, labelcolor=WARM_GRAY)


def callout_box(edge: str = GRID, face: str = PAPER) -> dict:
    return dict(boxstyle='round,pad=0.28', facecolor=face, edgecolor=edge, linewidth=1.1, alpha=0.92)

# ============================================================================
# FIGURE 6.1: The Bottleneck Principle - Information Compression
# ============================================================================

def figure_6_1():
    """Visualize how fixed-width residual stream creates compression pressure."""
    with plt.rc_context(CH6_WARM_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FIGURE_WIDTH * 1.75, FIGURE_HEIGHT * 1.05), constrained_layout=True
        )
        fig.patch.set_facecolor(PAPER)

        # Panel 1: Bottleneck diagram
        ax1.set_facecolor(PANEL)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.axis('off')
        ax1.text(
            0.5,
            0.95,
            'Residual Stream Bottleneck',
            ha='center',
            fontsize=11,
            fontweight='bold',
            color=INK,
            transform=ax1.transAxes,
        )

        box_style = "round,pad=0.03,rounding_size=0.04"
        input_box = FancyBboxPatch(
            (0.06, 0.36),
            0.26,
            0.32,
            boxstyle=box_style,
            facecolor=WARM_LIGHT_BLUE,
            edgecolor=GRID,
            linewidth=1.2,
        )
        ax1.add_patch(input_box)
        ax1.text(0.19, 0.54, 'Input tokens', ha='center', va='center', fontsize=9, fontweight='bold', color=INK)
        ax1.text(0.19, 0.42, 'Vocab ~50k\nContext window', ha='center', va='center', fontsize=7.5, color=WARM_GRAY)

        bottleneck = FancyBboxPatch(
            (0.44, 0.44),
            0.08,
            0.16,
            boxstyle=box_style,
            facecolor=WARM_SECONDARY,
            edgecolor=INK,
            linewidth=1.1,
        )
        ax1.add_patch(bottleneck)
        ax1.text(0.48, 0.52, 'width $d$\n(per position)', ha='center', va='center', fontsize=8.5, fontweight='bold', color=INK)
        ax1.text(0.48, 0.36, 'Fixed capacity', ha='center', va='center', fontsize=7.2, color=WARM_GRAY)

        output_box = FancyBboxPatch(
            (0.62, 0.36),
            0.32,
            0.32,
            boxstyle=box_style,
            facecolor=WARM_LIGHT_GREEN,
            edgecolor=GRID,
            linewidth=1.2,
        )
        ax1.add_patch(output_box)
        ax1.text(0.78, 0.54, 'Hidden state', ha='center', va='center', fontsize=9, fontweight='bold', color=INK)
        ax1.text(
            0.78,
            0.42,
            'Syntax, semantics,\ncontext',
            ha='center',
            va='center',
            fontsize=7.5,
            color=WARM_GRAY,
        )

        ax1.add_patch(
            FancyArrowPatch(
                (0.32, 0.52),
                (0.44, 0.52),
                arrowstyle='-|>',
                mutation_scale=12,
                linewidth=1.2,
                color=WARM_PRIMARY,
            )
        )
        ax1.text(0.36, 0.64, 'compress', fontsize=7.5, color=WARM_PRIMARY, ha='center', style='italic')
        ax1.add_patch(
            FancyArrowPatch(
                (0.52, 0.52),
                (0.62, 0.52),
                arrowstyle='-|>',
                mutation_scale=12,
                linewidth=1.2,
                color=WARM_TERTIARY,
            )
        )

        insight_box = FancyBboxPatch(
            (0.06, 0.12),
            0.88,
            0.12,
            boxstyle='round,pad=0.02,rounding_size=0.03',
            facecolor=HIGHLIGHT,
            edgecolor=GRID,
            linewidth=1.1,
        )
        ax1.add_patch(insight_box)
        ax1.text(
            0.5,
            0.18,
            'Model must discover structure to compress efficiently',
            ha='center',
            va='center',
            fontsize=8.5,
            fontweight='bold',
            color=INK,
        )

        # Panel 2: Rate-distortion tradeoff
        ax2.set_facecolor(PANEL)
        style_axes(ax2)
        ax2.set_title('Capacity vs Distortion', fontsize=11, fontweight='bold', pad=8)

        D = np.linspace(0.1, 5, 160)
        R = 4.0 * np.exp(-D * 0.45) + 0.35
        ax2.plot(D, R, color=WARM_PRIMARY, linewidth=2.6, label='Optimal $R(D)$')
        ax2.fill_between(D, 0, R, color=WARM_PRIMARY, alpha=0.15)

        d_opt = 2.0
        r_opt = 4.0 * np.exp(-d_opt * 0.45) + 0.35
        ax2.scatter(
            d_opt,
            r_opt,
            s=120,
            color=WARM_SECONDARY,
            edgecolor=INK,
            linewidth=0.9,
            zorder=5,
        )
        ax2.annotate(
            'Operating point',
            xy=(d_opt, r_opt),
            xytext=(1.3, 3.3),
            fontsize=8.5,
            color=INK,
            bbox=callout_box(),
            arrowprops=dict(arrowstyle='->', lw=1.0, color=WARM_GRAY),
        )

        ax2.annotate(
            'More capacity\n(lower distortion)',
            xy=(0.5, 3.4),
            xytext=(1.6, 4.2),
            fontsize=8,
            ha='center',
            color=WARM_GRAY,
            arrowprops=dict(arrowstyle='->', lw=1.0, color=WARM_GRAY),
        )
        ax2.annotate(
            'Structure discovery\nreduces rate',
            xy=(4.2, 0.85),
            xytext=(3.6, 1.35),
            fontsize=8,
            ha='center',
            color=WARM_GRAY,
            arrowprops=dict(arrowstyle='->', lw=1.0, color=WARM_GRAY),
        )

        ax2.set_xlabel('Distortion $D$ (prediction error)', fontsize=10)
        ax2.set_ylabel('Rate $R$ (bits per token)', fontsize=10)
        ax2.grid(True, alpha=0.4)
        ax2.set_xlim([0, 5.2])
        ax2.set_ylim([0, 5])
        ax2.legend(loc='lower right')

        # layout handled by constrained_layout
        plt.savefig('figures/ch06_representation_learning/fig_6_1_bottleneck_principle.pdf', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_1_bottleneck_principle.png', dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.1 saved")
        plt.close()

# ============================================================================
# FIGURE 6.2: Layer Hierarchy - Surface to Semantic
# ============================================================================

def figure_6_2():
    """Show how representations evolve from surface to semantic across layers."""
    with plt.rc_context(CH6_WARM_RC):
        fig = plt.figure(figsize=(FIGURE_WIDTH * 1.35, FIGURE_HEIGHT * 1.6), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)

        ax.text(
            0.5,
            0.96,
            'Hierarchy of Abstraction Across Layers',
            ha='center',
            fontsize=12,
            fontweight='bold',
            color=INK,
            transform=ax.transAxes,
        )

        layers = [
            ('Embeddings', 'Token IDs + position', WARM_LIGHT_BLUE, 0.84, 0.78),
            ('Early layers', 'Surface stats, local context', WARM_LIGHT_TEAL, 0.69, 0.70),
            ('Middle layers', 'Syntax, dependencies, composition', WARM_LIGHT_PURPLE, 0.52, 0.62),
            ('Late layers', 'Semantics, pragmatics, task cues', WARM_LIGHT_ORANGE, 0.36, 0.64),
            ('Output', 'Next-token distribution', WARM_LIGHT_GREEN, 0.22, 0.72),
        ]

        box_height = 0.13
        for i, (title, content, color, y_pos, width) in enumerate(layers):
            height = box_height
            x_left = 0.5 - width / 2
            ax.add_patch(
                FancyBboxPatch(
                    (x_left, y_pos - height / 2),
                    width,
                    height,
                    boxstyle='round,pad=0.02,rounding_size=0.03',
                    facecolor=color,
                    edgecolor=GRID,
                    linewidth=1.1,
                )
            )
            ax.text(0.5, y_pos + 0.032, title, ha='center', va='center', fontsize=9.6, fontweight='bold', color=INK)
            ax.text(0.5, y_pos - 0.028, content, ha='center', va='center', fontsize=8.0, color=WARM_GRAY)

            if i < len(layers) - 1:
                arrow_start_y = y_pos - height / 2 - 0.015
                arrow_end_y = layers[i + 1][3] + box_height / 2 + 0.015
                ax.add_patch(
                    FancyArrowPatch(
                        (0.5, arrow_start_y),
                        (0.5, arrow_end_y),
                        arrowstyle='-|>',
                        mutation_scale=14,
                        linewidth=1.4,
                        color=WARM_PRIMARY,
                    )
                )

        ax.text(0.05, 0.78, 'Concrete', rotation=90, ha='center', va='center', fontsize=9.5, color=WARM_GRAY)
        ax.text(0.05, 0.30, 'Abstract', rotation=90, ha='center', va='center', fontsize=9.5, color=WARM_GRAY)
        ax.add_patch(
            FancyArrowPatch(
                (0.055, 0.86),
                (0.055, 0.20),
                arrowstyle='-|>',
                mutation_scale=12,
                linewidth=1.1,
                color=WARM_GRAY,
            )
        )

        ax.text(0.92, 0.52, 'Information\nflow', rotation=270, ha='center', va='center', fontsize=9.5, color=WARM_TERTIARY)
        ax.add_patch(
            FancyArrowPatch(
                (0.92, 0.78),
                (0.92, 0.30),
                arrowstyle='-|>',
                mutation_scale=12,
                linewidth=1.1,
                color=WARM_TERTIARY,
            )
        )

        insight_box = FancyBboxPatch(
            (0.14, 0.05),
            0.72,
            0.08,
            boxstyle='round,pad=0.02,rounding_size=0.03',
            facecolor=HIGHLIGHT,
            edgecolor=GRID,
            linewidth=1.1,
        )
        ax.add_patch(insight_box)
        ax.text(
            0.5,
            0.09,
            'Each layer adds abstractions while preserving lower-level signals',
            ha='center',
            va='center',
            fontsize=8.5,
            fontweight='bold',
            color=INK,
        )

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')

        # layout handled by constrained_layout
        plt.savefig('figures/ch06_representation_learning/fig_6_2_layer_hierarchy.pdf', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_2_layer_hierarchy.png', dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.2 saved")
        plt.close()

# ============================================================================
# FIGURE 6.3: Information Bottleneck - I(X;Z) vs I(Z;Y) Plane
# ============================================================================

def figure_6_3():
    """Visualize the information bottleneck tradeoff."""
    with plt.rc_context(CH6_WARM_RC):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)
        ax.set_facecolor(PANEL)
        style_axes(ax)

        I_XZ = np.linspace(0, 12, 120)
        I_ZY_max = 6 * (1 - np.exp(-I_XZ / 4))

        ax.plot(I_XZ, I_ZY_max, color=WARM_PRIMARY, linewidth=2.6,
                label='IB bound', linestyle='--')
        ax.fill_between(I_XZ, 0, I_ZY_max, alpha=0.16, color=WARM_PRIMARY)

        rng = np.random.default_rng(42)
        trajectory_X = np.array([0.5, 1.5, 3, 5, 7, 8.5, 10, 11])
        trajectory_Y = np.array([0.3, 1.2, 2.5, 3.8, 4.5, 5, 5.3, 5.5])
        trajectory_Y = trajectory_Y + rng.normal(0, 0.08, len(trajectory_Y))

        ax.plot(
            trajectory_X,
            trajectory_Y,
            'o-',
            color=WARM_SECONDARY,
            linewidth=2.2,
            markersize=6,
            label='Training trajectory',
            markeredgecolor=INK,
            markeredgewidth=0.6,
        )

        ax.annotate(
            'Early training',
            xy=(trajectory_X[2], trajectory_Y[2]),
            xytext=(4.1, 1.1),
            fontsize=8.5,
            color=INK,
            bbox=callout_box(),
            arrowprops=dict(arrowstyle='->', lw=1.0, color=WARM_GRAY),
        )
        ax.annotate(
            'Compression phase',
            xy=(trajectory_X[5], trajectory_Y[5]),
            xytext=(7.6, 3.4),
            fontsize=8.5,
            color=INK,
            bbox=callout_box(),
            arrowprops=dict(arrowstyle='->', lw=1.0, color=WARM_GRAY),
        )

        ax.text(2.2, 5.2, 'Impossible\nregion', ha='center', fontsize=8.5,
                color=WARM_GRAY, style='italic')
        ax.text(
            2.6,
            0.9,
            'Suboptimal:\nredundant info',
            ha='center',
            fontsize=8,
            color=WARM_QUATERNARY,
            bbox=callout_box(edge=WARM_QUATERNARY, face=PAPER),
        )

        opt_x, opt_y = trajectory_X[-1], trajectory_Y[-1]
        ax.scatter(
            opt_x,
            opt_y,
            s=130,
            color=WARM_TERTIARY,
            marker='*',
            edgecolor=INK,
            linewidth=0.9,
            zorder=5,
            label='Converged',
        )

        ax.set_xlabel('$I(X; Z)$ — information about input', fontsize=10)
        ax.set_ylabel('$I(Z; Y)$ — information about target', fontsize=10)
        ax.set_title('Information Bottleneck: Learning to Compress', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=8.5)
        ax.grid(True, alpha=0.4)
        ax.set_xlim([0, 12])
        ax.set_ylim([0, 7])

        plt.savefig('figures/ch06_representation_learning/fig_6_3_information_bottleneck.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_3_information_bottleneck.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.3 saved")
        plt.close()

# ============================================================================
# FIGURE 6.4: Attention Head Specialization
# ============================================================================

def figure_6_4():
    """Visualize different types of specialized attention heads."""
    with plt.rc_context(CH6_WARM_RC):
        fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT * 1.55), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)

        def plot_attention_pattern(ax, pattern, title, description):
            ax.set_facecolor(PANEL)
            style_axes(ax)
            im = ax.imshow(pattern, cmap=WARM_CMAP, aspect='auto', vmin=0, vmax=1)
            ax.set_xlabel('Key position', fontsize=9, labelpad=6)
            ax.set_ylabel('Query position', fontsize=9)
            ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)
            ax.text(0.5, -0.28, description, transform=ax.transAxes,
                    ha='center', va='top', fontsize=8, color=WARM_GRAY, style='italic')
            return im

        seq_len = 20

        pattern1 = np.zeros((seq_len, seq_len))
        for i in range(3, seq_len):
            if i >= 6:
                pattern1[i, i-5:i-3] = 0.8
            pattern1[i, i-1] = 0.2
        plot_attention_pattern(axes[0, 0], pattern1, 'Induction head', 'Pattern continuation')

        pattern2 = np.zeros((seq_len, seq_len))
        for i in range(1, seq_len):
            pattern2[i, i-1] = 1.0
        plot_attention_pattern(axes[0, 1], pattern2, 'Previous-token head', 'Local dependencies')

        pattern3 = np.zeros((seq_len, seq_len))
        pattern3[15, 3:6] = 0.9
        pattern3[16, 3:6] = 0.85
        for i in range(seq_len):
            if i > 0:
                pattern3[i, max(0, i-2):i] = 0.1
        plot_attention_pattern(axes[1, 0], pattern3, 'Syntactic head', 'Long-range grammar')

        pattern4 = np.zeros((seq_len, seq_len))
        topic_words = [2, 7, 11, 18]
        for i in range(seq_len):
            for tw in topic_words:
                if tw < i:
                    pattern4[i, tw] = 0.6
        row_sums = pattern4.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        pattern4 = pattern4 / row_sums

        im = plot_attention_pattern(axes[1, 1], pattern4, 'Semantic head', 'Topic linkage')

        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, shrink=0.85, aspect=30)
        cbar.set_label('Attention weight', fontsize=9.5)

        fig.suptitle('Attention Head Specialization', fontsize=12.5, fontweight='bold', color=INK, y=1.03)

        plt.savefig('figures/ch06_representation_learning/fig_6_4_attention_specialization.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_4_attention_specialization.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.4 saved")
        plt.close()

# ============================================================================
# FIGURE 6.5: Induction Heads - Pattern Completion
# ============================================================================

def figure_6_5():
    """Detailed visualization of how induction heads work."""
    with plt.rc_context(CH6_WARM_RC):
        fig = plt.figure(figsize=(FIGURE_WIDTH * 1.35, FIGURE_HEIGHT * 1.35), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)

        ax.text(
            0.5,
            0.95,
            'Induction Heads: Pattern Completion',
            ha='center',
            fontsize=12,
            fontweight='bold',
            color=INK,
            transform=ax.transAxes,
        )

        tokens = ['When', 'Mary', 'and', 'John', 'went', 'shopping', ',', 'Mary', 'gave', 'John', '...']
        n_tokens = len(tokens)
        y_pos = 0.75
        token_spacing = 0.86 / n_tokens

        for i, token in enumerate(tokens):
            x = 0.1 + i * token_spacing
            if token in ['Mary', 'John'] and i >= 7:
                face = WARM_LIGHT_ORANGE
                edge = WARM_TERTIARY
                lw = 1.6
            elif token in ['Mary', 'John'] and i < 7:
                face = WARM_LIGHT_BLUE
                edge = WARM_PRIMARY
                lw = 1.3
            else:
                face = PAPER
                edge = GRID
                lw = 1.0

            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.03, y_pos - 0.03),
                    0.06,
                    0.06,
                    boxstyle='round,pad=0.01,rounding_size=0.015',
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=lw,
                )
            )
            ax.text(x, y_pos, token, ha='center', va='center', fontsize=7.5, rotation=15, color=INK)

        mary1_x = 0.1 + 1 * token_spacing
        mary2_x = 0.1 + 7 * token_spacing
        john1_x = 0.1 + 3 * token_spacing
        predict_x = 0.1 + 10 * token_spacing

        ax.add_patch(
            FancyArrowPatch(
                (mary1_x, y_pos - 0.08),
                (mary2_x, y_pos - 0.08),
                arrowstyle='<->',
                mutation_scale=12,
                linewidth=1.4,
                color=WARM_TERTIARY,
            )
        )
        ax.text((mary1_x + mary2_x) / 2, y_pos - 0.13, '1) Detect repetition', ha='center',
                fontsize=8, color=WARM_TERTIARY, fontweight='bold')

        ax.add_patch(
            FancyArrowPatch(
                (mary1_x, y_pos - 0.22),
                (john1_x, y_pos - 0.22),
                arrowstyle='-|>',
                mutation_scale=12,
                linewidth=1.3,
                color=WARM_PRIMARY,
            )
        )
        ax.text((mary1_x + john1_x) / 2, y_pos - 0.27, '2) Retrieve continuation', ha='center',
                fontsize=8, color=WARM_PRIMARY, fontweight='bold')

        ax.add_patch(
            FancyBboxPatch(
                (predict_x - 0.03, y_pos - 0.03),
                0.06,
                0.06,
                boxstyle='round,pad=0.01,rounding_size=0.015',
                facecolor=WARM_LIGHT_GREEN,
                edgecolor=WARM_TERTIARY,
                linewidth=1.6,
                linestyle='--',
            )
        )
        ax.text(predict_x, y_pos + 0.08, '?', ha='center', fontsize=12, fontweight='bold', color=WARM_TERTIARY)

        ax.add_patch(
            FancyArrowPatch(
                (john1_x, y_pos - 0.36),
                (predict_x, y_pos - 0.36),
                arrowstyle='-|>',
                mutation_scale=12,
                linewidth=1.3,
                color=WARM_SECONDARY,
                linestyle='--',
            )
        )
        ax.text((john1_x + predict_x) / 2, y_pos - 0.33, '3) Apply pattern', ha='center',
                fontsize=8, color=WARM_SECONDARY, fontweight='bold')

        ax.add_patch(
            FancyBboxPatch(
                (0.08, 0.06),
                0.84,
                0.24,
                boxstyle='round,pad=0.02,rounding_size=0.02',
                facecolor=WARM_LIGHT_TEAL,
                edgecolor=GRID,
                linewidth=1.1,
                alpha=0.9,
            )
        )
        ax.text(
            0.5,
            0.18,
            'Induction heads match a repeated token,\n'
            'attend to its earlier occurrence, retrieve what followed,\n'
            'and predict the same continuation.',
            ha='center',
            va='center',
            fontsize=8.6,
            color=INK,
            transform=ax.transAxes,
        )

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')

        plt.savefig('figures/ch06_representation_learning/fig_6_5_induction_heads.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_5_induction_heads.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.5 saved")
        plt.close()

# ============================================================================
# FIGURE 6.6: Superposition - Packing More Features Than Dimensions
# ============================================================================

def figure_6_6():
    """Visualize superposition geometry."""
    with plt.rc_context(CH6_WARM_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.85, FIGURE_HEIGHT * 1.08), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)

        # Panel 1: 2D projection of high-D superposition
        ax1.set_facecolor(PANEL)
        ax1.text(
            0.5,
            0.93,
            'Superposition: $n$ features in $d < n$ dimensions',
            ha='center',
            fontsize=11,
            fontweight='bold',
            color=INK,
            transform=ax1.transAxes,
        )

        circle = Circle((0, 0), 1, fill=False, edgecolor=GRID, linewidth=1.6, linestyle='--')
        ax1.add_patch(circle)

        n_features = 8
        rng = np.random.default_rng(42)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        angles += rng.normal(scale=0.08, size=n_features)

        colors = [
            WARM_PRIMARY,
            WARM_SECONDARY,
            WARM_TERTIARY,
            WARM_QUATERNARY,
            '#7aa673',
            '#8ba7b7',
            '#b58ba4',
            '#e0b57a',
        ]

        for i, (angle, color) in enumerate(zip(angles, colors)):
            x, y = np.cos(angle), np.sin(angle)
            ax1.add_patch(
                FancyArrowPatch(
                    (0, 0),
                    (0.9 * x, 0.9 * y),
                    arrowstyle='-|>',
                    mutation_scale=12,
                    linewidth=1.2,
                    color=color,
                    alpha=0.85,
                )
            )
            ax1.text(
                x * 1.12,
                y * 1.12,
                f'$v_{i+1}$',
                ha='center',
                va='center',
                fontsize=8.5,
                bbox=dict(boxstyle='circle,pad=0.12', facecolor=PAPER, edgecolor=GRID, linewidth=1.0),
            )

        ax1.plot([0, 0.9 * np.cos(angles[0])], [0, 0.9 * np.sin(angles[0])], color=INK, linewidth=2.1, alpha=0.7)
        ax1.plot([0, 0.9 * np.cos(angles[1])], [0, 0.9 * np.sin(angles[1])], color=INK, linewidth=2.1, alpha=0.7)
        arc_angles = np.linspace(angles[0], angles[1], 40)
        arc_r = 0.32
        ax1.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles), color=WARM_SECONDARY, linewidth=1.6)
        ax1.text(0.36, 0.14, r'$\theta_{ij}$', fontsize=9.5, color=WARM_SECONDARY, fontweight='bold')

        ax1.text(
            0.5,
            0.06,
            'Dot product: $v_i^\\top v_j = \\cos(\\theta_{ij}) \\approx 0.2$',
            ha='center',
            fontsize=8,
            color=INK,
            transform=ax1.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=HIGHLIGHT, edgecolor=GRID, linewidth=1.0),
        )

        ax1.set_xlim([-1.35, 1.35])
        ax1.set_ylim([-1.35, 1.35])
        ax1.set_aspect('equal')
        ax1.axis('off')

        # Panel 2: Interference from superposition
        ax2.set_facecolor(PANEL)
        style_axes(ax2)
        ax2.set_title('Cost of superposition: interference', fontsize=11, fontweight='bold', pad=4)

        n_samples = 220
        feature_sparsity = 0.1
        true_activations = rng.random((n_samples, 6)) < feature_sparsity
        interference_levels = true_activations.sum(axis=1)
        reconstruction_error = 0.018 * interference_levels + rng.normal(0, 0.009, n_samples)
        reconstruction_error = np.maximum(0, reconstruction_error)

        level_colors = {
            0: '#7aa673',
            1: WARM_PRIMARY,
            2: WARM_SECONDARY,
            3: WARM_QUATERNARY,
        }
        for n_active, error in zip(interference_levels, reconstruction_error):
            color = level_colors.get(int(n_active), '#8a6a55')
            jitter = rng.normal(0, 0.05)
            ax2.scatter(n_active + jitter, error, c=color, s=18, alpha=0.55, edgecolors='none')

        unique_levels = np.arange(0, interference_levels.max() + 1)
        mean_errors = [
            reconstruction_error[interference_levels == level].mean()
            if (interference_levels == level).sum() > 0
            else 0
            for level in unique_levels
        ]
        ax2.plot(unique_levels, mean_errors, linestyle='--', linewidth=1.8, color=INK, label='Mean error')

        ax2.set_xlabel('Number of active features', fontsize=10)
        ax2.set_ylabel('Reconstruction error', fontsize=10)
        ax2.grid(True, alpha=0.4)
        ax2.legend(loc='lower right')
        ax2.set_xticks(unique_levels)

        ax2.text(
            0.96,
            0.85,
            'Interference grows\nwith active features',
            transform=ax2.transAxes,
            ha='right',
            va='top',
            fontsize=8.5,
            color=INK,
            bbox=callout_box(edge=WARM_SECONDARY, face=PAPER),
        )

        plt.suptitle('Superposition: Trading Capacity for Interference', fontsize=12, fontweight='bold', color=INK, y=1.03)

        # layout handled by constrained_layout
        plt.savefig('figures/ch06_representation_learning/fig_6_6_superposition.pdf', dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_6_superposition.png', dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.6 saved")
        plt.close()

# ============================================================================
# FIGURE 6.7: Polysemanticity - One Neuron, Many Meanings
# ============================================================================

def figure_6_7():
    """Visualize polysemantic neurons."""
    with plt.rc_context(CH6_WARM_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT * 1.05), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)

        # Panel 1: Monosemantic
        ax1.set_facecolor(PANEL)
        ax1.text(0.5, 0.95, 'Monosemantic neuron (rare)', ha='center',
                 fontsize=11, fontweight='bold', color=INK, transform=ax1.transAxes)

        ax1.add_patch(Circle((0.5, 0.6), 0.15, facecolor=PAPER,
                             edgecolor=GRID, linewidth=1.6))
        ax1.text(0.5, 0.6, 'Neuron', ha='center', va='center',
                 fontsize=9.5, fontweight='bold', color=INK)

        ax1.add_patch(Circle((0.5, 0.86), 0.08, facecolor=WARM_LIGHT_GREEN,
                             edgecolor=WARM_TERTIARY, linewidth=1.3))
        ax1.text(0.5, 0.86, 'Numbers', ha='center', va='center', fontsize=8, color=INK)

        ax1.add_patch(FancyArrowPatch((0.5, 0.78), (0.5, 0.70),
                                      arrowstyle='-|>', mutation_scale=12,
                                      linewidth=1.2, color=WARM_TERTIARY))

        inputs = ['42', '3.14', '100', '0', '7']
        activations = [0.9, 0.85, 0.92, 0.88, 0.87]
        bar_base_y = 0.22

        for i, (inp, act) in enumerate(zip(inputs, activations)):
            x = 0.15 + i * 0.15
            y = bar_base_y
            ax1.add_patch(Rectangle((x - 0.03, y), 0.06, act * 0.2,
                                    facecolor=WARM_TERTIARY, alpha=0.6,
                                    edgecolor=GRID, linewidth=0.9))
            ax1.text(x, y - 0.08, inp, ha='center', fontsize=7.4, color=INK)
            ax1.text(x, y + act * 0.2 + 0.02, f'{act:.2f}', ha='center',
                     fontsize=7, color=WARM_GRAY)

        ax1.text(0.5, 0.03, 'Clean interpretation', ha='center',
                 fontsize=8.6, color=WARM_TERTIARY,
                 transform=ax1.transAxes, fontweight='bold')

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.axis('off')

        # Panel 2: Polysemantic
        ax2.set_facecolor(PANEL)
        ax2.text(0.5, 0.95, 'Polysemantic neuron (common)', ha='center',
                 fontsize=11, fontweight='bold', color=INK, transform=ax2.transAxes)

        ax2.add_patch(Circle((0.5, 0.6), 0.15, facecolor=GRID,
                             edgecolor=WARM_GRAY, linewidth=1.6))
        ax2.text(0.5, 0.6, 'Neuron', ha='center', va='center',
                 fontsize=9.5, fontweight='bold', color=INK)

        features = [
            ('Numbers', 0.3, 0.75, WARM_LIGHT_ORANGE),
            ('Code', 0.7, 0.75, WARM_LIGHT_BLUE),
            ('Math', 0.2, 0.55, WARM_LIGHT_PURPLE),
            ('Lists', 0.8, 0.55, WARM_LIGHT_GREEN),
        ]
        for feat, x, y, color in features:
            ax2.add_patch(Circle((x, y), 0.06, facecolor=color,
                                 edgecolor=GRID, linewidth=1.0))
            ax2.text(x, y, feat, ha='center', va='center', fontsize=7, color=INK)
            ax2.add_patch(FancyArrowPatch((x, y - 0.06 * np.sign(y - 0.6)),
                                          (0.5 + 0.08 * np.sign(x - 0.5), 0.6 + 0.08 * np.sign(y - 0.6)),
                                          arrowstyle='-|>', mutation_scale=10,
                                          linewidth=1.0, color=WARM_GRAY, alpha=0.7))

        inputs_poly = ['42', 'def', '[1,2]', r'$\int$', 'print']
        activations_poly = [0.8, 0.75, 0.7, 0.65, 0.72]
        colors_poly = [WARM_LIGHT_ORANGE, WARM_LIGHT_BLUE, WARM_LIGHT_GREEN,
                       WARM_LIGHT_PURPLE, WARM_LIGHT_BLUE]

        for i, (inp, act, col) in enumerate(zip(inputs_poly, activations_poly, colors_poly)):
            x = 0.15 + i * 0.15
            y = bar_base_y
            ax2.add_patch(Rectangle((x - 0.03, y), 0.06, act * 0.2,
                                    facecolor=col, alpha=0.7,
                                    edgecolor=GRID, linewidth=0.9))
            ax2.text(x, y - 0.08, inp, ha='center', fontsize=7.4, color=INK)
            ax2.text(x, y + act * 0.2 + 0.02, f'{act:.2f}', ha='center',
                     fontsize=7, color=WARM_GRAY)

        ax2.text(0.5, 0.03, 'Mixed signals → ambiguous meaning', ha='center',
                 fontsize=8.6, color=WARM_QUATERNARY,
                 transform=ax2.transAxes, fontweight='bold')

        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.axis('off')

        plt.suptitle('Polysemanticity: A Consequence of Superposition',
                     fontsize=12, fontweight='bold', color=INK, y=1.02)

        plt.savefig('figures/ch06_representation_learning/fig_6_7_polysemanticity.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_7_polysemanticity.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.7 saved")
        plt.close()

# ============================================================================
# FIGURE 6.8: Probing Accuracy Across Layers
# ============================================================================

def figure_6_8():
    """Show how probe accuracy varies across layers for different features."""
    with plt.rc_context(CH6_WARM_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)

        layers = np.arange(0, 13)
        rng = np.random.default_rng(42)
        pos_accuracy = 50 + 35 * np.exp(-(layers - 5)**2 / 8) + rng.normal(0, 1.5, 13)
        dep_accuracy = 45 + 40 * np.exp(-(layers - 6)**2 / 10) + rng.normal(0, 1.5, 13)
        ner_accuracy = 40 + 45 * np.exp(-(layers - 8)**2 / 12) + rng.normal(0, 1.5, 13)
        sent_accuracy = 35 + 50 * np.exp(-(layers - 10)**2 / 10) + rng.normal(0, 1.5, 13)

        ax1.set_facecolor(PANEL)
        style_axes(ax1)
        ax1.plot(layers, pos_accuracy, 'o-', color=WARM_PRIMARY, linewidth=2.0,
                 markersize=5, label='Part-of-speech', markeredgecolor=INK, markeredgewidth=0.4)
        ax1.plot(layers, dep_accuracy, 's-', color=WARM_SECONDARY, linewidth=2.0,
                 markersize=5, label='Dependencies', markeredgecolor=INK, markeredgewidth=0.4)
        ax1.plot(layers, ner_accuracy, '^-', color=WARM_TERTIARY, linewidth=2.0,
                 markersize=5, label='Named entities', markeredgecolor=INK, markeredgewidth=0.4)
        ax1.plot(layers, sent_accuracy, 'd-', color=WARM_QUATERNARY, linewidth=2.0,
                 markersize=5, label='Sentiment', markeredgecolor=INK, markeredgewidth=0.4)

        ax1.axvspan(-0.5, 3.5, alpha=0.12, color=WARM_LIGHT_BLUE, label='_nolegend_')
        ax1.axvspan(3.5, 8.5, alpha=0.12, color=WARM_LIGHT_TEAL, label='_nolegend_')
        ax1.axvspan(8.5, 12.5, alpha=0.12, color=WARM_LIGHT_ORANGE, label='_nolegend_')

        ax1.text(1.5, 92, 'Early\nsurface', ha='center', fontsize=8, color=INK, bbox=callout_box())
        ax1.text(6.0, 92, 'Middle\nstructure', ha='center', fontsize=8, color=INK, bbox=callout_box())
        ax1.text(10.5, 92, 'Late\nsemantics', ha='center', fontsize=8, color=INK, bbox=callout_box())

        ax1.set_xlabel('Layer', fontsize=10)
        ax1.set_ylabel('Linear probe accuracy (%)', fontsize=10)
        ax1.set_title('(a) Layer-wise feature accessibility', fontsize=11, fontweight='bold')
        # Label lines directly to avoid legend overlap
        labels = [
            ('Part-of-speech', WARM_PRIMARY, 4.0, 82.0),
            ('Dependencies', WARM_SECONDARY, 5.6, 85.0),
            ('Named entities', WARM_TERTIARY, 7.8, 88.5),
            ('Sentiment', WARM_QUATERNARY, 10.6, 84.0),
        ]
        for label, color, x, y in labels:
            ax1.text(
                x,
                y,
                label,
                fontsize=7.5,
                color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=PAPER, edgecolor=GRID, linewidth=0.8),
            )
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim([-0.5, 12.5])
        ax1.set_ylim([30, 95])
        ax1.set_xticks(layers)

        ax2.set_facecolor(PANEL)
        ax2.text(0.5, 0.95, 'What probing tells us', ha='center', fontsize=11, fontweight='bold',
                 color=INK, transform=ax2.transAxes)

        insights = [
            ('Surface features', 'Peak early (layers 1–3)', WARM_PRIMARY),
            ('Syntactic features', 'Peak middle (layers 4–8)', WARM_SECONDARY),
            ('Semantic features', 'Peak late (layers 8–12)', WARM_TERTIARY),
            ('Task-specific', 'Strongest in final layers', WARM_QUATERNARY),
        ]

        y_start = 0.75
        for i, (feature, desc, color) in enumerate(insights):
            y = y_start - i * 0.15
            ax2.add_patch(Circle((0.15, y), 0.03, facecolor=color, edgecolor=GRID, linewidth=1.0,
                                 transform=ax2.transAxes))
            ax2.text(0.22, y, feature, fontsize=9, fontweight='bold', color=INK,
                     transform=ax2.transAxes, va='center')
            ax2.text(0.55, y, desc, fontsize=8, color=WARM_GRAY, style='italic',
                     transform=ax2.transAxes, va='center')

        ax2.add_patch(FancyBboxPatch((0.1, 0.05), 0.8, 0.15,
                                     boxstyle='round,pad=0.02,rounding_size=0.02',
                                     facecolor=HIGHLIGHT, alpha=0.9,
                                     edgecolor=GRID, linewidth=1.1,
                                     transform=ax2.transAxes))
        note = ('High probe accuracy ⇒ information is linearly accessible.\n'
                'Low accuracy doesn’t mean absence; it may be nonlinear.')
        ax2.text(0.5, 0.125, note, ha='center', va='center', fontsize=8, color=INK,
                 transform=ax2.transAxes)

        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.axis('off')

        plt.suptitle('Probing Representations: What Models Learn Where',
                     fontsize=12, fontweight='bold', color=INK, y=1.03)

        plt.savefig('figures/ch06_representation_learning/fig_6_8_probing_accuracy.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_8_probing_accuracy.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.8 saved")
        plt.close()

# ============================================================================
# FIGURE 6.9: Transfer Learning - Why Pretraining Works
# ============================================================================

def figure_6_9():
    """Visualize why pretrained models transfer so well."""
    with plt.rc_context(CH6_WARM_RC):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FIGURE_WIDTH * 2, FIGURE_HEIGHT), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)

        data_amounts = np.logspace(2, 6, 50)

        # Panel 1: From scratch
        ax1.set_facecolor(PANEL)
        style_axes(ax1)
        ax1.text(0.5, 0.95, 'Training from scratch',
                 ha='center', fontsize=11, fontweight='bold', color=INK, transform=ax1.transAxes)
        accuracy_scratch = 85 * (1 - np.exp(-data_amounts / 100000))
        ax1.semilogx(data_amounts, accuracy_scratch, color=WARM_QUATERNARY, linewidth=2.4, label='From scratch')
        ax1.fill_between(data_amounts, 0, accuracy_scratch, alpha=0.15, color=WARM_QUATERNARY)
        ax1.annotate(
            'Needs huge datasets',
            xy=(100000, 60),
            xytext=(7000, 78),
            fontsize=8,
            color=INK,
            bbox=callout_box(),
            arrowprops=dict(arrowstyle='->', lw=1.0, color=WARM_GRAY),
        )
        ax1.set_xlabel('Training examples', fontsize=10)
        ax1.set_ylabel('Task accuracy (%)', fontsize=10)
        ax1.set_title('(a) High sample complexity', fontsize=10, pad=8)
        ax1.grid(True, alpha=0.4)
        ax1.set_ylim([0, 100])
        ax1.legend(loc='lower right', fontsize=8)

        # Panel 2: Fine-tuning
        ax2.set_facecolor(PANEL)
        style_axes(ax2)
        ax2.text(0.5, 0.95, 'Fine-tuning pretrained',
                 ha='center', fontsize=11, fontweight='bold', color=INK, transform=ax2.transAxes)
        accuracy_finetune = 70 + 25 * (1 - np.exp(-data_amounts / 5000))
        ax2.semilogx(data_amounts, accuracy_finetune, color=WARM_TERTIARY, linewidth=2.4, label='Fine-tuning')
        ax2.fill_between(data_amounts, 0, accuracy_finetune, alpha=0.15, color=WARM_TERTIARY)
        ax2.axhline(70, color=WARM_PRIMARY, linestyle='--', linewidth=1.6, label='Pretrained (0-shot)')
        ax2.annotate(
            'High accuracy\nwith few examples',
            xy=(5000, 90),
            xytext=(150000, 88),
            fontsize=8,
            color=INK,
            bbox=callout_box(),
            arrowprops=dict(arrowstyle='->', lw=1.0, color=WARM_GRAY),
        )
        ax2.set_xlabel('Training examples', fontsize=10)
        ax2.set_ylabel('Task accuracy (%)', fontsize=10)
        ax2.set_title('(b) Low sample complexity', fontsize=10, pad=8)
        ax2.grid(True, alpha=0.4)
        ax2.set_ylim([0, 100])
        ax2.legend(loc='lower right', fontsize=8)

        # Panel 3: Why transfer works
        ax3.set_facecolor(PANEL)
        ax3.text(0.5, 0.95, 'Why transfer works',
                 ha='center', fontsize=11, fontweight='bold', color=INK, transform=ax3.transAxes)

        y_positions = [0.75, 0.6, 0.45, 0.3]
        widths = [0.8, 0.75, 0.7, 0.3]
        labels = [
            'Universal features (syntax, semantics)',
            'Domain features (medical, legal, ...)',
            'Task-relevant features',
            'Task head',
        ]
        colors = [WARM_PRIMARY, WARM_SECONDARY, WARM_TERTIARY, WARM_LIGHT_GREEN]
        learned_where = [
            'Learned in\npretraining',
            'Learned in\npretraining',
            'Adapted in\nfine-tuning',
            'New in\nfine-tuning',
        ]

        for i, (y, width, label, color, learned) in enumerate(zip(y_positions, widths, labels, colors, learned_where)):
            x_left = 0.5 - width / 2
            height = 0.1
            ax3.add_patch(
                FancyBboxPatch(
                    (x_left, y - height / 2),
                    width,
                    height,
                    boxstyle='round,pad=0.01,rounding_size=0.02',
                    facecolor=color,
                    edgecolor=GRID,
                    linewidth=1.1,
                    alpha=0.85,
                    transform=ax3.transAxes,
                )
            )
            ax3.text(0.5, y, label, ha='center', va='center', fontsize=8,
                     transform=ax3.transAxes, fontweight='bold', color=INK)
            ax3.text(0.92, y, learned, ha='left', va='center', fontsize=7,
                     transform=ax3.transAxes, style='italic', color=WARM_GRAY)

            if i < len(y_positions) - 1:
                ax3.annotate('', xy=(0.5, y_positions[i+1] + 0.05),
                             xytext=(0.5, y - 0.05), transform=ax3.transAxes,
                             arrowprops=dict(arrowstyle='-|>', lw=1.2, color=WARM_GRAY))

        ax3.text(0.5, 0.08, 'Most features already learned\nOnly the task head needs training',
                 ha='center', va='center', fontsize=8.5, fontweight='bold',
                 transform=ax3.transAxes, color=INK,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=HIGHLIGHT,
                           alpha=0.9, edgecolor=GRID, linewidth=1.1))

        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        ax3.axis('off')

        plt.suptitle('Transfer Learning: The Power of Pretrained Representations',
                     fontsize=12, fontweight='bold', color=INK)

        plt.savefig('figures/ch06_representation_learning/fig_6_9_transfer_learning.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_9_transfer_learning.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.9 saved")
        plt.close()

# ============================================================================
# FIGURE 6.10: Fine-tuning Strategies
# ============================================================================

def figure_6_10():
    """Compare different fine-tuning approaches."""
    with plt.rc_context(CH6_WARM_RC):
        fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT * 1.5), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)

        strategies = [
            ('Feature extraction', 'Freeze all, train head only',
             [0, 0, 0, 0, 0, 1], WARM_PRIMARY),
            ('Full fine-tuning', 'Update all parameters',
             [1, 1, 1, 1, 1, 1], WARM_TERTIARY),
            ('Gradual unfreezing', 'Unfreeze layer by layer',
             [0.3, 0.4, 0.6, 0.8, 0.9, 1], WARM_SECONDARY),
            ('LoRA', 'Add low-rank updates',
             [0.2, 0.2, 0.2, 0.2, 0.2, 1], WARM_QUATERNARY),
        ]

        tradeoffs = {
            'Feature extraction': ('Fast, stable', 'May underfit'),
            'Full fine-tuning': ('Best performance', 'Needs more data'),
            'Gradual unfreezing': ('Balanced', 'More complex'),
            'LoRA': ('Parameter efficient', 'Slight performance drop'),
        }

        for ax, (title, desc, updates, color) in zip(axes.flat, strategies):
            ax.set_facecolor(PANEL)
            ax.text(0.5, 0.97, title, ha='center', va='top', fontsize=10.5,
                    fontweight='bold', color=INK, transform=ax.transAxes)
            ax.text(0.5, 0.90, desc, ha='center', va='top', fontsize=8,
                    style='italic', color=WARM_GRAY, transform=ax.transAxes,
                    bbox=callout_box())

            n_layers = len(updates)
            layer_height = 0.08
            y_start = 0.75

            for i, update_amount in enumerate(updates):
                y = y_start - i * (layer_height + 0.02)
                base_alpha = 0.22
                update_alpha = base_alpha + update_amount * 0.6
                ax.add_patch(
                    FancyBboxPatch(
                        (0.2, y - layer_height / 2),
                        0.6,
                        layer_height,
                        boxstyle='round,pad=0.01,rounding_size=0.02',
                        facecolor=color,
                        edgecolor=GRID,
                        linewidth=1.0,
                        alpha=update_alpha,
                        transform=ax.transAxes,
                    )
                )

                layer_name = f'Layer {i+1}' if i < n_layers - 1 else 'Task head'
                ax.text(0.15, y, layer_name, ha='right', va='center',
                        fontsize=7.2, color=INK, transform=ax.transAxes)

                if update_amount > 0:
                    ax.text(0.85, y, f'{int(update_amount*100)}%', ha='left',
                            va='center', fontsize=7.2, fontweight='bold',
                            color=INK, transform=ax.transAxes)

            pro, con = tradeoffs[title]
            ax.text(0.5, 0.07, f'Pro: {pro}\nCon: {con}', ha='center', va='bottom',
                    fontsize=7.2, transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.28', facecolor=PAPER,
                              alpha=0.95, edgecolor=GRID, linewidth=1.0))

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')

        plt.suptitle('Fine-tuning Strategies: Tradeoffs',
                     fontsize=12, fontweight='bold', color=INK)

        plt.savefig('figures/ch06_representation_learning/fig_6_10_finetuning_strategies.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_10_finetuning_strategies.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.10 saved")
        plt.close()

# ============================================================================
# FIGURE 6.11: Prompting vs Fine-tuning Tradeoffs
# ============================================================================

def figure_6_11():
    """Compare prompting and fine-tuning approaches."""
    with plt.rc_context(CH6_WARM_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT * 1.15), constrained_layout=True
        )
        fig.patch.set_facecolor(PAPER)
        ax1.set_facecolor(PANEL)
        ax2.set_facecolor(PANEL)

        # Panel 1: Comparison table
        ax1.text(
            0.5,
            0.95,
            'Prompting vs Fine-tuning: Different Information Budgets',
            ha='center',
            fontsize=11,
            fontweight='bold',
            color=INK,
            transform=ax1.transAxes,
        )

        # Table headers
        headers = ['Aspect', 'Prompting', 'Fine-tuning']
        col_positions = [0.15, 0.5, 0.85]

        ax1.text(
            col_positions[0],
            0.85,
            headers[0],
            ha='center',
            fontsize=9,
            fontweight='bold',
            color=INK,
            transform=ax1.transAxes,
        )
        ax1.text(
            col_positions[1],
            0.85,
            headers[1],
            ha='center',
            fontsize=9,
            fontweight='bold',
            color=WARM_PRIMARY,
            transform=ax1.transAxes,
        )
        ax1.text(
            col_positions[2],
            0.85,
            headers[2],
            ha='center',
            fontsize=9,
            fontweight='bold',
            color=WARM_QUATERNARY,
            transform=ax1.transAxes,
        )

        # Table rows
        rows = [
            ('Data needed', 'Few examples', 'Many examples'),
            ('Context used', 'High (in prompt)', 'Low (in parameters)'),
            ('Setup time', 'Instant', 'Hours/days'),
            ('Inference cost', 'Higher', 'Lower'),
            ('Performance', 'Good', 'Better'),
            ('Flexibility', 'Very high', 'Task-specific'),
        ]

        y_start = 0.75
        row_height = 0.11

        for i, (aspect, prompt_val, finetune_val) in enumerate(rows):
            y = y_start - i * row_height

            # Row background
            if i % 2 == 0:
                ax1.add_patch(
                    Rectangle(
                        (0.05, y - row_height / 2),
                        0.9,
                        row_height * 0.9,
                        facecolor=WARM_LIGHT_BLUE,
                        alpha=0.35,
                        edgecolor='none',
                        transform=ax1.transAxes,
                    )
                )

            ax1.text(col_positions[0], y, aspect, ha='center', va='center', fontsize=8, color=INK, transform=ax1.transAxes)
            ax1.text(col_positions[1], y, prompt_val, ha='center', va='center', fontsize=8, color=INK, transform=ax1.transAxes)
            ax1.text(col_positions[2], y, finetune_val, ha='center', va='center', fontsize=8, color=INK, transform=ax1.transAxes)

        # Decision guide
        ax1.add_patch(
            FancyBboxPatch(
                (0.05, 0.02),
                0.9,
                0.085,
                boxstyle='round,pad=0.01',
                facecolor=HIGHLIGHT,
                alpha=0.92,
                edgecolor=GRID,
                linewidth=1.2,
                transform=ax1.transAxes,
            )
        )

        guide = (
            'Use prompting for: quick experiments, few examples, flexible tasks\n'
            'Use fine-tuning for: production, lots of data, specialized performance'
        )
        ax1.text(0.5, 0.065, guide, ha='center', va='center', fontsize=8, color=INK, transform=ax1.transAxes)

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.axis('off')

        # Panel 2: Performance vs Data curve
        style_axes(ax2)
        ax2.text(
            0.5,
            0.95,
            'Performance vs Training Data',
            ha='center',
            fontsize=11,
            fontweight='bold',
            color=INK,
            transform=ax2.transAxes,
        )

        n_examples = np.logspace(0, 4, 100)

        # Prompting: starts high, plateaus quickly
        prompt_performance = 70 + 15 * (1 - np.exp(-n_examples / 10))

        # Fine-tuning: starts lower, scales better
        finetune_performance = 50 + 40 * (1 - np.exp(-n_examples / 1000))

        ax2.semilogx(
            n_examples,
            prompt_performance,
            color=WARM_PRIMARY,
            linewidth=2.6,
            label='Prompting (in-context)',
            linestyle='--',
        )
        ax2.semilogx(n_examples, finetune_performance, color=WARM_QUATERNARY, linewidth=2.6, label='Fine-tuning')

        ax2.fill_between(
            n_examples,
            prompt_performance,
            finetune_performance,
            where=(prompt_performance > finetune_performance),
            alpha=0.2,
            color=WARM_PRIMARY,
            label='_nolegend_',
        )
        ax2.fill_between(
            n_examples,
            prompt_performance,
            finetune_performance,
            where=(prompt_performance <= finetune_performance),
            alpha=0.2,
            color=WARM_QUATERNARY,
            label='_nolegend_',
        )

        # Crossover point
        crossover_idx = np.argmin(np.abs(prompt_performance - finetune_performance))
        crossover_x = n_examples[crossover_idx]
        crossover_y = prompt_performance[crossover_idx]

        ax2.scatter(
            crossover_x,
            crossover_y,
            s=190,
            color=WARM_SECONDARY,
            marker='*',
            edgecolor=INK,
            linewidth=1.4,
            zorder=5,
        )
        ax2.annotate(
            f'Crossover\n~{int(crossover_x)} examples',
            xy=(crossover_x, crossover_y),
            xytext=(crossover_x * 0.55, crossover_y - 8),
            fontsize=8,
            ha='right',
            color=INK,
            arrowprops=dict(arrowstyle='->', lw=1.2, color=WARM_GRAY),
        )

        ax2.set_xlabel('Number of Training Examples', fontsize=10)
        ax2.set_ylabel('Task Performance', fontsize=10)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.4)
        ax2.set_xlim([1, 10000])
        ax2.set_ylim([40, 95])

        # layout handled by constrained_layout
        plt.savefig('figures/ch06_representation_learning/fig_6_11_prompting_vs_finetuning.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_11_prompting_vs_finetuning.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.11 saved")
        plt.close()

# ============================================================================
# FIGURE 6.12: Training Dynamics of Representations
# ============================================================================

def figure_6_12():
    """Show how representations evolve during training."""
    with plt.rc_context(CH6_WARM_RC):
        np.random.seed(42)

        fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH * 2, FIGURE_HEIGHT * 1.6), constrained_layout=True)
        fig.patch.set_facecolor(PAPER)

        # Simulate latent space at different training stages
        n_samples = 100
        n_classes = 4

        stages = ['Initialization', 'Early Training', 'Mid Training', 'Late Training', 'Converged', 'Metrics']

        for idx, (ax, stage) in enumerate(zip(axes.flat[:5], stages[:5])):
            ax.set_facecolor(PANEL)
            style_axes(ax)
            if stage == 'Initialization':
                # Random, no structure
                data = np.random.randn(n_samples, 2) * 2
                labels = np.random.randint(0, n_classes, n_samples)
                title_color = WARM_QUATERNARY

            elif stage == 'Early Training':
                # Slight clustering emerging
                centers = np.random.randn(n_classes, 2) * 1.5
                data = []
                labels = []
                for i in range(n_classes):
                    cluster = centers[i] + np.random.randn(n_samples // n_classes, 2) * 0.8
                    data.append(cluster)
                    labels.extend([i] * (n_samples // n_classes))
                data = np.vstack(data)
                labels = np.array(labels)
                title_color = WARM_SECONDARY

            elif stage == 'Mid Training':
                # Clear clusters forming
                centers = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]])
                data = []
                labels = []
                for i in range(n_classes):
                    cluster = centers[i] + np.random.randn(n_samples // n_classes, 2) * 0.5
                    data.append(cluster)
                    labels.extend([i] * (n_samples // n_classes))
                data = np.vstack(data)
                labels = np.array(labels)
                title_color = WARM_TERTIARY

            elif stage == 'Late Training':
                # Tight, well-separated clusters
                centers = np.array([[-2.5, -2.5], [2.5, -2.5], [-2.5, 2.5], [2.5, 2.5]])
                data = []
                labels = []
                for i in range(n_classes):
                    cluster = centers[i] + np.random.randn(n_samples // n_classes, 2) * 0.3
                    data.append(cluster)
                    labels.extend([i] * (n_samples // n_classes))
                data = np.vstack(data)
                labels = np.array(labels)
                title_color = WARM_QUATERNARY

            else:  # Converged
                # Very tight, well-separated
                centers = np.array([[-3, -3], [3, -3], [-3, 3], [3, 3]])
                data = []
                labels = []
                for i in range(n_classes):
                    cluster = centers[i] + np.random.randn(n_samples // n_classes, 2) * 0.2
                    data.append(cluster)
                    labels.extend([i] * (n_samples // n_classes))
                data = np.vstack(data)
                labels = np.array(labels)
                title_color = WARM_PRIMARY

            # Plot
            colors = [WARM_PRIMARY, WARM_SECONDARY, WARM_TERTIARY, WARM_QUATERNARY]
            for i in range(n_classes):
                mask = labels == i
                ax.scatter(
                    data[mask, 0],
                    data[mask, 1],
                    c=colors[i],
                    s=20,
                    alpha=0.6,
                    edgecolors=INK,
                    linewidth=0.3,
                )

            # Prior contour (standard Gaussian)
            x_range = np.linspace(-4, 4, 100)
            y_range = np.linspace(-4, 4, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = stats.multivariate_normal.pdf(np.dstack([X, Y]), mean=[0, 0], cov=[[1, 0], [0, 1]])
            ax.contour(X, Y, Z, levels=3, colors=GRID, alpha=0.45, linewidths=0.9)

            ax.set_xlabel('$z_1$', fontsize=9)
            ax.set_ylabel('$z_2$', fontsize=9)
            ax.set_title(stage, fontsize=10, fontweight='bold', color=title_color, pad=6)
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.25)

        # Panel 6: Metrics over training
        ax_metrics = axes.flat[5]
        ax_metrics.set_facecolor(PANEL)
        style_axes(ax_metrics)

        epochs = np.linspace(0, 100, 100)

        # Cluster separation
        separation = 1 - np.exp(-epochs / 20)

        # Alignment with prior
        alignment = 0.3 + 0.5 * (1 - np.exp(-epochs / 30))

        # Probe accuracy
        probe_acc = 50 + 45 * (1 - np.exp(-epochs / 25))

        ax_metrics.plot(epochs, separation, color=WARM_TERTIARY, linewidth=2.4, label='Cluster separation')
        ax_metrics.plot(epochs, alignment, color=WARM_PRIMARY, linewidth=2.4, label='Prior alignment')

        ax_metrics_twin = ax_metrics.twinx()
        ax_metrics_twin.plot(
            epochs,
            probe_acc,
            color=WARM_QUATERNARY,
            linewidth=2.4,
            linestyle='--',
            label='Probe accuracy',
        )

        for spine in ax_metrics_twin.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.9)
        ax_metrics_twin.tick_params(color=GRID, labelcolor=WARM_GRAY)

        ax_metrics.set_xlabel('Training Epoch', fontsize=9)
        ax_metrics.set_ylabel('Normalized Metric', fontsize=9)
        ax_metrics_twin.set_ylabel('Probe Accuracy (%)', fontsize=9)
        ax_metrics.set_title('Training Metrics', fontsize=10, fontweight='bold', color=INK)

        # Combine legends
        lines1, labels1 = ax_metrics.get_legend_handles_labels()
        lines2, labels2 = ax_metrics_twin.get_legend_handles_labels()
        ax_metrics.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

        ax_metrics.grid(True, alpha=0.3)
        ax_metrics.set_ylim([0, 1])
        ax_metrics_twin.set_ylim([0, 100])

        plt.suptitle('Evolution of Representations During Training', fontsize=13, fontweight='bold', color=INK, y=1.02)

        # layout handled by constrained_layout
        plt.savefig('figures/ch06_representation_learning/fig_6_12_training_dynamics.pdf',
                    dpi=DPI, bbox_inches='tight')
        plt.savefig('figures/ch06_representation_learning/fig_6_12_training_dynamics.png',
                    dpi=DPI, bbox_inches='tight')
        print("✓ Figure 6.12 saved")
        plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Generating Chapter 6 Figures: Representation Learning")
    print("Beautiful, informative visualizations of what transformers learn")
    print("=" * 80)
    print()

    figure_6_1()
    figure_6_2()
    figure_6_3()
    figure_6_4()
    figure_6_5()
    figure_6_6()
    figure_6_7()
    figure_6_8()
    figure_6_9()
    figure_6_10()
    figure_6_11()
    figure_6_12()

    print()
    print("=" * 80)
    print("All figures generated successfully!")
    print("Output location: figures/ch06_representation_learning/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 80)
    print()
    print("Figure Summary:")
    print("  6.1:  Bottleneck Principle - Information Compression")
    print("  6.2:  Layer Hierarchy - Surface to Semantic")
    print("  6.3:  Information Bottleneck - I(X;Z) vs I(Z;Y)")
    print("  6.4:  Attention Head Specialization")
    print("  6.5:  Induction Heads - Pattern Completion Mechanism")
    print("  6.6:  Superposition - Packing Features in Limited Dimensions")
    print("  6.7:  Polysemanticity - One Neuron, Many Meanings")
    print("  6.8:  Probing Accuracy Across Layers")
    print("  6.9:  Transfer Learning - Why Pretraining Works")
    print("  6.10: Fine-tuning Strategies Comparison")
    print("  6.11: Prompting vs Fine-tuning Tradeoffs")
    print("  6.12: Training Dynamics of Representations")
    print()
