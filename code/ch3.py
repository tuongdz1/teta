"""
Figure generation utilities for Chapter 3 (Geometry of Probability).
Produces the eight figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import stats
import os
import shutil
from pathlib import Path

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Color palette (warm scientific)
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
            'axes.edgecolor': COLOR_GRAY,
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
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def _glow(line, color: str, lw: float) -> None:
    line.set_path_effects(
        [
            patheffects.Stroke(linewidth=lw * 2.2, foreground=color, alpha=0.18),
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
    return dict(boxstyle='round,pad=0.28', facecolor=face, edgecolor=edge, linewidth=1.2, alpha=0.9)

FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0
DPI = 300

OUTPUT_DIR = Path(__file__).resolve().parent / 'figures' / 'ch03_geometry_probability'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', dpi=DPI)
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=DPI)

# ============================================================================
# FIGURE 3.1: KL Divergence as Wasted Bits (ENHANCED)
# ============================================================================

def figure_3_1():
    """Visualize KL divergence as extra bits in cross entropy"""

    # Coin flip probabilities
    p_true = 0.7  # True probability of heads
    q_wrong = 0.5  # Wrong assumption (fair coin)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Show the distributions
    outcomes = ['Tails', 'Heads']
    x = np.arange(len(outcomes))

    p_probs = [1-p_true, p_true]
    q_probs = [1-q_wrong, q_wrong]

    width = 0.35
    bars1 = ax1.bar(x - width/2, p_probs, width, label='True $p$ (prob=0.7)',
                    color=COLOR_PRIMARY, alpha=0.75, edgecolor=INK, linewidth=1.4)
    bars2 = ax1.bar(x + width/2, q_probs, width, label='Wrong $q$ (prob=0.5)',
                    color=COLOR_SECONDARY, alpha=0.75, edgecolor=INK, linewidth=1.4)

    # Add probability values on bars
    for bars, probs in [(bars1, p_probs), (bars2, q_probs)]:
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.1f}', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Probability')
    ax1.set_title('(a) True vs Wrong Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(outcomes)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 0.8])
    style_axes(ax1)

    # Panel (b): Show code lengths and wasted bits
    H_p = -p_true * np.log2(p_true) - (1-p_true) * np.log2(1-p_true)
    H_pq = -p_true * np.log2(q_wrong) - (1-p_true) * np.log2(1-q_wrong)
    KL = H_pq - H_p

    metrics = ['Entropy\n$H(p)$', 'Cross Entropy\n$H(p,q)$']
    xpos = np.arange(len(metrics))

    ax2.bar(xpos[0], H_p, color=COLOR_PRIMARY, alpha=0.78,
            edgecolor=INK, linewidth=1.4, label='$H(p)$')
    ax2.bar(xpos[1], H_p, color=COLOR_PRIMARY, alpha=0.50,
            edgecolor=INK, linewidth=1.1)
    ax2.bar(xpos[1], KL, bottom=H_p, color=COLOR_QUATERNARY, alpha=0.85,
            edgecolor=INK, linewidth=1.1, hatch='///', label='KL (wasted)')

    # Add value labels
    ax2.text(xpos[0], H_p + 0.03, f'{H_p:.3f} bits',
             ha='center', va='bottom', fontsize=9, weight='bold')
    ax2.text(xpos[1], H_pq + 0.03, f'{H_pq:.3f} bits',
             ha='center', va='bottom', fontsize=9, weight='bold')

    # Show wasted bits with thicker arrow
    ax2.annotate('', xy=(xpos[1], H_p), xytext=(xpos[1], H_pq),
                arrowprops=dict(arrowstyle='<->', color=COLOR_QUATERNARY, lw=2,
                                mutation_scale=12, shrinkA=2, shrinkB=2))
    # Keep callout inside axes and slightly lower to avoid value labels
    mid_y = H_p + (H_pq - H_p)/2 - 0.05
    ax2.text(xpos[1] + 0.45, mid_y, f'KL = {KL:.3f} bits\n(wasted)',
            fontsize=10, color=COLOR_QUATERNARY, weight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=PAPER,
                     edgecolor=COLOR_QUATERNARY, linewidth=2))

    ax2.set_ylabel('Bits per flip')
    ax2.set_title('(b) Code Lengths')
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(metrics)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.2])
    style_axes(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig_3_1_kl_wasted_bits')
    print("✓ Figure 3.1 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 3.2: Forward KL vs Reverse KL (ENHANCED)
# ============================================================================

def figure_3_2():
    """Mode covering vs mode seeking behavior with clear terminology"""
    x = np.linspace(-6, 6, 1000)

    # Target: bimodal mixture
    p_target = 0.5 * stats.norm.pdf(x, loc=-2, scale=0.7) + \
               0.5 * stats.norm.pdf(x, loc=2, scale=0.7)

    # Forward KL approximation: wide Gaussian covering both modes
    q_forward = stats.norm.pdf(x, loc=0, scale=2.5)

    # Reverse KL approximation: narrow Gaussian on one mode
    q_reverse = stats.norm.pdf(x, loc=-2, scale=0.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.6, FIGURE_HEIGHT))

    ymax = max(np.max(p_target), np.max(q_forward), np.max(q_reverse)) * 1.12

    # Panel (a): Forward KL (mode covering)
    ax1.fill_between(x, 0, p_target, alpha=0.08, color=COLOR_GRAY)
    lp = ax1.plot(x, p_target, '--', color=COLOR_GRAY, linewidth=2.4, label='Target $p$ (bimodal)')[0]
    lq = ax1.plot(x, q_forward, color=COLOR_PRIMARY, linewidth=2.6,
                 label='$q^*$ minimizing $\\mathrm{KL}(p \\| q)$')[0]
    _glow(lq, COLOR_PRIMARY, 2.6)
    ax1.fill_between(x, 0, q_forward, alpha=0.16, color=COLOR_PRIMARY)

    # Highlight that it covers both modes
    ax1.annotate('Covers both modes\n(mode covering)',
                xy=(0, 0.15), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.5),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=COLOR_LIGHT_BLUE,
                         alpha=0.9, edgecolor=COLOR_GRAY, linewidth=1.2))

    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Forward KL: $\\min_q \\mathrm{KL}(p \\| q)$\n' +
                  'Zero-avoiding (I-projection)', fontsize=10)
    # Keep legend out of the way of the top callout
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, ymax])
    style_axes(ax1)

    # Panel (b): Reverse KL (mode seeking)
    ax2.fill_between(x, 0, p_target, alpha=0.08, color=COLOR_GRAY)
    lp2 = ax2.plot(x, p_target, '--', color=COLOR_GRAY, linewidth=2.4, label='Target $p$ (bimodal)')[0]
    lq2 = ax2.plot(x, q_reverse, color=COLOR_SECONDARY, linewidth=2.6,
                  label='$q^*$ minimizing $\\mathrm{KL}(q \\| p)$')[0]
    _glow(lq2, COLOR_SECONDARY, 2.6)
    ax2.fill_between(x, 0, q_reverse, alpha=0.16, color=COLOR_SECONDARY)

    # Highlight ignored mode; place callout via axes coords so it never gets cropped
    ax2.annotate('Ignores this mode!',
                xy=(2, 0.28),
                xytext=(0.98, 0.80), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=2.0),
                fontsize=9, color=COLOR_QUATERNARY, weight='bold',
                ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc=PAPER, ec=COLOR_QUATERNARY))

    ax2.annotate('Concentrates on\none mode\n(mode seeking)',
                xy=(-2, 0.25), xytext=(-4, 0.15),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.5),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=COLOR_LIGHT_ORANGE,
                         alpha=0.9, edgecolor=COLOR_GRAY, linewidth=1.2))

    ax2.set_xlabel('$x$')
    ax2.set_title('(b) Reverse KL: $\\min_q \\mathrm{KL}(q \\| p)$\n' +
                  'Zero-forcing (M-projection)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, ymax])
    style_axes(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig_3_2_forward_reverse_kl')
    print("✓ Figure 3.2 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 3.3: Jensen-Shannon Divergence (ENHANCED)
# ============================================================================

def figure_3_3():
    """Show JS divergence as symmetric via mixture"""
    x = np.linspace(-5, 8, 1000)

    # Two distributions
    p = stats.norm.pdf(x, loc=0, scale=1)
    q = stats.norm.pdf(x, loc=3, scale=1.5)

    # Mixture
    m = 0.5 * p + 0.5 * q

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Plot distributions
    lp = ax.plot(x, p, color=COLOR_PRIMARY, linewidth=2.7, label='$p(x)$')[0]
    lq = ax.plot(x, q, color=COLOR_SECONDARY, linewidth=2.7, label='$q(x)$')[0]
    lm = ax.plot(x, m, color=COLOR_TERTIARY, linewidth=3.0, linestyle='--',
                label='Mixture $m = \\frac{1}{2}(p+q)$')[0]
    _glow(lp, COLOR_PRIMARY, 2.7)
    _glow(lq, COLOR_SECONDARY, 2.7)
    _glow(lm, COLOR_TERTIARY, 3.0)

    # Shade regions
    ax.fill_between(x, 0, p, alpha=0.15, color=COLOR_PRIMARY)
    ax.fill_between(x, 0, q, alpha=0.15, color=COLOR_SECONDARY)
    ax.fill_between(x, 0, m, alpha=0.08, color=COLOR_TERTIARY)

    # Add formula without covering the main curves
    formula = (
        r'$\mathrm{JS}(p \| q) = \frac{1}{2}\mathrm{KL}(p \| m)$'
        '\n'
        r'$+\frac{1}{2}\mathrm{KL}(q \| m)$'
    )
    ax.text(
        0.04,
        0.06,
        formula,
        transform=ax.transAxes,
        fontsize=10.5,
        ha='left',
        va='bottom',
        bbox=dict(
            boxstyle='round,pad=0.45',
            facecolor=HIGHLIGHT,
            alpha=0.65,
            edgecolor=COLOR_GRAY,
            linewidth=1.3,
        ),
    )

    # Draw arrows to show symmetric nature
    ax.annotate('', xy=(0, 0.38), xytext=(1.5, 0.25),
                arrowprops=dict(arrowstyle='<->', color=COLOR_PRIMARY, lw=2.2))
    ax.annotate('', xy=(3, 0.25), xytext=(1.5, 0.25),
                arrowprops=dict(arrowstyle='<->', color=COLOR_SECONDARY, lw=2.2))

    ax.set_xlabel('$x$')
    ax.set_ylabel('Density')
    ax.set_title('Jensen-Shannon Divergence: Symmetric via Mixture')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.45])
    style_axes(ax)

    plt.tight_layout()
    save_figure(fig, 'fig_3_3_js_divergence')
    print("✓ Figure 3.3 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 3.4: Fisher Information Sensitivity (ENHANCED)
# ============================================================================

def figure_3_4():
    """Show how Fisher information measures likelihood sensitivity"""
    x_data = np.linspace(-3, 3, 1000)

    # Different parameter values
    mu_values = [-0.5, 0, 0.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Log likelihoods for different mu
    colors = [COLOR_QUATERNARY, COLOR_PRIMARY, COLOR_TERTIARY]
    linestyles = ['--', '-', '-.']  # Different line styles for better distinction

    for i, mu in enumerate(mu_values):
        x_obs = 1.0
        log_lik = -0.5 * (x_data - mu)**2 / 1.0

        label = f'$\\mu = {mu}$'
        line = ax1.plot(x_data, log_lik, color=colors[i], linewidth=2.5,
                        linestyle=linestyles[i], label=label)[0]
        _glow(line, colors[i], 2.5)

    # Mark observation point
    ax1.axvline(x_obs, color=COLOR_GRAY, linestyle=':', alpha=0.7, linewidth=2)
    ax1.text(x_obs + 0.15, -1.2, f'Observation\n$x = {x_obs}$', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=PAPER, alpha=0.85))

    ax1.set_xlabel('Parameter $\\mu$')
    ax1.set_ylabel('Log likelihood $\\log p(x=1 \\mid \\mu)$')
    ax1.set_title('(a) Likelihood Sensitivity to $\\mu$')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    style_axes(ax1)

    # Panel (b): Fisher information vs variance (smoother curve)
    sigma_sq_values = np.linspace(0.1, 5, 500)  # Avoid clipping at the top of the axis
    fisher_info = 1 / sigma_sq_values

    ax2.fill_between(sigma_sq_values, fisher_info, color=COLOR_PRIMARY, alpha=0.10)
    line = ax2.plot(sigma_sq_values, fisher_info, color=COLOR_PRIMARY, linewidth=2.6)[0]
    _glow(line, COLOR_PRIMARY, 2.6)

    # Mark a specific point
    sigma_sq_mark = 1.0
    fisher_mark = 1 / sigma_sq_mark
    ax2.plot(sigma_sq_mark, fisher_mark, 'o', markersize=12,
            color=COLOR_SECONDARY, zorder=5, markeredgecolor=INK, markeredgewidth=1.6)
    ax2.text(sigma_sq_mark + 0.25, fisher_mark,
            f'$\\sigma^2={sigma_sq_mark}$\n$I(\\mu)={fisher_mark:.1f}$',
            fontsize=9, weight='bold')

    # Detailed annotations for clarity
    ax2.annotate('High noise $\\Rightarrow$\nlow information',
                xy=(4, 0.25), xytext=(3.2, 1.8),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=2),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=HIGHLIGHT,
                         alpha=0.7, edgecolor=COLOR_GRAY, linewidth=1.4))

    ax2.annotate('Low noise $\\Rightarrow$\nhigh information',
                xy=(0.3, 3.3), xytext=(1.3, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=2),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=HIGHLIGHT,
                         alpha=0.7, edgecolor=COLOR_GRAY, linewidth=1.4))

    ax2.set_xlabel('Variance $\\sigma^2$')
    ax2.set_ylabel('Fisher information $I(\\mu) = 1/\\sigma^2$')
    ax2.set_title('(b) Fisher Information vs Noise Level')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 11])
    style_axes(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig_3_4_fisher_information')
    print("✓ Figure 3.4 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 3.5: I-projection vs M-projection Detailed
# ============================================================================

def figure_3_5():
    """Detailed comparison with numerical KL values"""
    x = np.linspace(-8, 8, 1000)

    # Target: equal mixture of two Gaussians
    p = 0.5 * stats.norm.pdf(x, loc=-2, scale=1) + \
        0.5 * stats.norm.pdf(x, loc=2, scale=1)

    # I-projection result: wide Gaussian centered at 0
    q_I = stats.norm.pdf(x, loc=0, scale=2.23)

    # M-projection result: narrow Gaussian on one mode
    q_M = stats.norm.pdf(x, loc=-2, scale=1)

    ymax = max(np.max(p), np.max(q_I), np.max(q_M)) * 1.08

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.8, FIGURE_HEIGHT*0.9))

    # Panel (a): Target distribution
    line = axes[0].plot(x, p, color=INK, linewidth=2.6)[0]
    _glow(line, INK, 2.6)
    axes[0].fill_between(x, 0, p, alpha=0.16, color=INK)

    # Mark the two modes
    mode_locations = [-2, 2]
    mode_values = [p[np.argmin(np.abs(x - loc))] for loc in mode_locations]
    axes[0].plot(mode_locations, mode_values, 'o', markersize=10, color=COLOR_QUATERNARY,
                markeredgecolor=INK, markeredgewidth=1.4)
    mode_label_y = 0.8 * ymax
    axes[0].text(-2, mode_label_y, 'Mode 1', ha='center', fontsize=8, weight='bold')
    axes[0].text(2, mode_label_y, 'Mode 2', ha='center', fontsize=8, weight='bold')

    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) Target $p(x)$\nBimodal mixture')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, ymax])
    style_axes(axes[0])

    # Panel (b): I-projection
    axes[1].plot(x, p, '--', color=COLOR_GRAY, linewidth=2.0, alpha=0.6, label='Target $p$')
    line = axes[1].plot(x, q_I, color=COLOR_PRIMARY, linewidth=2.6,
                        label='$q^*$ (I-proj)')[0]
    _glow(line, COLOR_PRIMARY, 2.6)
    axes[1].fill_between(x, 0, q_I, alpha=0.18, color=COLOR_PRIMARY)

    axes[1].set_xlabel('$x$')
    axes[1].set_title('(b) I-projection\n$q^* = \\mathcal{N}(0, 5)$\nCovers both modes')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, ymax])
    style_axes(axes[1])

    # Panel (c): M-projection
    axes[2].plot(x, p, '--', color=COLOR_GRAY, linewidth=2.0, alpha=0.6, label='Target $p$')
    line = axes[2].plot(x, q_M, color=COLOR_SECONDARY, linewidth=2.6,
                        label='$q^*$ (M-proj)')[0]
    _glow(line, COLOR_SECONDARY, 2.6)
    axes[2].fill_between(x, 0, q_M, alpha=0.18, color=COLOR_SECONDARY)

    # Show ignored mode
    axes[2].axvline(2, color=COLOR_QUATERNARY, linestyle=':', linewidth=3, alpha=0.7)
    axes[2].text(2, 0.83 * ymax, 'Ignored!', ha='center', fontsize=9,
                color=COLOR_QUATERNARY, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=PAPER,
                         edgecolor=COLOR_QUATERNARY, linewidth=2))

    axes[2].set_xlabel('$x$')
    axes[2].set_title('(c) M-projection\n$q^* = \\mathcal{N}(-2, 1)$\nSeeks one mode')
    # Move legend away from red callout box: bottom-right
    axes[2].legend(loc='lower right', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, ymax])
    style_axes(axes[2])

    plt.tight_layout()
    save_figure(fig, 'fig_3_5_projections_detailed')
    print("✓ Figure 3.5 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 3.6: Statistical Manifold and Geodesics (MAJOR ENHANCEMENT)
# ============================================================================

def figure_3_6():
    """Visualize curved space with geodesic vs Euclidean path - IMPROVED"""
    # Create a better 2D manifold visualization
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, np.pi/2, 50)
    THETA, PHI = np.meshgrid(theta, phi)

    # Parametric surface (hemisphere-like)
    X = np.sin(PHI) * np.cos(THETA)
    Y = np.sin(PHI) * np.sin(THETA)
    Z = np.cos(PHI)

    fig = plt.figure(figsize=(FIGURE_WIDTH*1.3, FIGURE_HEIGHT*1.2))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with better transparency
    ax.plot_surface(X, Y, Z, alpha=0.28, color=COLOR_LIGHT_BLUE,
                   edgecolor=GRID, linewidth=0.35, antialiased=True)

    # Two points on the manifold
    t1, p1 = np.pi/6, np.pi/4
    point1 = np.array([np.sin(p1)*np.cos(t1), np.sin(p1)*np.sin(t1), np.cos(p1)])

    t2, p2 = 5*np.pi/6, np.pi/4
    point2 = np.array([np.sin(p2)*np.cos(t2), np.sin(p2)*np.sin(t2), np.cos(p2)])

    # Plot points with better visibility
    ax.scatter(*point1, color=COLOR_PRIMARY, s=150, marker='o',
              edgecolor=INK, linewidth=1.8, zorder=10, label='$p_1$', depthshade=False)
    ax.scatter(*point2, color=COLOR_SECONDARY, s=150, marker='o',
              edgecolor=INK, linewidth=1.8, zorder=10, label='$p_2$', depthshade=False)

    # Geodesic (great circle arc on hemisphere)
    t_geo = np.linspace(t1, t2, 100)
    p_geo = p1 * np.ones_like(t_geo)
    geo_path = np.array([np.sin(p_geo)*np.cos(t_geo),
                        np.sin(p_geo)*np.sin(t_geo),
                        np.cos(p_geo)])
    geo_line = ax.plot(geo_path[0], geo_path[1], geo_path[2],
                      color=COLOR_TERTIARY, linewidth=4.2, linestyle='-',
                      label='Geodesic\n(natural gradient)',
                      zorder=5, solid_capstyle='round')[0]
    _glow(geo_line, COLOR_TERTIARY, 4)

    # Euclidean path (straight line in 3D)
    euclidean_path = np.array([np.linspace(point1[i], point2[i], 100) for i in range(3)])
    eu_line = ax.plot(euclidean_path[0], euclidean_path[1], euclidean_path[2],
                     color=COLOR_QUATERNARY, linewidth=4.2, linestyle='--',
                     label='Euclidean\n(standard gradient)', zorder=5, solid_capstyle='round')[0]
    _glow(eu_line, COLOR_QUATERNARY, 4)

    # Better axis labels with improved visibility
    ax.set_xlabel('$\\theta_1$', fontsize=11, labelpad=8)
    ax.set_ylabel('$\\theta_2$', fontsize=11, labelpad=8)
    ax.set_zlabel(' ', fontsize=1)  # Minimal z-label
    ax.set_title('Statistical Manifold: Geodesic vs Euclidean Path',
                fontsize=12, pad=15)
    # Nudge legend slightly inward to avoid covering the geodesic start
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.03, 0.97),
        fontsize=9,
        framealpha=0.9,
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=1.6,
        handletextpad=0.7,
    )

    # Better viewing angle for clarity
    ax.view_init(elev=25, azim=50)

    # Improve grid visibility
    ax.grid(True, alpha=0.28, linewidth=0.5)
    ax.xaxis.pane.set_facecolor(PANEL)
    ax.yaxis.pane.set_facecolor(PANEL)
    ax.zaxis.pane.set_facecolor(PANEL)
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)

    # Set better axis limits
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([-0.2, 1.0])
    ax.set_zlim([0.0, 1.0])

    plt.tight_layout()
    save_figure(fig, 'fig_3_6_manifold_geodesic')
    print("✓ Figure 3.6 saved (ENHANCED)")
    plt.close(fig)

# ============================================================================
# FIGURE 3.7: Maximum Entropy Distributions
# ============================================================================

def figure_3_7():
    """Show max entropy distributions for different constraints"""
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.8, FIGURE_HEIGHT*0.9))

    # Panel (a): No constraints → Uniform
    n = 6
    x_discrete = np.arange(n)
    uniform_probs = np.ones(n) / n

    axes[0].bar(x_discrete, uniform_probs, color=COLOR_PRIMARY,
               alpha=0.75, edgecolor=INK, linewidth=1.4)
    axes[0].axhline(1/n, color=COLOR_GRAY, linestyle='--', linewidth=1.5, alpha=0.6)
    axes[0].set_xlabel('Outcome')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('(a) No constraints\nUniform: $p(x) = 1/n$')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 0.3])
    style_axes(axes[0])

    # Panel (b): Known mean → Exponential
    x_exp = np.linspace(0, 5, 300)
    lam = 1
    exp_pdf = lam * np.exp(-lam * x_exp)

    line = axes[1].plot(x_exp, exp_pdf, color=COLOR_SECONDARY, linewidth=2.6)[0]
    _glow(line, COLOR_SECONDARY, 2.6)
    axes[1].fill_between(x_exp, 0, exp_pdf, alpha=0.18, color=COLOR_SECONDARY)

    # Mark the mean
    mean = 1/lam
    axes[1].axvline(mean, color=COLOR_GRAY, linestyle='--', linewidth=1.5, alpha=0.6)
    axes[1].text(mean + 0.3, 0.7, f'Mean = {mean}', fontsize=9)

    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('Density')
    axes[1].set_title('(b) Known mean $\\mu$\nExponential: $p(x) = \\lambda e^{-\\lambda x}$')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 5])
    axes[1].set_ylim([0, 1.2])
    style_axes(axes[1])

    # Panel (c): Known mean and variance → Gaussian
    x_gauss = np.linspace(-4, 4, 300)
    mu, sigma = 0, 1
    gauss_pdf = stats.norm.pdf(x_gauss, mu, sigma)

    line = axes[2].plot(x_gauss, gauss_pdf, color=COLOR_TERTIARY, linewidth=2.6)[0]
    _glow(line, COLOR_TERTIARY, 2.6)
    axes[2].fill_between(x_gauss, 0, gauss_pdf, alpha=0.18, color=COLOR_TERTIARY)

    # Mark mean and variance
    axes[2].axvline(mu, color=COLOR_GRAY, linestyle='--', linewidth=1.5, alpha=0.6)
    axes[2].text(mu + 0.3, 0.35, f'$\\mu = {mu}$', fontsize=9)

    # Show ±1 sigma
    axes[2].axvspan(mu - sigma, mu + sigma, alpha=0.18, color=COLOR_LIGHT_BLUE)
    axes[2].text(0, 0.05, f'$\\sigma^2 = {sigma**2}$', fontsize=9, ha='center')

    axes[2].set_xlabel('$x$')
    axes[2].set_title('(c) Known $\\mu$ and $\\sigma^2$\nGaussian: $\\mathcal{N}(\\mu, \\sigma^2)$')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([-4, 4])
    axes[2].set_ylim([0, 0.45])
    style_axes(axes[2])

    plt.tight_layout()
    save_figure(fig, 'fig_3_7_max_entropy')
    print("✓ Figure 3.7 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 3.8: Natural vs Standard Gradient (ENHANCED WITH ARROWS)
# ============================================================================

def figure_3_8():
    """Compare convergence with directional arrows and loss values"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Create contour plot of loss landscape
    theta1 = np.linspace(-3, 3, 100)
    theta2 = np.linspace(-3, 3, 100)
    T1, T2 = np.meshgrid(theta1, theta2)

    # Ill-conditioned quadratic
    a, b = 0.5, 5
    Loss = a * T1**2 + b * T2**2

    # Panel (a): Standard gradient descent
    levels = np.logspace(0, 2, 15)
    ax1.contour(T1, T2, Loss, levels=levels, colors=GRID, alpha=0.5)
    ax1.contourf(T1, T2, Loss, levels=levels, cmap=WARM_CMAP, alpha=0.25)

    # Standard gradient path
    start = np.array([2.5, 2.5])
    path_std = [start]
    theta = start.copy()
    lr = 0.15
    for _ in range(15):
        grad = np.array([2*a*theta[0], 2*b*theta[1]])
        theta = theta - lr * grad
        path_std.append(theta.copy())

    path_std = np.array(path_std)

    # Plot path with arrows
    line = ax1.plot(path_std[:, 0], path_std[:, 1], 'o-', color=COLOR_QUATERNARY,
                    linewidth=2.2, markersize=4.5, label='Standard gradient', zorder=5, alpha=0.9)[0]
    _glow(line, COLOR_QUATERNARY, 2.2)

    # Add directional arrows at key points
    arrow_indices = [0, 5, 10]
    for idx in arrow_indices:
        if idx < len(path_std) - 1:
            dx = path_std[idx+1, 0] - path_std[idx, 0]
            dy = path_std[idx+1, 1] - path_std[idx, 1]
            ax1.arrow(path_std[idx, 0], path_std[idx, 1], dx*0.5, dy*0.5,
                     head_width=0.15, head_length=0.1, fc=COLOR_QUATERNARY,
                     ec=COLOR_QUATERNARY, zorder=6, alpha=0.7)

    # Mark start and end
    ax1.plot(start[0], start[1], 'o', markersize=12, color=COLOR_TERTIARY,
            markeredgecolor=INK, markeredgewidth=1.6, zorder=10, label='Start')
    ax1.plot(0, 0, '*', markersize=18, color=COLOR_SECONDARY,
            markeredgecolor=INK, markeredgewidth=1.6, zorder=10, label='Optimum')

    # Add loss value at start (placed using axes coords to avoid overlap)
    loss_start = a * start[0]**2 + b * start[1]**2
    ax1.text(0.02, 0.98, f'Loss={loss_start:.1f}', transform=ax1.transAxes,
            va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor=PAPER, alpha=0.9))

    ax1.set_xlabel('$\\theta_1$')
    ax1.set_ylabel('$\\theta_2$')
    ax1.set_title('(a) Standard Gradient Descent\n(15 steps, zigzags)', fontsize=10)
    # Move legend to bottom-left to avoid callout box
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_aspect('equal')
    style_axes(ax1)

    # Panel (b): Natural gradient descent
    ax2.contour(T1, T2, Loss, levels=levels, colors=GRID, alpha=0.5)
    ax2.contourf(T1, T2, Loss, levels=levels, cmap=WARM_CMAP, alpha=0.25)

    # Natural gradient path
    path_nat = [start]
    theta = start.copy()

    for _ in range(6):
        grad = np.array([2*a*theta[0], 2*b*theta[1]])
        nat_grad = np.array([grad[0]/a, grad[1]/b])
        theta = theta - lr * nat_grad
        path_nat.append(theta.copy())

    path_nat = np.array(path_nat)

    # Plot path with arrows
    line = ax2.plot(path_nat[:, 0], path_nat[:, 1], 'o-', color=COLOR_PRIMARY,
                    linewidth=2.7, markersize=5.5, label='Natural gradient', zorder=5)[0]
    _glow(line, COLOR_PRIMARY, 2.7)

    # Add directional arrows
    for idx in range(len(path_nat) - 1):
        dx = path_nat[idx+1, 0] - path_nat[idx, 0]
        dy = path_nat[idx+1, 1] - path_nat[idx, 1]
        ax2.arrow(path_nat[idx, 0], path_nat[idx, 1], dx*0.5, dy*0.5,
                 head_width=0.15, head_length=0.1, fc=COLOR_PRIMARY,
                 ec=COLOR_PRIMARY, zorder=6)

    # Mark start and end
    ax2.plot(start[0], start[1], 'o', markersize=12, color=COLOR_TERTIARY,
            markeredgecolor=INK, markeredgewidth=1.6, zorder=10, label='Start')
    ax2.plot(0, 0, '*', markersize=18, color=COLOR_SECONDARY,
            markeredgecolor=INK, markeredgewidth=1.6, zorder=10, label='Optimum')

    # Add loss value at start (stable placement)
    ax2.text(0.02, 0.98, f'Loss={loss_start:.1f}', transform=ax2.transAxes,
            va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.25', facecolor=PAPER, alpha=0.9))

    ax2.set_xlabel('$\\theta_1$')
    ax2.set_title('(b) Natural Gradient Descent\n(6 steps, more direct)', fontsize=10)
    # Move legend to bottom-left to avoid callout box
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax2.set_aspect('equal')
    style_axes(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig_3_8_natural_gradient')
    print("✓ Figure 3.8 saved (ENHANCED)")
    plt.close(fig)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    configure_matplotlib()
    print("=" * 70)
    print(" ENHANCED Chapter 3 Figures: The Geometry of Probability")
    print("=" * 70)
    print()
    print("Improvements:")
    print("  • Better color contrast and line styles")
    print("  • Improved 3D visualization (Fig 3.6)")
    print("  • Directional arrows on optimization paths (Fig 3.8)")
    print("  • Clearer annotations and labels")
    print("  • Smoother curves and better documentation")
    print()

    figure_3_1()
    figure_3_2()
    figure_3_3()
    figure_3_4()
    figure_3_5()
    figure_3_6()
    figure_3_7()
    figure_3_8()

    print()
    print("=" * 70)
    print("✓ All figures generated successfully!")
    print("=" * 70)
    print(f"Output location: figures/ch03_geometry_probability/")
    print("Formats: PDF (vector) + PNG (high-res raster)")
    print()
    print("Figure Summary:")
    print("  3.1: KL Divergence as Wasted Bits")
    print("  3.2: Forward KL vs Reverse KL (mode covering vs seeking)")
    print("  3.3: Jensen-Shannon Divergence (symmetric)")
    print("  3.4: Fisher Information Sensitivity")
    print("  3.5: I-projection vs M-projection (detailed)")
    print("  3.6: Statistical Manifold and Geodesics [ENHANCED]")
    print("  3.7: Maximum Entropy Distributions")
    print("  3.8: Natural vs Standard Gradient Convergence [ENHANCED]")
    print("=" * 70)
