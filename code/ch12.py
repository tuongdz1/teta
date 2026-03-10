"""
Figure generation utilities for Chapter 12 (Scaling Laws and Training at Scale).
Produces the nine figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Patch
from matplotlib.patches import Polygon
from matplotlib import patheffects as pe
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy import stats
import os

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# LaTeX font configuration for publication quality
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
rc('axes', labelsize=12)
rc('xtick', labelsize=11)
rc('ytick', labelsize=11)
rc('legend', fontsize=10)

# Color palette - blue/orange academic
COLOR_PRIMARY = '#1f77b4'    # Blue
COLOR_SECONDARY = '#ff7f0e'  # Orange
COLOR_TERTIARY = '#2ca02c'   # Green
COLOR_QUATERNARY = '#d62728' # Red
COLOR_GRAY = '#7f7f7f'
COLOR_LIGHT_BLUE = '#aec7e8'
COLOR_LIGHT_ORANGE = '#ffbb78'
COLOR_PURPLE = '#9467bd'
COLOR_BEIGE = '#f5e6c4'

# Consistent annotation styles
ANNOTATION_BOX = dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.8, edgecolor=COLOR_GRAY, linewidth=0.8)
ANNOTATION_BOX_STRONG = dict(boxstyle='round,pad=0.45', facecolor='white',
                             alpha=0.9, edgecolor=COLOR_GRAY, linewidth=1.0)

def _standard_layout(fig, wspace=0.35, hspace=None):
    """Apply consistent margins/padding across subplots."""
    if hspace is None:
        fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.15, wspace=wspace)
    else:
        fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.15, wspace=wspace, hspace=hspace)

# Standard figure size for book (width in inches)
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0

# DPI for high quality
DPI = 300

# Create output directory
os.makedirs('figures/ch12_uncertainty', exist_ok=True)

# ============================================================================
# FIGURE 12.1: Chinchilla vs GPT-3 Allocation
# ============================================================================

def figure_12_1():
    """Compare GPT-3 and Chinchilla resource allocation"""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH * 1.55, FIGURE_HEIGHT),
        gridspec_kw={'wspace': 0.42}
    )

    models = ['GPT-3', 'Chinchilla']

    # Data (in billions for parameters, billions for tokens)
    params = np.array([175, 70])
    tokens = np.array([300, 1400])

    # Panel (a): Parameters vs Data (normalized to a common scale)
    ax1 = axes[0]
    x_pos = np.arange(len(models))
    width = 0.35

    # Normalize to max = 1 for clear comparison on one axis
    p_norm = params / params.max()
    t_norm = tokens / tokens.max()

    bars1 = ax1.bar(x_pos - width/2, p_norm, width, label='Parameters (rel.)',
                    color=COLOR_PRIMARY, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x_pos + width/2, t_norm, width, label='Training Tokens (rel.)',
                    color=COLOR_SECONDARY, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add absolute value labels inside bars for context
    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                 f'{int(val)}B', ha='center', va='center', fontsize=10, weight='bold',
                 color='white',
                 path_effects=[pe.withStroke(linewidth=2, foreground='black', alpha=0.35)])
    for bar, val in zip(bars2, tokens):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                 f'{int(val)}B', ha='center', va='center', fontsize=10, weight='bold',
                 color='white',
                 path_effects=[pe.withStroke(linewidth=2, foreground='black', alpha=0.35)])

    ax1.set_ylabel('Relative scale (max = 1.0)', fontsize=13)
    ax1.set_title('(a) Resource Allocation (normalized)', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models)
    ax1.set_ylim([0, 1.22])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2,
               fontsize=9, framealpha=0.95, facecolor='white', edgecolor=COLOR_GRAY)

    # Add annotation in whitespace
    ax1.text(0.03, 0.9, 'Similar total compute\n$\\sim 5 \\times 10^{23}$ FLOPs',
            transform=ax1.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                      alpha=0.85, edgecolor='none'))

    # Panel (b): Inference Cost Comparison
    ax2 = axes[1]

    # Inference compute is roughly proportional to parameter count (at fixed context length).
    # We plot a relative proxy here to emphasize the scaling, not an exact FLOPs/token figure.
    inference_cost = params.copy()

    bars3 = ax2.bar(x_pos, inference_cost, color=[COLOR_QUATERNARY, COLOR_TERTIARY],
                    alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels and savings
    for bar, val in zip(bars3, inference_cost):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 5,
                f'{int(val)}B params', ha='center', fontsize=10, weight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=0.9)])

    # Arrow showing savings
    ax2.text(0.96, 0.9, '2.5× cheaper\nper query!', ha='right', va='top', fontsize=10,
             transform=ax2.transAxes, weight='bold', color=COLOR_TERTIARY,
             bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                       edgecolor=COLOR_TERTIARY, linewidth=1.2, alpha=0.9))

    ax2.set_ylabel('Inference Cost ($\\propto$ parameters)', fontsize=13)
    ax2.set_title('(b) Deployment Efficiency', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 220])

    plt.tight_layout()
    _standard_layout(fig, wspace=0.45)
    plt.savefig('figures/ch12_uncertainty/fig_12_1_chinchilla_vs_gpt3.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_1_chinchilla_vs_gpt3.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.1 saved")
    plt.close()

# ============================================================================
# FIGURE 12.2: The Three Scaling Laws (Power Laws)
# ============================================================================

def figure_12_2():
    """Show the three fundamental scaling laws on log-log plots"""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT),
        gridspec_kw={'wspace': 0.28}
    )

    # Panel (a): Loss vs Compute
    ax1 = axes[0]

    C = np.logspace(18, 24, 100)  # Compute in FLOPs
    L_inf = 1.5  # Irreducible loss
    alpha = 0.05
    C_0 = 1e21

    L_C = (C / C_0)**(-alpha) + L_inf

    curve_C, = ax1.loglog(C, L_C, color=COLOR_PRIMARY, linewidth=3.8)
    curve_C.set_path_effects([pe.SimpleLineShadow(offset=(1, -1), alpha=0.2),
                              pe.Normal()])
    ax1.axhline(L_inf, color=COLOR_GRAY, linestyle='--', linewidth=2.2,
               label='$L_\\infty$ (irreducible)')

    # Annotate slope in white space with leader line
    slope_point_C = 1.2e22
    slope_point_L = (slope_point_C / C_0)**(-alpha) + L_inf
    ax1.annotate(r'Slope $\alpha \approx 0.05$', xy=(slope_point_C, slope_point_L),
                 xytext=(6e22, 3.05),
                 fontsize=9, color=COLOR_QUATERNARY, weight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                           alpha=0.9, edgecolor=COLOR_QUATERNARY, linewidth=0.8),
                 arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=1.6,
                                 connectionstyle='arc3,rad=-0.25'))

    ax1.set_xlabel('Compute $C$ (FLOPs)')
    ax1.set_ylabel('Loss $L$')
    ax1.set_title('(a) $L(C) = (C/C_0)^{-\\alpha} + L_\\infty$', pad=16, fontsize=14)
    ax1.grid(True, alpha=0.35, which='major', linestyle='--', linewidth=0.7)
    ax1.grid(True, alpha=0.1, which='minor', linestyle=':', linewidth=0.5)
    # Mark reference scale
    ax1.axvline(C_0, color=COLOR_GRAY, linestyle='--', linewidth=1.8, alpha=0.6)
    ax1.text(C_0*1.05, 3.85, '$C_0$', fontsize=9, color=COLOR_GRAY)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95,
               facecolor='white', edgecolor=COLOR_GRAY)
    ax1.set_xlim([1e19, 4e23])
    ax1.set_ylim([1.45, 4])

    # Panel (b): Loss vs Parameters
    ax2 = axes[1]

    N = np.logspace(6, 12, 100)  # Parameters
    beta = 0.076
    N_0 = 1e9

    L_N = (N / N_0)**(-beta) + L_inf

    curve_N, = ax2.loglog(N, L_N, color=COLOR_SECONDARY, linewidth=3.8)
    curve_N.set_path_effects([pe.SimpleLineShadow(offset=(1, -1), alpha=0.2),
                              pe.Normal()])
    ax2.axhline(L_inf, color=COLOR_GRAY, linestyle='--', linewidth=3)

    slope_point_N = 1.5e10
    slope_point_LN = (slope_point_N / N_0)**(-beta) + L_inf
    ax2.annotate(r'Slope $\beta \approx 0.076$', xy=(slope_point_N, slope_point_LN),
                 xytext=(2e11, 3.2),
                 fontsize=9, color=COLOR_QUATERNARY, weight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                           alpha=0.9, edgecolor=COLOR_QUATERNARY, linewidth=0.8),
                 arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=1.6,
                                 connectionstyle='arc3,rad=-0.3'))

    ax2.set_xlabel('Parameters $N$')
    ax2.set_ylabel('Loss $L$')
    ax2.set_title('(b) $L(N) = (N/N_0)^{-\\beta} + L_\\infty$', pad=16, fontsize=14)
    ax2.grid(True, alpha=0.35, which='major', linestyle='--', linewidth=0.7)
    ax2.grid(True, alpha=0.1, which='minor', linestyle=':', linewidth=0.5)
    # Mark reference scale
    ax2.axvline(N_0, color=COLOR_GRAY, linestyle='--', linewidth=1.8, alpha=0.6)
    ax2.text(N_0*1.1, 3.5, '$N_0$', fontsize=9, color=COLOR_GRAY)
    ax2.set_xlim([2e7, 3e12])
    ax2.set_ylim([1.45, 3.6])

    # Panel (c): Loss vs Data
    ax3 = axes[2]

    D = np.logspace(8, 13, 100)  # Tokens
    gamma = 0.095
    D_0 = 1e10

    L_D = (D / D_0)**(-gamma) + L_inf

    curve_D, = ax3.loglog(D, L_D, color=COLOR_TERTIARY, linewidth=3.8)
    curve_D.set_path_effects([pe.SimpleLineShadow(offset=(1, -1), alpha=0.2),
                              pe.Normal()])
    ax3.axhline(L_inf, color=COLOR_GRAY, linestyle='--', linewidth=3)

    slope_point_D = 1.8e11
    slope_point_LD = (slope_point_D / D_0)**(-gamma) + L_inf
    ax3.annotate(r'Slope $\gamma \approx 0.095$', xy=(slope_point_D, slope_point_LD),
                 xytext=(4e12, 2.75),
                 fontsize=9, color=COLOR_QUATERNARY, weight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                           alpha=0.9, edgecolor=COLOR_QUATERNARY, linewidth=0.8),
                 arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=1.6,
                                 connectionstyle='arc3,rad=-0.25'))

    ax3.set_xlabel('Training Tokens $D$')
    ax3.set_ylabel('Loss $L$')
    ax3.set_title('(c) $L(D) = (D/D_0)^{-\\gamma} + L_\\infty$', pad=16, fontsize=14)
    ax3.grid(True, alpha=0.35, which='major', linestyle='--', linewidth=0.7)
    ax3.grid(True, alpha=0.1, which='minor', linestyle=':', linewidth=0.5)
    # Mark reference scale
    ax3.axvline(D_0, color=COLOR_GRAY, linestyle='--', linewidth=1.8, alpha=0.6)
    ax3.text(D_0*1.3, 3.35, '$D_0$', fontsize=9, color=COLOR_GRAY)
    ax3.set_xlim([1e8, 3e13])
    ax3.set_ylim([1.45, 3.5])

    plt.tight_layout()
    _standard_layout(fig, wspace=0.32)
    plt.savefig('figures/ch12_uncertainty/fig_12_2_scaling_laws.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_2_scaling_laws.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.2 saved")
    plt.close()

# ============================================================================
# FIGURE 12.3: Compute-Optimal Frontier
# ============================================================================

def figure_12_3():
    """2D visualization of compute-optimal frontier"""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Create grid of (N, D) values
    N = np.logspace(8, 12, 100)  # Parameters (100M to 1T)
    D = np.logspace(9, 13.75, 100)  # Tokens (1B to ~56T)

    N_grid, D_grid = np.meshgrid(N, D)

    # Compute constraint: C = 6ND (approximate)
    C_grid = 6 * N_grid * D_grid

    # Loss function (simplified model)
    L_inf = 1.5
    beta = 0.076
    gamma = 0.095
    N_0 = 1e9
    D_0 = 1e10

    L_grid = (N_grid / N_0)**(-beta) + (D_grid / D_0)**(-gamma) + L_inf

    # Plot loss contours (simplified - fewer lines, more visible)
    levels = [2.0, 2.5, 3.0]
    contour = ax.contour(np.log10(N_grid), np.log10(D_grid), L_grid,
                         levels=levels, colors=COLOR_GRAY, alpha=0.25, linewidths=1.2)
    try:
        ax.clabel(contour, inline=True, fontsize=8, fmt='L=%.1f', colors=COLOR_GRAY)
    except Exception:
        pass

    # Plot compute budget iso-lines
    C_values = [1e20, 1e22, 1e24]
    for idx, C_val in enumerate(C_values):
        D_line = C_val / (6 * N)
        valid = (D_line >= D.min()) & (D_line <= D.max())
        ax.plot(np.log10(N[valid]), np.log10(D_line[valid]), '--',
                color='#5a8ab8', alpha=0.35, linewidth=1.3)
        # Label all iso-lines with better visibility
        label_idx = len(N[valid]) // 2
        if valid.sum() > 0:
            sci_label = int(np.log10(C_val))
            ax.text(np.log10(N[valid][label_idx]), np.log10(D_line[valid][label_idx]) + 0.15,
                    f'$C=10^{{{sci_label}}}$', fontsize=7.5, rotation=-45,
                    color=COLOR_GRAY, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.92, edgecolor=COLOR_GRAY, linewidth=0.6))

    # Compute-optimal path (Chinchilla scaling)
    C_opt = np.logspace(20, 24, 50)
    N_opt = 0.73 * C_opt**0.49
    D_opt = 22.8 * C_opt**0.51

    ax.plot(np.log10(N_opt), np.log10(D_opt), color=COLOR_SECONDARY,
            linewidth=7.5, alpha=1.0, label='Compute-optimal path', zorder=6)

    # Mark specific points
    # GPT-3 style (overparameterized)
    N_gpt3 = 175e9
    D_gpt3 = 300e9
    # Subtle glow behind markers
    ax.plot(np.log10(N_gpt3), np.log10(D_gpt3), 'o', color=COLOR_QUATERNARY,
            markersize=22, alpha=0.15, markeredgecolor='none', zorder=6, label='_nolegend_')
    ax.plot(np.log10(N_gpt3), np.log10(D_gpt3), 'o', color=COLOR_QUATERNARY,
            markersize=16, markeredgecolor='white', markeredgewidth=1.8,
            label='GPT-3 style (overparameterized)', zorder=7)

    # Chinchilla style (optimal)
    N_chinchilla = 70e9
    D_chinchilla = 1400e9
    ax.plot(np.log10(N_chinchilla), np.log10(D_chinchilla), 's', color=COLOR_TERTIARY,
            markersize=22, alpha=0.15, markeredgecolor='none', zorder=6, label='_nolegend_')
    ax.plot(np.log10(N_chinchilla), np.log10(D_chinchilla), 's', color=COLOR_TERTIARY,
            markersize=16, markeredgecolor='white', markeredgewidth=1.8,
            label='Chinchilla style (compute-optimal)', zorder=7)

    ax.set_xlabel('$\\log_{10}$ Parameters $N$')
    ax.set_ylabel('$\\log_{10}$ Training Tokens $D$')
    ax.set_title('Compute-Optimal Frontier: $C = 6ND$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95,
              facecolor='white', edgecolor=COLOR_GRAY)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([8, 12])
    ax.set_ylim([9, 13.75])

    # Add annotations - repositioned to stay within boundaries
    ax.text(11.3, 10.0, 'Undertrained\n(too many params)', fontsize=8,
            ha='center', va='center', color=COLOR_QUATERNARY, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.9, edgecolor=COLOR_QUATERNARY, linewidth=0.8))
    ax.text(8.8, 12.3, 'Overtrained\n(wasted compute)', fontsize=8,
            ha='center', va='center', color=COLOR_QUATERNARY, style='italic',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.9, edgecolor=COLOR_QUATERNARY, linewidth=0.8))

    plt.tight_layout()
    _standard_layout(fig, wspace=0.35)
    plt.savefig('figures/ch12_uncertainty/fig_12_3_compute_optimal_frontier.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_3_compute_optimal_frontier.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.3 saved")
    plt.close()

# ============================================================================
# FIGURE 12.4: Emergent Abilities and Phase Transitions
# ============================================================================

def figure_12_4():
    """Show sharp capability emergence at specific scales"""
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.8, FIGURE_HEIGHT))

    # Model sizes (log scale)
    model_sizes = np.logspace(8, 11, 100)  # 100M to 100B parameters

    # Panel (a): In-context learning
    ax1 = axes[0]

    # Sigmoid-like emergence around 1B parameters
    threshold_1 = np.log10(1e9)
    steepness_1 = 3
    accuracy_icl = 100 / (1 + np.exp(-steepness_1 * (np.log10(model_sizes) - threshold_1)))

    ax1.semilogx(model_sizes, accuracy_icl, color=COLOR_PRIMARY, linewidth=3.4)
    ax1.axvline(1e9, color=COLOR_GRAY, linestyle='--', linewidth=3.0, alpha=0.6)
    ax1.text(1.3e9, 52, '~1B params', va='center', fontsize=10,
             color=COLOR_GRAY,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.85, edgecolor='none'))

    ax1.set_xlabel('Model Size (parameters, log scale)')
    ax1.set_ylabel('Accuracy (\\%)')
    ax1.set_title('(a) In-Context Learning', fontsize=14)
    ax1.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
    ax1.set_ylim([0, 102])

    # Shade emergence region
    ax1.axvspan(5e8, 2e9, alpha=0.35, color=COLOR_BEIGE,
                label='Emergence region')
    ax1.text(8e8, 90, 'Emergence region', fontsize=10, color=COLOR_SECONDARY,
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.85, edgecolor='none'))
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.92,
               facecolor='white', edgecolor=COLOR_GRAY)

    # Panel (b): Chain-of-thought reasoning
    ax2 = axes[1]

    # Sigmoid emergence around 10B parameters
    threshold_2 = np.log10(1e10)
    steepness_2 = 2.5
    accuracy_cot = 90 / (1 + np.exp(-steepness_2 * (np.log10(model_sizes) - threshold_2)))

    ax2.semilogx(model_sizes, accuracy_cot, color=COLOR_SECONDARY, linewidth=3.4)
    ax2.axvline(1e10, color=COLOR_GRAY, linestyle='--', linewidth=3.0, alpha=0.6)
    ax2.text(1.4e10, 48, '~10B params', va='center', fontsize=10,
             color=COLOR_GRAY,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       alpha=0.85, edgecolor='none'))

    ax2.set_xlabel('Model Size (parameters, log scale)')
    ax2.set_ylabel('Accuracy (\\%)')
    ax2.set_title('(b) Chain-of-Thought Reasoning', fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax2.set_ylim([0, 102])
    ax2.axvspan(5e9, 2e10, alpha=0.35, color=COLOR_BEIGE)
    ax2.text(8e9, 85, 'Emergence region', fontsize=10, color=COLOR_SECONDARY,
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.85, edgecolor='none'))

    # Panel (c): Factual recall capacity
    ax3 = axes[2]

    # Linear in log space (power law)
    facts_recalled = model_sizes / 100  # Rough approximation: 1 fact per 100 params

    ax3.loglog(model_sizes, facts_recalled, color=COLOR_TERTIARY, linewidth=3.4)

    # Mark thresholds
    ax3.axhline(1e6, color=COLOR_GRAY, linestyle='--', linewidth=2, alpha=0.6)
    ax3.text(5e10, 1.4e6, '1M facts', fontsize=11, color=COLOR_GRAY,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.85, edgecolor='none'))

    ax3.axhline(1e7, color=COLOR_GRAY, linestyle='--', linewidth=2, alpha=0.6)
    ax3.text(5e10, 1.35e7, '10M facts', fontsize=11, color=COLOR_GRAY,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.85, edgecolor='none'))

    ax3.set_xlabel('Model Size (parameters, log scale)')
    ax3.set_ylabel('Facts Recalled')
    ax3.set_title('(c) Factual Recall Capacity', fontsize=14)
    ax3.grid(True, alpha=0.28, which='both', linestyle='--', linewidth=0.6)
    ax3.set_xlim([9e7, 2.5e11])
    ax3.set_ylim([5e5, 2e9])

    def _facts_formatter(value, _):
        if value >= 1e9:
            return f'{value/1e9:.0f}B'
        if value >= 1e6:
            return f'{value/1e6:.0f}M'
        return f'{value:,.0f}'

    ax3.yaxis.set_major_formatter(FuncFormatter(_facts_formatter))

    # Ensure identical x-axis ranges for comparability
    for ax in axes:
        ax.set_xlim([1e8, 1e11])
    plt.tight_layout()
    _standard_layout(fig, wspace=0.35)
    plt.savefig('figures/ch12_uncertainty/fig_12_4_emergent_abilities.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_4_emergent_abilities.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.4 saved")
    plt.close()

# ============================================================================
# FIGURE 12.5: Learning Rate Schedule
# ============================================================================

def figure_12_5():
    """Show three-phase learning rate schedule"""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.2, FIGURE_HEIGHT))

    # Parameters
    total_steps = 100000
    warmup_steps = 2000
    decay_start = 80000
    peak_lr = 3e-4

    steps = np.arange(0, total_steps + 1)
    lr = np.zeros_like(steps, dtype=float)

    # Warmup phase
    warmup_mask = steps <= warmup_steps
    lr[warmup_mask] = peak_lr * (steps[warmup_mask] / warmup_steps)

    # Stable phase
    stable_mask = (steps > warmup_steps) & (steps <= decay_start)
    lr[stable_mask] = peak_lr

    # Cosine decay phase
    decay_mask = steps > decay_start
    decay_steps = steps[decay_mask] - decay_start
    total_decay = total_steps - decay_start
    lr[decay_mask] = peak_lr * 0.5 * (1 + np.cos(np.pi * decay_steps / total_decay))

    # Plot
    line_lr, = ax.plot(steps, lr, color=COLOR_PRIMARY, linewidth=3.0)
    line_lr.set_path_effects([
        pe.SimpleLineShadow(offset=(1.2, -1.2), alpha=0.3),
        pe.Normal()
    ])

    # Shade phases
    ax.axvspan(0, warmup_steps, alpha=0.22, color=COLOR_TERTIARY, label='Warmup')
    ax.axvspan(warmup_steps, decay_start, alpha=0.16, color=COLOR_PRIMARY, label='Stable')
    ax.axvspan(decay_start, total_steps, alpha=0.22, color=COLOR_SECONDARY, label='Decay')

    # Vertical dashed lines at phase boundaries
    ax.axvline(warmup_steps, color=COLOR_TERTIARY, linestyle='--', linewidth=1.8, alpha=0.8)
    ax.axvline(decay_start, color=COLOR_SECONDARY, linestyle='--', linewidth=1.8, alpha=0.8)

    # Annotate phases (repositioned to avoid overlap)
    label_box = dict(boxstyle='round,pad=0.35', facecolor='white',
                     linewidth=1.6, alpha=0.92)
    ax.text(warmup_steps/2, peak_lr * 0.45, 'Linear\\\\warmup', ha='center', fontsize=10,
           bbox={**label_box, 'edgecolor': COLOR_TERTIARY})

    ax.text((warmup_steps + decay_start)/2, peak_lr * 1.12, 'Constant\\\\learning rate', ha='center', fontsize=10,
           bbox={**label_box, 'edgecolor': COLOR_PRIMARY})

    ax.text((decay_start + total_steps)/2, peak_lr * 0.5, 'Cosine\\\\decay', ha='center', fontsize=10,
           bbox={**label_box, 'edgecolor': COLOR_SECONDARY})

    # Mark key points
    ax.plot([warmup_steps], [peak_lr], 'o', color=COLOR_TERTIARY, markersize=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.6)
    ax.plot([decay_start], [peak_lr], 'o', color=COLOR_SECONDARY, markersize=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.6)

    ax.set_xlabel('Training Steps (0–100k)')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule: Warmup $\\rightarrow$ Stable $\\rightarrow$ Decay', fontsize=14)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95,
              facecolor='white', edgecolor=COLOR_GRAY)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, total_steps])
    ax.set_ylim([peak_lr * -0.02, peak_lr * 1.18])
    ax.set_yticks([0, 1e-4, 2e-4, 3e-4])

    # Format y-axis as scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    _standard_layout(fig, wspace=0.35)
    plt.savefig('figures/ch12_uncertainty/fig_12_5_learning_rate_schedule.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_5_learning_rate_schedule.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.5 saved")
    plt.close()

# ============================================================================
# FIGURE 12.6: Critical Batch Size Phenomenon
# ============================================================================

def figure_12_6():
    """Show how training efficiency plateaus beyond critical batch size"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.55, FIGURE_HEIGHT))

    # Data from table in text
    batch_sizes = np.array([256, 512, 1024, 2048, 4096, 8192])
    steps_to_target = np.array([50000, 26000, 14000, 8000, 6500, 6000])
    wall_clock_hours = np.array([10, 6, 4, 3.5, 3.8, 4.5])

    # Panel (a): Steps to target vs batch size
    ax1.semilogx(batch_sizes, steps_to_target, 'o-', color=COLOR_PRIMARY,
                 linewidth=2.5, markersize=8)

    # Mark critical batch size
    critical_bs = 2048
    ax1.axvline(critical_bs, color=COLOR_QUATERNARY, linestyle='-', linewidth=3.2,
                label='Critical batch size')

    # Shade regions
    ax1.axvspan(256, critical_bs, alpha=0.1, color=COLOR_TERTIARY,
               label='Efficient scaling region')
    ax1.axvspan(critical_bs, 8192, alpha=0.1, color=COLOR_QUATERNARY,
               label='Diminishing returns')

    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Steps to reach target loss', fontsize=12)
    ax1.set_title('(a) Convergence Speed', pad=12, fontsize=14)
    # Simplified legend
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95,
               facecolor='white', edgecolor=COLOR_GRAY)
    ax1.grid(True, alpha=0.3)

    # Add annotation away from data with leader line
    crit_idx = batch_sizes.tolist().index(critical_bs)
    ax1.annotate('Beyond critical batch size\nMore compute, no faster gains',
                 xy=(critical_bs, steps_to_target[crit_idx]),
                 xycoords='data',
                 xytext=(0.28, 0.8),
                 textcoords='axes fraction',
                 ha='center', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                           edgecolor=COLOR_QUATERNARY, linewidth=1.5, alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=2.0,
                                 connectionstyle='arc3,rad=-0.25'))

    # Panel (b): Wall-clock time vs batch size
    ax2.semilogx(batch_sizes, wall_clock_hours, 's-', color=COLOR_SECONDARY,
                 linewidth=2.7, markersize=10)

    # Mark optimal
    optimal_idx = np.argmin(wall_clock_hours)
    ax2.plot(batch_sizes[optimal_idx], wall_clock_hours[optimal_idx], 'o',
             color=COLOR_TERTIARY, markersize=16, zorder=5,
             markeredgecolor='white', markeredgewidth=1.8,
             label=f'Optimal: BS={batch_sizes[optimal_idx]}')
    ax2.annotate('Optimal batch size',
                 xy=(batch_sizes[optimal_idx], wall_clock_hours[optimal_idx]),
                 xycoords='data',
                 xytext=(0.25, 0.18), textcoords='axes fraction',
                 fontsize=9, color=COLOR_TERTIARY, ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.9, edgecolor=COLOR_TERTIARY, linewidth=0.8),
                 arrowprops=dict(arrowstyle='->', color=COLOR_TERTIARY, lw=1.8,
                                 connectionstyle='arc3,rad=0.2'))

    ax2.axvline(critical_bs, color=COLOR_QUATERNARY, linestyle='--', linewidth=3)

    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Training Time (hours)', fontsize=12)
    ax2.set_title('(b) Training Time', pad=12, fontsize=14)
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.95,
               facecolor='white', edgecolor=COLOR_GRAY, bbox_to_anchor=(0.02, 0.02))
    ax2.grid(True, alpha=0.3)

    # Add annotation with callout to high batch regime - repositioned to avoid overlap
    ax2.annotate('Too large:\ncommunication\noverhead dominates',
                 xy=(4096, wall_clock_hours[4]),
                 xycoords='data',
                 xytext=(0.62, 0.75),
                 textcoords='axes fraction',
                 ha='center', va='top', fontsize=8,
                 style='italic', color=COLOR_QUATERNARY,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.9, edgecolor=COLOR_QUATERNARY, linewidth=0.8),
                 arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=1.3,
                                 connectionstyle='arc3,rad=-0.2'))

    plt.tight_layout()
    _standard_layout(fig, wspace=0.4)
    plt.savefig('figures/ch12_uncertainty/fig_12_6_critical_batch_size.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_6_critical_batch_size.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.6 saved")
    plt.close()

# ============================================================================
# FIGURE 12.7A/B: Training Pathologies (Split Figures)
# ============================================================================

def _training_pathology_data():
    """Generate synthetic loss curves for the pathology visualizations."""
    np.random.seed(42)
    steps = np.arange(0, 5000)

    loss_explode = (3.0 + np.random.normal(0, 0.05, 2000)).tolist()
    loss_explode += (3.0 - 0.0005 * (steps[2000:2500] - 2000) + np.random.normal(0, 0.05, 500)).tolist()
    loss_explode += (3.0 + 0.1 * (steps[2500:] - 2500) + np.random.normal(0, 0.2, len(steps) - 2500)).tolist()
    loss_explode = np.array(loss_explode)

    loss_plateau = 3.0 - 0.0003 * steps[:1000] + np.random.normal(0, 0.02, 1000)
    loss_plateau = np.concatenate([loss_plateau, 2.7 + np.random.normal(0, 0.02, 4000)])

    trend = 3.0 - 0.0003 * steps
    oscillation = 0.3 * np.sin(steps / 200)
    loss_oscillate = trend + oscillation + np.random.normal(0, 0.05, len(steps))

    loss_vanish = 3.0 - 0.0005 * steps[:1000] + np.random.normal(0, 0.02, 1000)
    loss_vanish = np.concatenate([loss_vanish, 2.5 + np.random.normal(0, 0.01, 4000)])

    loss_nolearn = 3.0 + np.random.normal(0, 0.03, len(steps))

    loss_healthy = 1.5 * np.exp(-steps / 2000) + 1.5 + np.random.normal(0, 0.02, len(steps))

    return steps, {
        'loss_explode': loss_explode,
        'loss_plateau': loss_plateau,
        'loss_oscillate': loss_oscillate,
        'trend': trend,
        'loss_vanish': loss_vanish,
        'loss_nolearn': loss_nolearn,
        'loss_healthy': loss_healthy,
    }


def figure_12_7a():
    """Training instabilities: exploding loss, early plateau, oscillations."""
    steps, curves = _training_pathology_data()

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.8, FIGURE_HEIGHT*1.25))

    # (a) Loss explodes
    ax1 = axes[0]
    ax1.plot(steps, curves['loss_explode'], color=COLOR_QUATERNARY, linewidth=2.6)
    ax1.axvline(2500, color='black', linestyle='--', alpha=0.6, linewidth=2.4)
    # Label the vertical event line
    ax1.text(2500, 19, 'LR increase', fontsize=10, ha='center', va='top', color='black')
    ax1.annotate('Explosion after LR increase',
                 xy=(2500, curves['loss_explode'][2500]),
                 xycoords='data',
                 xytext=(0.72, 0.84),
                 textcoords='axes fraction',
                 fontsize=11, color=COLOR_QUATERNARY, weight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.9, edgecolor=COLOR_QUATERNARY, linewidth=0.6),
                 arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=2.2,
                                 connectionstyle='arc3,rad=-0.15'))
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Loss Explodes', fontsize=12, pad=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 20])
    ax1.text(0.02, 0.88, 'LR too high', transform=ax1.transAxes, fontsize=10.5,
             color=COLOR_QUATERNARY, style='italic')

    # (b) Early plateau
    ax2 = axes[1]
    ax2.plot(steps, curves['loss_plateau'], color=COLOR_SECONDARY, linewidth=2.4)
    ax2.axhline(2.7, color='black', linestyle='--', alpha=0.7, linewidth=3)
    ax2.axhspan(2.68, 2.78, color=COLOR_SECONDARY, alpha=0.08)
    ax2.set_ylabel('Loss')
    ax2.set_title('(b) Early Plateau', fontsize=12, pad=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1.4, 3.2])
    ax2.text(0.05, 0.98, 'LR too low or\nmodel capacity limited', transform=ax2.transAxes,
             fontsize=10.5, color=COLOR_SECONDARY, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.85, edgecolor='none'))
    ax2.text(0.5, 0.04, 'Stuck at high loss after brief improvement',
             transform=ax2.transAxes, fontsize=9.5, color=COLOR_SECONDARY,
             ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='none'))

    # (c) Oscillations
    ax3 = axes[2]
    ax3.plot(steps, curves['loss_oscillate'], color=COLOR_PRIMARY, linewidth=2.6, alpha=0.35)
    ax3.plot(steps, curves['trend'], color='black', linestyle='--', linewidth=3.6, label='Trend')
    ax3.set_ylabel('Loss')
    ax3.set_title('(c) Oscillations', fontsize=12, pad=12)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([1.4, 3.2])
    ax3.text(0.02, 0.85, 'Batch size too small', transform=ax3.transAxes,
             fontsize=10, color=COLOR_PRIMARY, style='italic')

    plt.tight_layout()
    _standard_layout(fig, wspace=0.35)
    plt.savefig('figures/ch12_uncertainty/fig_12_7a_training_instabilities.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_7a_training_instabilities.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.7a saved")
    plt.close()


def figure_12_7b():
    """Learning failures and healthy baseline."""
    steps, curves = _training_pathology_data()

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.8, FIGURE_HEIGHT*1.25))

    # (d) Gradient vanishing
    ax4 = axes[0]
    ax4.plot(steps, curves['loss_vanish'], color=COLOR_PURPLE, linewidth=2.6)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Loss')
    ax4.set_title('(d) Gradient Vanishing', fontsize=12, pad=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([1.4, 3.2])
    ax4.text(0.02, 0.88, 'Poor initialization', transform=ax4.transAxes,
             fontsize=11, style='italic', color=COLOR_PURPLE)
    ax4.annotate('Gradients vanish here',
                 xy=(3600, curves['loss_vanish'][3600]),
                 xycoords='data',
                 xytext=(0.78, 0.8), textcoords='axes fraction',
                 fontsize=11, color=COLOR_PURPLE, weight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                           alpha=0.9, edgecolor=COLOR_PURPLE, linewidth=0.9),
                 arrowprops=dict(arrowstyle='->', color=COLOR_PURPLE, lw=2.0,
                                 connectionstyle='arc3,rad=-0.35'))

    # (e) No learning
    ax5 = axes[1]
    ax5.plot(steps, curves['loss_nolearn'], color=COLOR_GRAY, linewidth=2, alpha=0.3)
    ax5.axhline(3.0, color='black', linestyle='--', linewidth=2.2, alpha=0.6)
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Loss')
    ax5.set_title('')  # Clear any prior title artifacts
    ax5.set_title('(e) No Learning', fontsize=12, pad=12)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([1.4, 3.2])
    ax5.text(0.5, 0.92, 'Bug in data or optimizer', transform=ax5.transAxes,
             fontsize=9.5, color=COLOR_GRAY, ha='center', va='top', style='italic',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.85, edgecolor='none'))

    # (f) Healthy training
    ax6 = axes[2]
    ax6.plot(steps, curves['loss_healthy'], color=COLOR_TERTIARY, linewidth=2.6)
    # Subtle milestone markers every 1000 steps
    ax6.plot(steps[::1000], curves['loss_healthy'][::1000], 'o', color=COLOR_TERTIARY,
             markersize=3.5, alpha=0.8)
    ax6.set_xlabel('Training Steps')
    ax6.set_ylabel('Loss')
    ax6.set_title('(f) Healthy Training Dynamics', fontsize=12, pad=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([1.4, 3.2])
    ax6.text(0.65, 0.25, 'Steady improvement', transform=ax6.transAxes, fontsize=10,
             color=COLOR_TERTIARY, weight='bold',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.85, edgecolor='none'))
    ax6.margins(x=0.02)

    plt.tight_layout()
    _standard_layout(fig, wspace=0.35)
    plt.savefig('figures/ch12_uncertainty/fig_12_7b_learning_failures.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_7b_learning_failures.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.7b saved")
    plt.close()

# ============================================================================
# FIGURE 12.8: Distributed Training Paradigms
# ============================================================================

def figure_12_8():
    """Visual comparison of data/model/pipeline parallelism"""
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.8, FIGURE_HEIGHT*1.15))

    # Panel (a): Data Parallelism
    ax1 = axes[0]
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 11])
    ax1.axis('off')

    # Draw 3 GPUs with more horizontal spacing
    gpu_positions = [(1.2, 7.5), (4, 7.5), (6.8, 7.5)]
    for i, (x, y) in enumerate(gpu_positions):
        # GPU box
        gpu = FancyBboxPatch((x-0.7, y-1.0), 1.4, 2.0, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=COLOR_PRIMARY,
                              linewidth=2.5, alpha=0.45)
        ax1.add_patch(gpu)
        gpu.set_path_effects([pe.SimplePatchShadow(offset=(0.5, -0.5), alpha=0.3),
                              pe.Normal()])
        ax1.text(x, y+0.6, f'GPU {i+1}', ha='center', fontsize=10, weight='bold', color='black')
        ax1.text(x, y+0.1, 'Full', ha='center', fontsize=9, color='black')
        ax1.text(x, y-0.3, 'Model', ha='center', fontsize=9, color='black')

    # Synchronization box
    sync_box = FancyBboxPatch((3.5, 3.5), 3.0, 1.5, boxstyle="round,pad=0.12",
                              edgecolor='black', facecolor=COLOR_SECONDARY,
                              linewidth=2.5, alpha=0.4)
    ax1.add_patch(sync_box)
    sync_box.set_path_effects([pe.SimplePatchShadow(offset=(0.5, -0.5), alpha=0.3),
                               pe.Normal()])
    ax1.text(5, 4.5, 'All-Reduce', ha='center', fontsize=10, weight='bold', color='black')
    ax1.text(5, 3.9, '(gradient sync)', ha='center', fontsize=8, style='italic', color='black')

    # Longer arrows showing data flow
    for x, y in gpu_positions:
        ax1.annotate('', xy=(5, 4.9), xytext=(x, y-1.1),
                     arrowprops=dict(arrowstyle='->', lw=2.0, color='black'))

    ax1.text(5, 10.5, '(a) Data Parallelism', ha='center', fontsize=12, fontweight='bold')
    ax1.text(5, 9.8, 'Best for: batch-size scaling', ha='center', fontsize=8.5, style='italic')
    ax1.text(5, 9.3, 'Each GPU: full model, different data', ha='center', fontsize=8,
             style='italic', fontweight='bold')

    # Panel (b): Model Parallelism
    ax2 = axes[1]
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 11])
    ax2.axis('off')

    # Draw 4 GPUs with model partitions - increased vertical spacing
    layers = ['L1-3', 'L4-6', 'L7-9', 'L10-12']
    colors = [COLOR_TERTIARY, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_PURPLE]

    for i, (layer, color) in enumerate(zip(layers, colors)):
        y = 8.5 - i*2.0  # Increased spacing from 1.8 to 2.0
        gpu = FancyBboxPatch((2.5, y-0.5), 5.0, 1.0, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color,
                              linewidth=2.5, alpha=0.45)
        ax2.add_patch(gpu)
        gpu.set_path_effects([pe.SimplePatchShadow(offset=(0.5, -0.5), alpha=0.3),
                              pe.Normal()])
        ax2.text(5, y, f'GPU {i+1}: {layer}', ha='center', va='center',
                 fontsize=10, weight='bold', color='black')

        # Longer arrows between GPUs (activations) - positioned OUTSIDE boxes
        if i < 3:
            arrow_start_y = y - 0.6
            arrow_end_y = arrow_start_y - 1.3  # Longer arrow
            ax2.annotate('', xy=(5, arrow_end_y), xytext=(5, arrow_start_y),
                        arrowprops=dict(arrowstyle='->', lw=2.2, color='black'))
            ax2.text(7.0, (arrow_start_y + arrow_end_y) / 2, 'activations', fontsize=7.5,
                    style='italic', color='black', ha='left')

    ax2.text(5, 10.5, '(b) Model Parallelism', ha='center', fontsize=12, fontweight='bold')
    ax2.text(5, 9.8, 'Best for: very large models', ha='center', fontsize=8.5, style='italic')
    ax2.text(5, 9.3, 'Each GPU: model partition, same data', ha='center', fontsize=8,
             style='italic', fontweight='bold')

    # Panel (c): Pipeline Parallelism
    ax3 = axes[2]
    ax3.set_xlim([0, 10])
    ax3.set_ylim([0, 11])
    ax3.axis('off')

    # Draw pipeline stages with increased spacing
    for i in range(4):
        y = 8.5 - i*2.0  # Increased spacing
        gpu = FancyBboxPatch((1.5, y-0.5), 6.5, 1.0, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=COLOR_PRIMARY,
                              linewidth=2.5, alpha=0.45)
        ax3.add_patch(gpu)
        gpu.set_path_effects([pe.SimplePatchShadow(offset=(0.5, -0.5), alpha=0.3),
                              pe.Normal()])
        ax3.text(2.3, y, f'GPU {i+1}', ha='center', va='center',
                 fontsize=10, weight='bold', color='black')

        # Show micro-batches in pipeline (smaller, better positioned)
        for j in range(3):
            x_offset = 3.5 + j*1.3
            if i + j < 4:
                batch_box = Rectangle((x_offset, y-0.45), 1.1, 0.9,
                                      facecolor=COLOR_SECONDARY, edgecolor='black',
                                      linewidth=1.5, alpha=0.7)
                ax3.add_patch(batch_box)
                batch_box.set_path_effects([pe.SimplePatchShadow(offset=(0.3, -0.3), alpha=0.3),
                                            pe.Normal()])
                ax3.text(x_offset+0.55, y, r'$\mu$B{0}'.format(j+1), ha='center', va='center',
                        fontsize=9, weight='bold', color='white')

    ax3.text(5, 10.5, '(c) Pipeline Parallelism', ha='center', fontsize=12, fontweight='bold')
    ax3.text(5, 9.8, 'Best for: high throughput', ha='center', fontsize=8.5, style='italic')
    ax3.text(5, 9.3, 'Each GPU: model stage, pipelined micro-batches', ha='center', fontsize=8,
             style='italic', fontweight='bold')
    # Legend for micro-batch positioned safely at bottom
    ax3.text(1.5, 1.0, r'$\mu$B$:$ micro-batch', fontsize=8, color=COLOR_SECONDARY,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       alpha=0.92, edgecolor=COLOR_GRAY, linewidth=0.8))
    # Longer forward arrow positioned outside boxes
    ax3.annotate('forward', xy=(8.5, 8.8), xytext=(8.5, 1.8),
                 arrowprops=dict(arrowstyle='->', lw=1.8, color='black'),
                 fontsize=8, ha='center')

    plt.tight_layout()
    _standard_layout(fig, wspace=0.35)
    plt.savefig('figures/ch12_uncertainty/fig_12_8_distributed_training.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_8_distributed_training.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.8 saved")
    plt.close()

# ============================================================================
# FIGURE 12.9: Training vs Inference Cost Over Time
# ============================================================================

def figure_12_9():
    """Show how inference costs dominate for high-volume applications"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Cumulative costs over time
    years = np.arange(0, 6)

    # Model A: 70B params
    training_cost_A = 500  # $500k one-time
    inference_cost_per_year_A = 25.5  # $25.5k/year
    cumulative_A = training_cost_A + inference_cost_per_year_A * years

    # Model B: 175B params
    training_cost_B = 500  # Same training cost
    inference_cost_per_year_B = 63.9  # $63.9k/year
    cumulative_B = training_cost_B + inference_cost_per_year_B * years

    ax1.plot(years, cumulative_A, 'o-', color=COLOR_TERTIARY, linewidth=3,
            markersize=10, label='Model A (70B params)')
    ax1.plot(years, cumulative_B, 's-', color=COLOR_QUATERNARY, linewidth=3,
            markersize=10, label='Model B (175B params)')

    # Shade training cost
    ax1.axhspan(0, training_cost_A, alpha=0.1, color=COLOR_GRAY)

    # Show savings
    savings_k = cumulative_B - cumulative_A
    ax1.fill_between(years, cumulative_A, cumulative_B,
                     alpha=0.18, color=COLOR_TERTIARY)
    ax1.text(0.55, 0.88, f'\\${savings_k[-1]:.0f}k saved over 5 years',
             transform=ax1.transAxes,
             fontsize=9, ha='center', weight='bold', color=COLOR_TERTIARY,
             bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                       edgecolor=COLOR_TERTIARY, linewidth=2, alpha=0.9))

    ax1.set_xlabel('Years in Production', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Cost (thousands of dollars)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Total Cost Over Time\n(1B queries/day)', pad=12)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95,
               facecolor='white', edgecolor=COLOR_GRAY)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax1.yaxis.set_major_locator(MultipleLocator(200))
    ax1.set_ylim(0, cumulative_B.max() * 1.05)
    ax1.set_xlim([0, 5])
    ax1.set_xticks([0,1,2,3,4,5])

    # Panel (b): Cost breakdown at year 5
    categories = ['Model A\n(70B)', 'Model B\n(175B)']
    training_costs = [training_cost_A, training_cost_B]
    inference_costs = [inference_cost_per_year_A * 5, inference_cost_per_year_B * 5]

    x_pos = np.arange(len(categories))
    width = 0.6

    bars1 = ax2.bar(x_pos, training_costs, width, label='Training (upfront cost)',
                   color=COLOR_GRAY, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x_pos, inference_costs, width, bottom=training_costs,
                   label='Inference (5 years)', color=COLOR_SECONDARY,
                   alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels (simplified to reduce clutter)
    for i, (train, infer) in enumerate(zip(training_costs, inference_costs)):
        total = train + infer
        # Total cost above bar
        ax2.text(i, total + 25, f'\\${total:.0f}k', ha='center', va='bottom',
                 fontsize=10.5, weight='bold',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                           alpha=0.92, edgecolor=COLOR_GRAY, linewidth=0.8))
        # Percentage label in middle of inference section
        infer_pct = 100 * infer / total
        ax2.text(i, train + infer/2, f'{infer_pct:.0f}\\% inference', ha='center',
                 fontsize=9.5, weight='bold', color='white',
                 path_effects=[pe.withStroke(linewidth=2.5, foreground=COLOR_SECONDARY, alpha=0.9)])

    ax2.set_ylabel('Total Cost (thousands of dollars)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Cost Breakdown After 5 Years', pad=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.set_xlim([-0.5, 1.5])
    # Add headroom to avoid top labels cramping
    max_total = max([t + i for t, i in zip(training_costs, inference_costs)])
    ax2.set_ylim(0, max_total * 1.22)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.95,
               facecolor='white', edgecolor=COLOR_GRAY)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.6)

    plt.tight_layout()
    _standard_layout(fig, wspace=0.4)
    plt.savefig('figures/ch12_uncertainty/fig_12_9_training_vs_inference_cost.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch12_uncertainty/fig_12_9_training_vs_inference_cost.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 12.9 saved")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Chapter 12 Figures: Scaling Laws and Training at Scale")
    print("=" * 60)
    print()

    figure_12_1()
    figure_12_2()
    figure_12_3()
    figure_12_4()
    figure_12_5()
    figure_12_6()
    figure_12_7a()
    figure_12_7b()
    figure_12_8()
    figure_12_9()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("Output location: figures/ch12_uncertainty/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 60)
    print()
    print("Figure Summary:")
    print("  12.1: Chinchilla vs GPT-3 Resource Allocation")
    print("  12.2: The Three Scaling Laws (Power Laws)")
    print("  12.3: Compute-Optimal Frontier")
    print("  12.4: Emergent Abilities and Phase Transitions")
    print("  12.5: Learning Rate Schedule (3 Phases)")
    print("  12.6: Critical Batch Size Phenomenon")
    print("  12.7a: Training Instabilities (Exploding/Plateau/Oscillations)")
    print("  12.7b: Learning Failures and Healthy Baseline")
    print("  12.8: Distributed Training Paradigms")
    print("  12.9: Training vs Inference Cost Analysis")
