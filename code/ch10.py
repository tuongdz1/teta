"""
Figure generation utilities for Chapter 10 (Calibration and Confidence).
Produces the figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rc, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import os

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# LaTeX font configuration
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
rc('axes', labelsize=11)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)

# Color palette
COLOR_PRIMARY = '#1f77b4'      # Blue
COLOR_SECONDARY = '#ff7f0e'    # Orange
COLOR_TERTIARY = '#2ca02c'     # Green
COLOR_QUATERNARY = '#d62728'   # Red
COLOR_GRAY = '#7f7f7f'
COLOR_LIGHT_BLUE = '#aec7e8'
COLOR_LIGHT_ORANGE = '#ffbb78'
COLOR_LIGHT_GREEN = '#98df8a'
COLOR_LIGHT_RED = '#ff9896'
COLOR_MUTED_PURPLE = '#9467bd'
COLOR_MUTED_TEAL = '#17becf'

# Figure dimensions
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0
DPI = 300

# Consistent annotation styling
ANNOTATION_BOX = dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      edgecolor=COLOR_GRAY,
                      linewidth=1.0,
                      alpha=0.85)

# Create output directory
os.makedirs('figures/ch10_calibration', exist_ok=True)

# ============================================================================
# FIGURE 10.1: Calibration Plots (Well-Calibrated vs Overconfident)
# ============================================================================

def figure_10_1():
    """Visualize calibration plots for well-calibrated and overconfident models."""
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH * 1.7, FIGURE_HEIGHT * 1.05)
    )

    # Panel 1: Well-calibrated model
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    predicted_conf = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Well-calibrated: actual accuracy ≈ predicted confidence
    actual_acc_good = predicted_conf + np.random.randn(n_bins) * 0.02
    actual_acc_good = np.clip(actual_acc_good, 0, 1)

    # ECE calculation
    ece_good = np.mean(np.abs(actual_acc_good - predicted_conf))

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax1.fill_between([0, 1], [0, 1], [0, 1], alpha=0.1, color=COLOR_TERTIARY,
                     transform=ax1.transData)

    # Add tolerance band
    tolerance = 0.05
    xband = np.linspace(0, 1, 100)
    ylow = np.clip(xband - tolerance, 0, 1)
    yhigh = np.clip(xband + tolerance, 0, 1)
    tol_poly = ax1.fill_between(
        xband,
        ylow,
        yhigh,
        alpha=0.22,
        facecolor=COLOR_TERTIARY,
        edgecolor=COLOR_TERTIARY,
        linewidth=1.2,
        label='$\\pm 5\\%$ tolerance',
    )

    ax1.plot(predicted_conf, actual_acc_good, 'o-', color=COLOR_PRIMARY,
            linewidth=2.5, markersize=10, markeredgecolor='black',
            markeredgewidth=1.5, label='Model predictions')

    ax1.set_xlabel('Predicted Confidence', fontsize=10)
    ax1.set_ylabel('Actual Accuracy', fontsize=10)
    ax1.set_title(
        f'(a) Well-Calibrated Model\n$\\mathrm{{ECE}} = {ece_good:.2f}$',
        fontsize=11,
        pad=14,
    )
    # Legends inside these small axes were overlapping with data.
    # Move them above the axes and spread across columns for readability.
    ax1.legend(
        loc='lower right',
        ncol=1,
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        handlelength=2.0,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')

    # Panel 2: Overconfident model
    # Overconfident: predicted confidence higher than actual accuracy
    actual_acc_bad = 0.78 * predicted_conf + 0.10
    actual_acc_bad = np.clip(actual_acc_bad, 0, 1)

    ece_bad = np.mean(np.abs(actual_acc_bad - predicted_conf))

    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax2.plot(predicted_conf, actual_acc_bad, 'o-', color=COLOR_QUATERNARY,
            linewidth=2.5, markersize=10, markeredgecolor='black',
            markeredgewidth=1.5, label='Model predictions')

    # Highlight the gap
    overconf_point = (0.85, 0.78 * 0.85 + 0.10)
    ax2.annotate(
        '',
        xy=(overconf_point[0], overconf_point[1] + 0.05),
        xytext=overconf_point,
        arrowprops=dict(arrowstyle='<->', lw=2, color=COLOR_GRAY, alpha=0.9),
    )
    # Reposition annotation to guaranteed empty space (upper-left below legend)
    ax2.annotate(
        'Overconfident:\n85\\% predicted vs 78\\% actual',
        xy=overconf_point,
        xytext=(0.04, 0.75),
        textcoords='axes fraction',
        ha='left',
        va='top',
        fontsize=9.5,
        bbox=ANNOTATION_BOX,
        arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_GRAY),
    )

    ax2.set_xlabel('Predicted Confidence', fontsize=10)
    ax2.set_ylabel('Actual Accuracy', fontsize=10)
    ax2.set_title(
        f'(b) Overconfident Model\n$\\mathrm{{ECE}} = {ece_bad:.2f}$',
        fontsize=11,
        pad=14,
    )
    ax2.legend(
        loc='lower right',
        ncol=1,
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        handlelength=2.0,
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.32)
    plt.savefig('figures/ch10_calibration/fig_10_1_reliability_diagram.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_1_reliability_diagram.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.1 saved")
    plt.close()

# ============================================================================
# FIGURE 10.2: Temperature Scaling Effects
# ============================================================================

def figure_10_2():
    """Show how temperature affects probability distributions."""
    np.random.seed(42)

    fig, axes = plt.subplots(
        1, 3, figsize=(FIGURE_WIDTH * 2.1, FIGURE_HEIGHT * 0.95)
    )

    # Original logits
    logits = np.array([2.6, 2.0, 1.6, 1.2, 0.9, 0.6, 0.3, 0.15, 0.1, 0.05, 0.02, 0.01])
    classes = np.arange(1, len(logits) + 1)

    temperatures = [0.5, 1.0, 2.0]
    titles = [
        ('(a)', '$T = 0.5$', 'Sharpened (overconfident)'),
        ('(b)', '$T = 1.0$', 'Original distribution'),
        ('(c)', '$T = 2.0$', 'Softened (more uncertain)'),
    ]

    for ax, T, (panel_label, temp_label, descriptor) in zip(axes, temperatures, titles):
        # Apply temperature scaling
        scaled_logits = logits / T
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Plot
        bars = ax.bar(classes, probs, color=COLOR_SECONDARY, alpha=0.7,
                     edgecolor='black', linewidth=1)

        # Highlight top class
        bars[0].set_color(COLOR_SECONDARY)
        bars[0].set_alpha(1.0)

        # Color others
        for i in range(1, len(bars)):
            bars[i].set_color(COLOR_PRIMARY)
            bars[i].set_alpha(0.6)

        # Use a shared x-label; remove per-axis label to reduce redundancy
        ax.set_xlabel('', fontsize=10)
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_title(
            f'{panel_label} {temp_label}\n{descriptor}',
            fontsize=12,
            fontweight='bold',
            pad=14,
        )
        # Emphasize entropy value inside each subplot (top-right to avoid tall left bar)
        ax.text(
            0.96, 0.95,
            f'$\\mathbf{{H = {entropy:.2f}\\;nats}}$',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=11,
            bbox=ANNOTATION_BOX,
        )
        ax.set_ylim([0, 0.7])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add note about rank preservation
    fig.text(0.5, -0.02, 'Rank order preserved across all temperatures',
             ha='center', fontsize=11, style='italic', color=COLOR_GRAY)
    fig.text(0.5, -0.065, 'Class', ha='center', fontsize=11)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, wspace=0.32)
    plt.savefig('figures/ch10_calibration/fig_10_2_temperature_scaling.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_2_temperature_scaling.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.2 saved")
    plt.close()

# ============================================================================
# FIGURE 10.3: Decoding Strategies
# ============================================================================

def figure_10_3():
    """Compare different decoding strategies."""
    np.random.seed(42)

    fig, axes = plt.subplots(
        2, 2, figsize=(FIGURE_WIDTH * 1.9, FIGURE_HEIGHT * 1.8)
    )

    # Mock vocabulary probabilities (more balanced for readability)
    vocab = [
        'Paris',
        'France',
        'Lyon',
        'Marseille',
        'Bordeaux',
        'Nice',
        'Toulouse',
        'Lille',
        'Other-1',
        'Other-2',
    ]
    probs = np.array([0.42, 0.18, 0.12, 0.08, 0.06, 0.04, 0.035, 0.025, 0.015, 0.015])
    probs = probs / probs.sum()
    logits = np.log(probs + 1e-10)
    bar_positions = np.arange(len(vocab))

    background_palette = ['#f1f6f4', '#f2f4fa', '#f8f5ef', '#f5f2f9']

    # (a) Greedy: Always highest probability (deterministic)
    ax = axes[0, 0]
    ax.set_facecolor(background_palette[0])
    bars = ax.bar(
        bar_positions,
        probs,
        color=COLOR_TERTIARY,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )
    for i in range(1, len(bars)):
        bars[i].set_color(COLOR_GRAY)
        bars[i].set_alpha(0.55)

    ax.set_ylabel('Token Probability', fontsize=11)
    ax.set_title(
        '(a) Greedy Decoding\n$k = 1$, deterministic argmax',
        fontsize=12,
        fontweight='bold',
        pad=10,
    )
    ax.set_ylim([0, 0.5])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(vocab, rotation=30, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.text(
        0.05,
        0.95,
        'All mass on top token.\nHigh repetition risk.',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10.5,
        bbox=ANNOTATION_BOX,
    )

    # (b) Temperature sampling
    ax = axes[0, 1]
    ax.set_facecolor(background_palette[1])
    temperature = 0.9
    probs_temp = np.exp(logits / temperature)
    probs_temp = probs_temp / probs_temp.sum()
    bars = ax.bar(
        bar_positions,
        probs_temp,
        color=COLOR_MUTED_TEAL,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )
    bars[0].set_alpha(0.95)

    ax.set_ylabel('Token Probability', fontsize=11)
    ax.set_title(
        f'(b) Temperature Sampling\n$T = {temperature}$ balances exploration',
        fontsize=12,
        fontweight='bold',
        pad=10,
    )
    ax.set_ylim([0, 0.5])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(vocab, rotation=30, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.text(
        0.62,
        0.92,
        f'Entropy: {(-np.sum(probs_temp * np.log(probs_temp + 1e-10))):.2f} nats',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10.5,
        bbox=ANNOTATION_BOX,
    )

    # (c) Top-k (k=5): Fixed cutoff
    ax = axes[1, 0]
    ax.set_facecolor(background_palette[2])
    k = 5
    probs_topk = probs.copy()
    probs_topk[k:] = 0
    probs_topk = probs_topk / probs_topk.sum()

    bars = ax.bar(
        bar_positions,
        probs_topk,
        color=COLOR_PRIMARY,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )
    for i in range(k):
        bars[i].set_alpha(0.95)
    for i in range(k, len(bars)):
        bars[i].set_alpha(0.35)

    ax.set_xlabel('Token', fontsize=11)
    ax.set_ylabel('Token Probability', fontsize=11)
    ax.set_title(
        f'(c) Top-$k$ Sampling\n$k = {k}$ keeps highest-probability tokens',
        fontsize=12,
        fontweight='bold',
        pad=10,
    )
    ax.set_ylim([0, 0.5])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(vocab, rotation=30, ha='right', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.text(
        0.64,
        0.92,
        (
            lambda pct: f'$\\mathbf{{Covers\\ {pct}\\ of\\ mass}}$' + '\nCut tail entirely.'
        )(f"{np.sum(probs[:k]):.0%}".replace('%', '\\%')),
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10.5,
        bbox=ANNOTATION_BOX,
    )

    # (d) Nucleus (p=0.90): Adaptive cutoff
    ax = axes[1, 1]
    ax.set_facecolor(background_palette[3])
    p_threshold = 0.90
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    nucleus_size = np.where(cumsum >= p_threshold)[0][0] + 1

    probs_nucleus = np.zeros_like(probs)
    nucleus_indices = sorted_indices[:nucleus_size]
    probs_nucleus[nucleus_indices] = probs[nucleus_indices]
    probs_nucleus = probs_nucleus / probs_nucleus.sum()

    bars = ax.bar(
        bar_positions,
        probs_nucleus,
        color=COLOR_MUTED_PURPLE,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )
    for idx in nucleus_indices:
        bars[idx].set_alpha(0.95)
    for idx in range(len(bars)):
        if idx not in nucleus_indices:
            bars[idx].set_alpha(0.3)

    ax.set_xlabel('Token', fontsize=11)
    ax.set_ylabel('Token Probability', fontsize=11)
    ax.set_title(
        f'(d) Nucleus Sampling\n$p = {p_threshold}$ adapts to support',
        fontsize=12,
        fontweight='bold',
        pad=10,
    )
    ax.set_ylim([0, 0.5])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(vocab, rotation=30, ha='right', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.text(
        0.62,
        0.92,
        (
            lambda tm: f'Nucleus size: {nucleus_size} tokens\nTail mass: {tm}'
        )((f"{(1 - np.sum(probs[nucleus_indices])):.0%}").replace('%', '\\%')),
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=10.5,
        bbox=ANNOTATION_BOX,
    )

    fig.tight_layout()
    plt.savefig('figures/ch10_calibration/fig_10_3_decoding_strategies.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_3_decoding_strategies.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.3 saved")
    plt.close()

# ============================================================================
# FIGURE 10.4: Diversity-Error Tradeoff
# ============================================================================

def figure_10_4():
    """Visualize the diversity-error tradeoff with Pareto frontier."""
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.65, FIGURE_HEIGHT * 1.1))

    # Generate data points for different temperatures
    temperatures = np.linspace(0.1, 3.2, 15)

    # Diversity (entropy) increases with temperature
    diversity = 1.2 + 0.7 * temperatures + 0.05 * np.random.randn(len(temperatures))

    # Error rate: U-shaped - low at moderate temperatures
    error_base = 8 + 15 * (temperatures - 1.5)**2
    error = error_base + 2 * np.random.randn(len(temperatures))

    # Create scatter plot with color mapping
    scatter = ax.scatter(
        diversity,
        error,
        c=temperatures,
        cmap='coolwarm',
        s=140,
        alpha=0.85,
        edgecolors='black',
        linewidth=1.0,
        zorder=3,
    )

    # Add Pareto frontier
    pareto_indices = []
    for i in range(len(diversity)):
        is_pareto = True
        for j in range(len(diversity)):
            if i != j:
                if diversity[j] >= diversity[i] and error[j] <= error[i]:
                    if diversity[j] > diversity[i] or error[j] < error[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)

    pareto_indices = sorted(pareto_indices, key=lambda i: diversity[i])
    pareto_line, = ax.plot(
        diversity[pareto_indices],
        error[pareto_indices],
        color='black',
        linewidth=2.5,
        linestyle='-',
        label='Pareto frontier',
        zorder=4,
        alpha=0.85,
    )
    pareto_line.set_path_effects(
        [patheffects.Stroke(linewidth=4.0, foreground='white'), patheffects.Normal()]
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature $T$', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Shade regions
    shaded_box = {**ANNOTATION_BOX, 'alpha': 0.95}

    factual_band = ax.axhspan(
        0,
        15,
        alpha=0.12,
        facecolor=COLOR_LIGHT_GREEN,
        label='Acceptable for factual tasks',
        edgecolor=COLOR_TERTIARY,
        linewidth=1.2,
    )
    creative_band = ax.axhspan(
        15,
        25,
        alpha=0.12,
        facecolor=COLOR_LIGHT_ORANGE,
        label='Acceptable for creative tasks',
        edgecolor=COLOR_SECONDARY,
        linewidth=1.2,
    )
    error_band = ax.axhspan(
        25,
        45,
        alpha=0.12,
        facecolor=COLOR_LIGHT_RED,
        label='Too many errors',
        edgecolor=COLOR_QUATERNARY,
        linewidth=1.2,
    )

    # Add annotations
    # Low diversity, low error (code generation)
    idx_low = np.argmin(diversity)
    ax.annotate(
        'Low diversity\nlow error\n(code gen)',
        xy=(diversity[idx_low], error[idx_low]),
        xytext=(0.15, 0.46),
        textcoords='axes fraction',
        fontsize=9,
        ha='center',
        bbox=shaded_box,
        arrowprops=dict(arrowstyle='-|>', lw=1.8, color=COLOR_GRAY),
    )

    # Balanced (chatbot)
    idx_mid = np.argmin(np.abs(temperatures - 1.0))
    ax.annotate(
        'Balanced\n(chatbot)',
        xy=(diversity[idx_mid], error[idx_mid]),
        xytext=(0.62, 0.24),
        textcoords='axes fraction',
        fontsize=9.5,
        fontweight='bold',
        ha='center',
        bbox=shaded_box,
        arrowprops=dict(
            arrowstyle='-|>',
            lw=2.2,
            color=COLOR_PRIMARY,
            connectionstyle='arc3,rad=-0.15',
        ),
    )
    # Emphasize balanced region with a subtle highlight ring
    from matplotlib.patches import Circle as _Circle
    ax.add_patch(
        _Circle((diversity[idx_mid], error[idx_mid]), 0.09,
                fill=False, linewidth=2.0, edgecolor=COLOR_PRIMARY, alpha=0.6, zorder=5)
    )

    # High diversity, high error (creative)
    idx_high = np.argmax(diversity)
    ax.annotate(
        'High diversity\nhigh error\n(creative)',
        xy=(diversity[idx_high], error[idx_high]),
        xytext=(0.83, 0.82),
        textcoords='axes fraction',
        fontsize=9,
        ha='center',
        bbox=shaded_box,
        arrowprops=dict(arrowstyle='-|>', lw=1.8, color=COLOR_GRAY),
    )

    # Arrow showing temperature increase with clear endpoints
    start_idx, end_idx = 0, -1
    start_xy = (diversity[start_idx], error[start_idx])
    end_xy = (diversity[end_idx], error[end_idx])
    # Single curved arrow from low to high temperature
    ax.annotate(
        '',
        xy=end_xy,
        xytext=start_xy,
        arrowprops=dict(
            arrowstyle='-|>',
            lw=2.5,
            color=COLOR_QUATERNARY,
            connectionstyle='arc3,rad=0.3',
        ),
    )
    # Label near the arrow in empty space (moved down to avoid data points)
    ax.text(
        2.0, 28.5,
        f'Increasing $T$\n({temperatures[start_idx]:.1f} $\\rightarrow$ {temperatures[end_idx]:.1f})',
        fontsize=9.5,
        ha='center',
        va='center',
        bbox=shaded_box,
        color=COLOR_QUATERNARY,
        fontweight='bold',
    )

    ax.set_xlabel('Diversity (entropy of outputs)', fontsize=10)
    ax.set_ylabel('Error Rate (\\%)', fontsize=10)
    ax.set_title('Diversity-Error Tradeoff: The Fundamental Constraint', fontsize=11, fontweight='bold')
    # Combined legend in lower right (cleaner)
    ax.legend(
        handles=[pareto_line, factual_band, creative_band, error_band],
        loc='lower right',
        frameon=True,
        framealpha=0.92,
        edgecolor=COLOR_GRAY,
        fontsize=8.5,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.9, 3.3])
    ax.set_ylim([0, 45])

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig('figures/ch10_calibration/fig_10_4_diversity_error_tradeoff.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_4_diversity_error_tradeoff.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.4 saved")
    plt.close()

# ============================================================================
# FIGURE 10.5: Before and After Calibration
# ============================================================================

def figure_10_5():
    """Show calibration improvement with temperature scaling."""
    np.random.seed(42)

    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.7, FIGURE_HEIGHT * 1.55))

    # Create grid for subplots
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.2, 1.5],
        width_ratios=[1, 1],
        hspace=0.42,
        wspace=0.4,
    )

    # Panel (a): Before calibration
    ax1 = fig.add_subplot(gs[0, 0])

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    predicted_conf = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Overconfident before calibration
    actual_acc_before = 0.7 * predicted_conf + 0.15 + np.random.randn(n_bins) * 0.02
    actual_acc_before = np.clip(actual_acc_before, 0, 1)

    ece_before = np.mean(np.abs(actual_acc_before - predicted_conf))

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration',
            zorder=1)
    conf_band = 0.035
    lower = np.clip(actual_acc_before - conf_band, 0, 1)
    upper = np.clip(actual_acc_before + conf_band, 0, 1)
    band1 = ax1.fill_between(
        predicted_conf,
        lower,
        upper,
        color=COLOR_LIGHT_RED,
        alpha=0.45,
        edgecolor=COLOR_QUATERNARY,
        linewidth=1.0,
        zorder=1,
        label='$\\pm$ calibration error band',
    )
    ax1.plot(predicted_conf, actual_acc_before, 'o-', color=COLOR_QUATERNARY,
            linewidth=2.6, markersize=9, markeredgecolor='black',
            markeredgewidth=1.2, label='Model predictions', zorder=3)

    ax1.set_xlabel('Predicted Confidence', fontsize=10)
    ax1.set_ylabel('Actual Accuracy', fontsize=10)
    ax1.set_title(
        f'Before Calibration ($T = 1.0$)\n$\\textbf{{{{ECE}}}} = \\mathbf{{{{{ece_before:.3f}}}}}$',
        fontsize=12,
        pad=14,
    )
    ax1.legend(loc='upper left', fontsize=9)
    ax1.text(
        0.98, 0.08, 'Top-1 accuracy unchanged', transform=ax1.transAxes,
        ha='right', va='bottom', fontsize=9, bbox=ANNOTATION_BOX,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.text(
        -0.18,
        1.05,
        '(a)',
        transform=ax1.transAxes,
        fontsize=12,
        fontweight='bold',
        va='bottom',
    )

    # Histogram below (a)
    ax1_hist = fig.add_subplot(gs[1, 0])
    confidences_before = np.random.beta(5, 2, 1000)  # Skewed toward high confidence
    bins = np.linspace(0, 1, n_bins + 1)
    ax1_hist.hist(confidences_before, bins=bins, color=COLOR_QUATERNARY,
                 alpha=0.6, edgecolor='black', linewidth=1.1)
    ax1_hist.set_xlabel('Predicted Confidence', fontsize=10)
    ax1_hist.set_ylabel('Count', fontsize=10)
    ax1_hist.set_xlim([0, 1])
    ax1_hist.grid(True, axis='both', alpha=0.25, linestyle='--')
    ax1_hist.tick_params(labelsize=8)
    ax1_hist.text(
        -0.18,
        1.05,
        '(c)',
        transform=ax1_hist.transAxes,
        fontsize=12,
        fontweight='bold',
        va='bottom',
    )

    # Panel (b): After calibration
    ax2 = fig.add_subplot(gs[0, 1])

    # Well-calibrated after temperature scaling
    actual_acc_after = predicted_conf + np.random.randn(n_bins) * 0.015
    actual_acc_after = np.clip(actual_acc_after, 0, 1)

    ece_after = np.mean(np.abs(actual_acc_after - predicted_conf))

    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration',
            zorder=1)
    lower_after = np.clip(actual_acc_after - conf_band, 0, 1)
    upper_after = np.clip(actual_acc_after + conf_band, 0, 1)
    band2 = ax2.fill_between(
        predicted_conf,
        lower_after,
        upper_after,
        color=COLOR_LIGHT_BLUE,
        alpha=0.45,
        edgecolor=COLOR_PRIMARY,
        linewidth=1.0,
        zorder=1,
    )
    ax2.plot(predicted_conf, actual_acc_after, 'o-', color=COLOR_PRIMARY,
            linewidth=2.6, markersize=9, markeredgecolor='black',
            markeredgewidth=1.2, label='Model predictions', zorder=3)

    # Add note about accuracy
    ax2.text(
        0.98, 0.08, 'Top-1 accuracy unchanged', transform=ax2.transAxes,
        ha='right', va='bottom', fontsize=9, bbox=ANNOTATION_BOX,
    )

    ax2.set_xlabel('Predicted Confidence', fontsize=10)
    ax2.set_ylabel('Actual Accuracy', fontsize=10)
    ax2.set_title(
        f'After Calibration ($T = 1.5$)\n$\\textbf{{{{ECE}}}} = \\mathbf{{{{{ece_after:.3f}}}}}$',
        fontsize=12,
        pad=14,
    )
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.text(
        -0.18,
        1.05,
        '(b)',
        transform=ax2.transAxes,
        fontsize=12,
        fontweight='bold',
        va='bottom',
    )

    # Histogram below (b)
    ax2_hist = fig.add_subplot(gs[1, 1])
    confidences_after = np.random.beta(2, 2, 1000)  # More uniform
    ax2_hist.hist(confidences_after, bins=bins, color=COLOR_PRIMARY,
                 alpha=0.6, edgecolor='black', linewidth=1.1)
    ax2_hist.set_xlabel('Predicted Confidence', fontsize=10)
    ax2_hist.set_ylabel('Count', fontsize=10)
    ax2_hist.set_xlim([0, 1])
    ax2_hist.grid(True, axis='both', alpha=0.25, linestyle='--')
    ax2_hist.tick_params(labelsize=8)
    ax2_hist.text(
        -0.18,
        1.05,
        '(d)',
        transform=ax2_hist.transAxes,
        fontsize=12,
        fontweight='bold',
        va='bottom',
    )

    fig.tight_layout()
    # Extra top margin to accommodate legends moved above the axes
    fig.subplots_adjust(top=0.88)

    plt.savefig('figures/ch10_calibration/fig_10_5_before_after_calibration.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_5_before_after_calibration.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.5 saved")
    plt.close()

# ============================================================================
# FIGURE 10.6: Token-Level vs Sequence-Level Calibration
# ============================================================================

def figure_10_6():
    """Show difference between token-level and sequence-level calibration."""
    np.random.seed(42)

    fig = plt.figure(figsize=(FIGURE_WIDTH * 2.0, FIGURE_HEIGHT * 1.95))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.4, 1.15],
        width_ratios=[2, 1.05],
        # Increase vertical spacing to make room for legend below (a)
        hspace=0.9,
        wspace=0.4,
    )

    # Panel (a): Token-level visualization
    ax1 = fig.add_subplot(gs[0, :])

    # Sequence of tokens with confidences
    tokens = ['The', 'capital', 'of', 'France', 'is', 'Paris', '.', 'It', 'has']
    confidences = [0.8, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.8]
    correct = [True, False, True, True, True, True, True, False, True]
    correct_color = COLOR_MUTED_TEAL
    incorrect_color = COLOR_SECONDARY

    for i, (token, conf, corr) in enumerate(zip(tokens, confidences, correct)):
        color = correct_color if corr else incorrect_color
        rect = Rectangle((i, 0), 0.75, conf, facecolor=color, alpha=0.8,
                        edgecolor='black', linewidth=1.2)
        ax1.add_patch(rect)

        # Add token text
        ax1.text(i + 0.375, -0.16, token, ha='center', va='top', fontsize=11,
                 fontweight='bold')

        # Add checkmark or X
        symbol = '$\\checkmark$' if corr else '$\\times$'
        ax1.text(
            i + 0.375,
            conf + 0.08,
            symbol,
            ha='center',
            va='bottom',
            fontsize=18,
            color='black',
            fontweight='bold',
        )

    # Removed model confidence line to avoid overlay on bars

    # Removed inset reliability diagram (accuracy vs confidence) to avoid overlay

    ax1.set_xlim([-0.2, len(tokens)])
    ax1.set_ylim([-0.25, 1.05])
    ax1.set_ylabel('Token Confidence', fontsize=10)
    ax1.set_title(
        'Token-Level Calibration: Each Prediction $\\approx$ 80\\% confident',
        fontsize=12,
        pad=12,
    )
    # Keep legend but move it further below the x-axis; use proxy to indicate model confidence
    line_proxy = Line2D([0], [0], color='black', linestyle='--', linewidth=2, alpha=0.7,
                        label='Model confidence: 80\\%')
    ax1.legend(
        handles=[line_proxy],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.28),  # place below axis so it doesn't cover labels
        fontsize=9,
        framealpha=0.9,
    )
    ax1.set_xticks([])
    ax1.spines['bottom'].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.25, linestyle='--')
    ax1.text(
        -0.02,
        1.05,
        '(a)',
        transform=ax1.transAxes,
        fontsize=12,
        fontweight='bold',
        va='bottom',
    )

    # Panel (b): Sequence-level distribution
    ax2 = fig.add_subplot(gs[1, 0])

    # Simulate sequence qualities
    # If tokens are 80% accurate independently, sequences vary
    n_tokens = 8
    n_sequences = 1000
    token_accuracy = 0.8

    sequence_qualities = []
    for _ in range(n_sequences):
        correct_count = np.random.binomial(n_tokens, token_accuracy)
        sequence_qualities.append(correct_count / n_tokens)

    bins_seq = np.linspace(0, 1, 21)
    ax2.hist(sequence_qualities, bins=bins_seq, color=COLOR_PRIMARY, alpha=0.65,
            edgecolor='black', linewidth=1.1)

    # Add markers for mean and model prediction
    model_conf = 0.8
    actual_mean = np.mean(sequence_qualities)

    ax2.axvline(
        model_conf,
        color=COLOR_MUTED_PURPLE,
        linestyle='--',
        linewidth=2.5,
        label=(f'Model expectation: {model_conf:.0%}'.replace('%', '\\%')),
        zorder=3,
    )
    ax2.axvline(
        actual_mean,
        color=correct_color,
        linestyle='-',
        linewidth=2.5,
        label=(f'Observed mean: {actual_mean:.1%}'.replace('%', '\\%')),
        zorder=3,
    )

    ax2.text(0.24, 0.96, f'P(all correct) = $0.8^{n_tokens}$',
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='white', alpha=0.9))

    ax2.set_xlabel('Sequence Quality (fraction correct)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Sequence-Level Distribution: Errors Compound', fontsize=11, pad=10)
    ax2.legend(fontsize=9, loc='lower left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='both', linestyle='--')
    ymax = ax2.get_ylim()[1]
    ax2.annotate(
        'Confidence band moves right,\nbut realized quality lags',
        xy=(actual_mean, ymax * 0.7),
        xytext=(0.60, 0.70),
        textcoords='axes fraction',
        fontsize=8.8,
        ha='left',
        bbox=ANNOTATION_BOX,
        arrowprops=dict(arrowstyle='-|>', lw=1.5, color=COLOR_GRAY),
    )
    ax2.text(
        -0.12,
        1.05,
        '(b)',
        transform=ax2.transAxes,
        fontsize=12,
        fontweight='bold',
        va='bottom',
    )

    # Panel (c): Sequence-level miscalibration
    ax3 = fig.add_subplot(gs[1, 1])

    # Perfect calibration line
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Model point (overconfident at sequence level)
    ax3.plot(0.8, 0.8**8, 'o', markersize=15, color=COLOR_QUATERNARY,
            markeredgecolor='black', markeredgewidth=2, label='Model', zorder=3)

    ax3.annotate(
        '\\textbf{Miscalibrated!}',
        xy=(0.8, 0.8**8),
        xytext=(0.45, 0.35),
        fontsize=10,
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', edgecolor=COLOR_QUATERNARY, linewidth=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle='-|>', lw=1.8, color=COLOR_QUATERNARY),
    )

    ax3.set_xlabel('Predicted Confidence', fontsize=11)
    ax3.set_ylabel('Actual Sequence Quality', fontsize=11)
    ax3.set_title('Sequence-Level Calibration Gap', fontsize=11, pad=10)
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.text(
        -0.18,
        1.05,
        '(c)',
        transform=ax3.transAxes,
        fontsize=12,
        fontweight='bold',
        va='bottom',
    )

    # Clarifying subtitle
    fig.text(0.5, 0.02, 'Good token-level calibration does not guarantee good sequence-level calibration.',
             ha='center', fontsize=11, style='italic', color=COLOR_GRAY)

    plt.savefig('figures/ch10_calibration/fig_10_6_token_vs_sequence.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_6_token_vs_sequence.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.6 saved")
    plt.close()

# ============================================================================
# FIGURE 10.7: Compression vs Decision Quality
# ============================================================================

def figure_10_7():
    """Show that better compression doesn't always mean better decisions."""
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.7, FIGURE_HEIGHT))

    # Panel (a): Cross-entropy (compression quality)
    models = ['Model A\n(confidently\nwrong)', 'Model B\n(cautiously\ncorrect)']
    perplexity = [10, 12]
    colors = [COLOR_QUATERNARY, COLOR_PRIMARY]

    bars = ax1.bar(models, perplexity, color=colors, alpha=0.75,
                  edgecolor='black', linewidth=1.8)

    # Add star proxy color
    star_color = '#ffb000'
    # Position star above the best bar (Model A - lowest perplexity)
    best_bar_idx = 0
    bar = bars[best_bar_idx]
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        perplexity[best_bar_idx] + 1.8,
        '$\\bigstar$',
        ha='center',
        va='bottom',
        fontsize=22,
        color=star_color,
    )

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, perplexity)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.6, f'{val}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylabel('Perplexity', fontsize=11, rotation=0, labelpad=35)
    ax1.yaxis.set_label_coords(-0.28, 0.5)
    ax1.set_title('Cross-Entropy Loss\n(Compression Quality)', fontsize=12, pad=12)
    ax1.set_ylim([0, 15])
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    # Move '(lower is better)' above the '12' label on the second bar
    bar2 = bars[1]
    x2 = bar2.get_x() + bar2.get_width() / 2.0
    y2 = perplexity[1] + 1.4
    ax1.text(x2, y2, '(lower is better)',
             ha='center', va='bottom', fontsize=9, color=COLOR_GRAY)

    # Panel (b): Task performance (decision quality)
    accuracy = [65, 78]

    bars = ax2.bar(models, accuracy, color=colors, alpha=0.75,
                  edgecolor='black', linewidth=1.8)

    # Position star above the best bar (Model B - highest accuracy)
    best_bar_idx = 1
    bar = bars[best_bar_idx]
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        accuracy[best_bar_idx] + 6,
        '$\\bigstar$',
        ha='center',
        va='bottom',
        fontsize=22,
        color=star_color,
    )

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, accuracy)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2.2, f'{val}\\%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_ylabel('Accuracy (\\%)', fontsize=11, rotation=0, labelpad=35)
    ax2.yaxis.set_label_coords(-0.25, 0.5)
    ax2.set_title('Task Performance\n(Decision Quality)', fontsize=12, pad=12)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.text(0.02, 0.92, '(higher is better)', transform=ax2.transAxes,
            ha='left', va='top', fontsize=9, color=COLOR_GRAY)

    # Add key message with star explanation
    fig.text(0.5, -0.05, 'Better compression $\\neq$ better decisions',
             ha='center', fontsize=14, fontweight='bold', color=COLOR_QUATERNARY,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff4e6', edgecolor='#e69f00', linewidth=1.2, alpha=0.95))

    # Add star legend (positioned safely away from edge)
    fig.text(
        0.5, 0.96,
        '$\\bigstar$ = Best score in each metric',
        ha='center',
        fontsize=10,
        color=COLOR_GRAY,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.18)
    plt.savefig('figures/ch10_calibration/fig_10_7_compression_vs_decisions.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_7_compression_vs_decisions.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.7 saved")
    plt.close()

# ============================================================================
# FIGURE 10.8: Data Contamination Detection
# ============================================================================

def figure_10_8():
    """Visualize data contamination detection using n-gram overlap."""
    np.random.seed(42)

    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.65, FIGURE_HEIGHT * 1.68))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.38, wspace=0.3)

    # Panel (a): Clean split
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    # Draw Venn diagram
    circle1 = Circle((0.35, 0.5), 0.25, facecolor=COLOR_LIGHT_BLUE,
                    edgecolor=COLOR_PRIMARY, linewidth=3, alpha=0.6)
    circle2 = Circle((0.65, 0.5), 0.25, facecolor=COLOR_LIGHT_GREEN,
                    edgecolor=COLOR_TERTIARY, linewidth=3, alpha=0.6)
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)

    ax1.text(0.3, 0.5, 'Training\nSet', ha='center', va='center',
            fontsize=10, color=COLOR_PRIMARY, fontweight='bold')
    ax1.text(0.7, 0.5, 'Test\nSet', ha='center', va='center',
            fontsize=10, color=COLOR_TERTIARY, fontweight='bold')

    ax1.text(
        0.08,
        0.22,
        '50-gram matches: 0.0\\%\n20-gram matches: 0.1\\%',
        ha='left',
        va='bottom',
        fontsize=10.5,
        bbox=ANNOTATION_BOX,
    )

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('Clean Split: Minimal Overlap', fontsize=11,
                 fontweight='bold', pad=10)
    ax1.text(
        0.02,
        0.9,
        '(a)',
        transform=ax1.transAxes,
        fontsize=12,
        fontweight='bold',
    )

    # Panel (b): Contaminated split
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Draw overlapping circles
    circle1 = Circle((0.38, 0.5), 0.28, facecolor=COLOR_LIGHT_BLUE,
                    edgecolor=COLOR_PRIMARY, linewidth=3, alpha=0.6)
    circle2 = Circle((0.62, 0.5), 0.28, facecolor=COLOR_LIGHT_GREEN,
                    edgecolor=COLOR_TERTIARY, linewidth=3, alpha=0.6)
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)

    # Highlight leak
    leak = Circle((0.5, 0.5), 0.08, facecolor=COLOR_QUATERNARY,
                 edgecolor='darkred', linewidth=3.0, alpha=0.3)
    ax2.add_patch(leak)

    ax2.text(0.28, 0.5, 'Train', ha='center', va='center',
            fontsize=9, color=COLOR_PRIMARY, fontweight='bold')
    ax2.text(0.72, 0.5, 'Test', ha='center', va='center',
            fontsize=9, color=COLOR_TERTIARY, fontweight='bold')

    ax2.annotate(
        'Leak detected',
        xy=(0.5, 0.5),
        xytext=(0.62, 0.78),
        fontsize=9,
        ha='left',
        bbox=ANNOTATION_BOX,
        arrowprops=dict(arrowstyle='-|>', lw=2.2, color=COLOR_QUATERNARY),
    )

    ax2.text(
        0.08,
        0.15,
        '50-gram matches: 12\\%\n20-gram matches: 23\\%\n\\textbf{Inflated metrics}',
        ha='left',
        fontsize=10,
        bbox=ANNOTATION_BOX,
    )

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('Contaminated Split: Test Data Revisited', fontsize=11,
                 fontweight='bold', pad=10)
    ax2.text(
        0.02,
        0.9,
        '(b)',
        transform=ax2.transAxes,
        fontsize=12,
        fontweight='bold',
    )

    # Panel (c): Distribution of n-gram overlaps
    ax3 = fig.add_subplot(gs[1, :])

    # Clean split: mostly low overlap
    clean_overlaps = np.random.gamma(1, 2, 500)
    clean_overlaps = np.clip(clean_overlaps, 0, 30)

    # Contaminated split: some high overlap
    contaminated_overlaps = np.concatenate([
        np.random.gamma(1, 2, 400),
        np.random.uniform(50, 150, 100)
    ])
    contaminated_overlaps = np.clip(contaminated_overlaps, 0, 150)

    # Plot distributions
    bins = np.concatenate([np.linspace(0, 40, 21), np.linspace(40, 150, 18)])
    ax3.hist(clean_overlaps, bins=bins, alpha=0.6, color=COLOR_TERTIARY,
            edgecolor='black', linewidth=1, label='Clean split', rwidth=0.85)
    ax3.hist(contaminated_overlaps, bins=bins, alpha=0.6, color=COLOR_QUATERNARY,
            edgecolor='black', linewidth=1, label='Contaminated split', rwidth=0.85)

    # Add threshold line
    threshold = 50
    ax3.axvline(threshold, color=COLOR_QUATERNARY, linestyle='--', linewidth=3.0,
               label='Threshold (50 tokens)')

    # Add regions
    ax3.axvspan(0, threshold, alpha=0.18, color=COLOR_LIGHT_GREEN)
    ax3.axvspan(threshold, 150, alpha=0.2, color=COLOR_LIGHT_RED)

    ax3.set_xlabel('Longest Common N-gram Length (tokens)', fontsize=10)
    ax3.set_ylabel('Density (log scale)', fontsize=10)
    ax3.set_title('Detection Method: N-gram Overlap Distribution', fontsize=11,
                 fontweight='bold', pad=12)
    ax3.set_xlim([0, 150])
    ax3.set_yscale('log')
    ymax = ax3.get_ylim()[1]
    ax3.text(
        threshold + 4,
        ymax / 9,
        'Flag sequences with $\\geq$ 50 token matches',
        fontsize=9.5,
        bbox=ANNOTATION_BOX,
    )
    # Moved to bottom left to avoid crowding upper area
    ax3.text(0.02, 0.15, 'Log scale to show both tails', transform=ax3.transAxes,
             ha='left', va='bottom', fontsize=9, color=COLOR_GRAY)
    ax3.legend(
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor=COLOR_GRAY,
        fontsize=9,
    )
    ax3.grid(True, alpha=0.3, axis='both', linestyle='--')
    ax3.text(
        -0.06,
        1.02,
        '(c)',
        transform=ax3.transAxes,
        fontsize=12,
        fontweight='bold',
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.02,
             'Clean split: 0.1% matches | Contaminated split: 23% matches (230× increase)',
             ha='center', fontsize=10, color=COLOR_GRAY)

    plt.savefig('figures/ch10_calibration/fig_10_8_contamination_detection.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch10_calibration/fig_10_8_contamination_detection.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 10.8 saved")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 10: Calibration and Confidence - Figure Generation")
    print("=" * 70)
    print()

    figure_10_1()
    figure_10_2()
    figure_10_3()
    figure_10_4()
    figure_10_5()
    figure_10_6()
    figure_10_7()
    figure_10_8()

    print()
    print("=" * 70)
    print("✓ All figures generated successfully!")
    print("Output location: figures/ch10_calibration/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 70)
    print()
    print("Figure Summary:")
    print("  10.1: Calibration Plots (Well-Calibrated vs Overconfident)")
    print("  10.2: Temperature Scaling Effects")
    print("  10.3: Decoding Strategies Comparison")
    print("  10.4: Diversity-Error Tradeoff")
    print("  10.5: Before and After Calibration")
    print("  10.6: Token-Level vs Sequence-Level Calibration")
    print("  10.7: Compression Quality vs Decision Quality")
    print("  10.8: Data Contamination Detection")
    print()
