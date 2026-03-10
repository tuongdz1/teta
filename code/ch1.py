"""
Figure generation utilities for Chapter 1 (Information as Compression).
Produces the eight figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats
from scipy.interpolate import interp1d
import os

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# LaTeX font configuration for publication quality
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
rc('axes', labelsize=11)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)

# Color palette - blue/orange academic
COLOR_PRIMARY = '#1f77b4'    # Blue
COLOR_SECONDARY = '#ff7f0e'  # Orange
COLOR_TERTIARY = '#2ca02c'   # Green
COLOR_QUATERNARY = '#d62728' # Red
COLOR_GRAY = '#7f7f7f'
COLOR_LIGHT_BLUE = '#aec7e8'
COLOR_LIGHT_ORANGE = '#ffbb78'

# Standard figure size for book (width in inches)
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0

# DPI for high quality
DPI = 300

# Create output directory
os.makedirs('figures/ch01_information_compression/', exist_ok=True)

# ============================================================================
# FIGURE 1.1: Probability and Code Length Relationship
# ============================================================================

def figure_1_1():
    """Shannon's fundamental insight: probability vs code length"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.6, FIGURE_HEIGHT))

    # Panel (a): Continuous relationship
    p = np.linspace(0.01, 1, 1000)
    codelength_bits = -np.log2(p)
    codelength_nats = -np.log(p)

    ax1.plot(p, codelength_bits, color=COLOR_PRIMARY, linewidth=2, label='Bits ($\\log_2(1/p)$)')
    ax1.plot(p, codelength_nats, color=COLOR_SECONDARY, linewidth=2, label='Nats ($\\ln(1/p)$)')
    ax1.set_xlabel('Probability $p$')
    ax1.set_ylabel('Optimal Code Length')
    ax1.set_title('(a) Probability vs Code Length')
    ax1.grid(True, alpha=0.3)
    leg1 = ax1.legend(loc='upper right')
    # Improve legend readability on any background
    leg1.get_frame().set_alpha(0.9)
    leg1.get_frame().set_facecolor('white')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 8])

    # Panel (b): Discrete examples with better spacing
    probabilities = [1/2, 1/4, 1/8, 1/16]
    code_lengths = [1, 2, 3, 4]
    labels = ['1/2', '1/4', '1/8', '1/16']
    codes = ['0', '00', '000', '0000']

    x_pos = np.arange(len(probabilities))
    bars = ax2.bar(x_pos, code_lengths, color=COLOR_PRIMARY, alpha=0.7, edgecolor='black', width=0.6)

    # Add binary code labels above bars with more spacing
    for i, (bar, code) in enumerate(zip(bars, codes)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{code}', ha='center', va='bottom', fontsize=9, family='monospace')

    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Code Length (bits)')
    ax2.set_title('(b) Concrete Examples')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'$p = {l}$' for l in labels])
    # Extra headroom for bar-top labels to avoid any clipping
    ax2.set_ylim([0, 5.4])
    ax2.grid(True, alpha=0.3, axis='y')
    leg2 = ax2.legend(loc='best') if ax2.get_legend() is None else ax2.get_legend()
    if leg2 is not None:
        leg2.get_frame().set_alpha(0.9)
        leg2.get_frame().set_facecolor('white')

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/ch01_information_compression/fig_1_1_probability_codelength.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_1_probability_codelength.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.1 saved")
    plt.close()

# ============================================================================
# FIGURE 1.2: MDL Polynomial Fitting Example
# ============================================================================

def figure_1_2():
    """Visualize underfitting, good fit, and overfitting in MDL context"""
    np.random.seed(42)

    # Generate data with cubic underlying structure + noise
    x = np.linspace(0, 1, 20)
    y_true = 2 + 3*x - 5*x**2 + 2*x**3
    y = y_true + np.random.normal(0, 0.3, len(x))

    # Fit models
    x_smooth = np.linspace(0, 1, 200)

    # Model 1: Linear (degree 1, 2 parameters)
    p1 = np.polyfit(x, y, 1)
    y1 = np.polyval(p1, x_smooth)

    # Model 2: Cubic (degree 3, 4 parameters)
    p2 = np.polyfit(x, y, 3)
    y2 = np.polyval(p2, x_smooth)

    # Model 3: Degree 19 (20 parameters)
    p3 = np.polyfit(x, y, min(19, len(x)-1))
    y3 = np.polyval(p3, x_smooth)

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.9, FIGURE_HEIGHT*1.05))

    # Model 1: Underfitting
    axes[0].scatter(x, y, color='black', s=30, alpha=0.6, zorder=3, label='Data')
    axes[0].plot(x_smooth, y1, color=COLOR_PRIMARY, linewidth=2, label='Linear fit')
    axes[0].set_title('(a) Linear: 2 params\n$L_{\\text{model}}=20$ bits\n$L_{\\text{data}}=500$ bits\n$L_{\\text{total}}=520$ bits',
                      fontsize=9.5, pad=10)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].grid(True, alpha=0.3)
    leg = axes[0].legend(loc='upper left', fontsize=8)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')

    # Model 2: Good fit
    axes[1].scatter(x, y, color='black', s=30, alpha=0.6, zorder=3, label='Data')
    axes[1].plot(x_smooth, y2, color=COLOR_SECONDARY, linewidth=2, label='Cubic fit')
    axes[1].set_title('(b) Cubic: 4 params\n$L_{\\text{model}}=40$ bits\n$L_{\\text{data}}=200$ bits\n$L_{\\text{total}}=240$ bits',
                      fontsize=9.5, pad=10)
    axes[1].set_xlabel('$x$')
    axes[1].grid(True, alpha=0.3)
    leg = axes[1].legend(loc='upper left', fontsize=8)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')

    # Model 3: Overfitting
    axes[2].scatter(x, y, color='black', s=30, alpha=0.6, zorder=3, label='Data')
    axes[2].plot(x_smooth, y3, color=COLOR_QUATERNARY, linewidth=2, label='Degree 19 fit')
    axes[2].set_title('(c) Degree 19: 20 params\n$L_{\\text{model}}=200$ bits\n$L_{\\text{data}}=0$ bits\n$L_{\\text{total}}=200$ bits',
                      fontsize=9.5, pad=10)
    axes[2].set_xlabel('$x$')
    axes[2].grid(True, alpha=0.3)
    leg = axes[2].legend(loc='upper left', fontsize=8)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    # Small vertical headroom so markers/titles never touch borders
    for ax in axes:
        ax.margins(y=0.05)

    plt.tight_layout(pad=1.9)
    # Slightly increase horizontal spacing between panels
    plt.subplots_adjust(wspace=0.28)
    plt.savefig('figures/ch01_information_compression/fig_1_2_mdl_polynomial.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_2_mdl_polynomial.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.2 saved")
    plt.close()

# ============================================================================
# FIGURE 1.3: Cross Entropy and KL Divergence
# ============================================================================

def figure_1_3():
    """Visualize cross entropy as coding inefficiency"""
    x = np.linspace(-4, 6, 1000)

    # True distribution p (blue)
    p = stats.norm.pdf(x, loc=0, scale=1)

    # Model distribution q (orange, shifted and wider)
    q = stats.norm.pdf(x, loc=1.5, scale=1.3)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT*1.05))

    # Plot distributions
    ax.plot(x, p, color=COLOR_PRIMARY, linewidth=2.5, label='True distribution $p(x)$')
    ax.plot(x, q, color=COLOR_SECONDARY, linewidth=2.5, label='Model distribution $q(x)$')

    # Shade region showing extra cost
    mask = p > 0.01
    ax.fill_between(x[mask], 0, p[mask], alpha=0.15, color=COLOR_PRIMARY,
                     label='Entropy $H(p)$')

    # Add annotation in better position
    ax.annotate('Extra bits paid\nwhen $q \\neq p$\n$= \\text{KL}(p \\| q)$',
                xy=(0.5, 0.22), xytext=(-2.8, 0.35),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_ORANGE, alpha=0.7))

    ax.set_xlabel('$x$')
    ax.set_ylabel('Probability Density')
    ax.set_title('Cross Entropy: $H(p,q) = H(p) + \\text{KL}(p \\| q)$', pad=12)
    leg = ax.legend(loc='upper right', fontsize=9)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.45])
    ax.set_xlim([-4, 6])

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch01_information_compression/fig_1_3_cross_entropy_kl.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_3_cross_entropy_kl.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.3 saved")
    plt.close()

# ============================================================================
# FIGURE 1.4: KL Divergence Asymmetry (I-projection vs M-projection)
# ============================================================================

def figure_1_4():
    """Demonstrate KL(p||q) ≠ KL(q||p) with I-projection vs M-projection"""
    x = np.linspace(-5, 10, 1000)

    # True distribution: mixture of two Gaussians
    p = 0.5 * stats.norm.pdf(x, loc=-1, scale=0.8) + 0.5 * stats.norm.pdf(x, loc=3, scale=0.8)

    # I-projection: minimize KL(p||q) - inclusive, covers both modes
    q_inclusive = stats.norm.pdf(x, loc=1, scale=2.5)

    # M-projection: minimize KL(q||p) - exclusive, picks one mode
    q_exclusive = stats.norm.pdf(x, loc=-1, scale=0.9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.6, FIGURE_HEIGHT*1.05))

    # Panel (a): I-projection
    ax1.plot(x, p, color='black', linewidth=2.5, linestyle='--', label='True $p(x)$ (bimodal)')
    ax1.plot(x, q_inclusive, color=COLOR_PRIMARY, linewidth=2.5, label='$q^*(x)$ (I-projection)')
    ax1.fill_between(x, 0, q_inclusive, alpha=0.2, color=COLOR_PRIMARY)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('(a) $\\min_q \\text{KL}(p \\| q)$\n(I-projection)\nMoment matching, covers modes',
                  fontsize=10, pad=10)
    leg = ax1.legend(loc='upper right', fontsize=9)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.35])

    # Panel (b): M-projection
    ax2.plot(x, p, color='black', linewidth=2.5, linestyle='--', label='True $p(x)$ (bimodal)')
    ax2.plot(x, q_exclusive, color=COLOR_SECONDARY, linewidth=2.5, label='$q^*(x)$ (M-projection)')
    ax2.fill_between(x, 0, q_exclusive, alpha=0.2, color=COLOR_SECONDARY)
    ax2.set_xlabel('$x$')
    ax2.set_title('(b) $\\min_q \\text{KL}(q \\| p)$\n(M-projection)\nMode seeking, picks one mode',
                  fontsize=10, pad=10)
    leg = ax2.legend(loc='upper right', fontsize=9)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.35])

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/ch01_information_compression/fig_1_4_kl_asymmetry.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_4_kl_asymmetry.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.4 saved")
    plt.close()

# ============================================================================
# FIGURE 1.5: Rate-Distortion Curve
# ============================================================================

def figure_1_5():
    """Show the fundamental R(D) tradeoff"""
    # Create a realistic R-D curve (concave, decreasing)
    D = np.linspace(0.01, 2, 100)
    R = 2 * np.exp(-D) + 0.1

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT*1.05))

    # Plot the R-D curve
    ax.plot(D, R, color=COLOR_PRIMARY, linewidth=3, label='$R(D)$ curve', zorder=3)
    # Ensure fills sit visually behind the curve for clarity
    ax.fill_between(D, R, 5, alpha=0.15, color=COLOR_LIGHT_BLUE,
                     label='Achievable region', zorder=1)
    ax.fill_between(D, 0, R, alpha=0.25, color='red',
                     label='Impossible region', zorder=0)

    # Mark operating points with better positioning
    points = [
        (0.3, 2 * np.exp(-0.3) + 0.1, 'High rate\nLow distortion', (0.5, 1.85)),
        (1.0, 2 * np.exp(-1.0) + 0.1, 'Balanced', (1.15, 0.95)),
        (1.7, 2 * np.exp(-1.7) + 0.1, 'Low rate\nHigh distortion', (1.35, 0.45))
    ]

    for d, r, label, textpos in points:
        ax.plot(d, r, 'o', markersize=8, color=COLOR_SECONDARY, zorder=5)
        ax.annotate(label, xy=(d, r), xytext=textpos,
                   fontsize=9, ha='left',
                   arrowprops=dict(arrowstyle='->', color='black', lw=1))

    ax.set_xlabel('Distortion $D$')
    ax.set_ylabel('Rate $R$ (bits)')
    ax.set_title('Rate-Distortion Function: Fundamental Tradeoff', pad=12)
    leg = ax.legend(loc='upper right', fontsize=9)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2.5])

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch01_information_compression/fig_1_5_rate_distortion.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_5_rate_distortion.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.5 saved")
    plt.close()

# ============================================================================
# FIGURE 1.6: Beta-VAE on Rate-Distortion Curve
# ============================================================================

def figure_1_6():
    """Show how different β values trace the R-D curve"""
    # R-D curve
    D = np.linspace(0.05, 1.5, 100)
    R = 1.5 * np.exp(-D*0.8) + 0.2

    # Beta-VAE operating points
    betas = [0.5, 1.0, 2.0, 5.0]
    beta_points = {
        0.5: (1.2, 1.5 * np.exp(-1.2*0.8) + 0.2),
        1.0: (0.6, 1.5 * np.exp(-0.6*0.8) + 0.2),
        2.0: (0.3, 1.5 * np.exp(-0.3*0.8) + 0.2),
        5.0: (0.1, 1.5 * np.exp(-0.1*0.8) + 0.2),
    }

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.15, FIGURE_HEIGHT*1.05))

    # Plot R-D curve
    ax.plot(D, R, color=COLOR_GRAY, linewidth=2.5, linestyle='--',
            label='Optimal $R(D)$ frontier', zorder=1)

    # Plot beta-VAE points
    colors = [COLOR_QUATERNARY, COLOR_SECONDARY, COLOR_PRIMARY, COLOR_TERTIARY]
    for i, beta in enumerate(betas):
        d, r = beta_points[beta]
        label = f'$\\beta = {beta}$' if beta != 1.0 else f'$\\beta = {beta}$ (std VAE)'
        ax.plot(d, r, 'o', markersize=10, color=colors[i],
                label=label, zorder=5, markeredgecolor='black', markeredgewidth=1.5)

        # Add annotations for extreme cases only
        if beta == 0.5:
            ax.annotate('Low compression\nHigh fidelity',
                       xy=(d, r), xytext=(d+0.35, r+0.2),
                       fontsize=8, ha='left',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1))
        elif beta == 5.0:
            ax.annotate('High compression\nLow fidelity',
                       xy=(d, r), xytext=(d+0.25, r-0.25),
                       fontsize=8, ha='left',
                       arrowprops=dict(arrowstyle='->', color='black', lw=1))

    ax.set_xlabel('Distortion (reconstruction error)')
    ax.set_ylabel('Rate (KL to prior)')
    ax.set_title('$\\beta$-VAE: Controlling the Rate-Distortion Tradeoff', pad=12)
    leg = ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    leg.get_frame().set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.5])
    ax.set_ylim([0, 1.8])

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch01_information_compression/fig_1_6_beta_vae_rate_distortion.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_6_beta_vae_rate_distortion.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.6 saved")
    plt.close()

# ============================================================================
# FIGURE 1.7: Non-linear Cross Entropy to Accuracy
# ============================================================================

def figure_1_7():
    """Show diminishing returns in cross entropy reduction"""
    # Data from the example in the text
    ce = np.array([5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0])
    accuracy = np.array([65, 78, 85, 89, 91.5, 92.5, 93.0])

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.15, FIGURE_HEIGHT*1.05))

    # Plot the curve
    ax.plot(ce, accuracy, 'o-', color=COLOR_PRIMARY, linewidth=2.5,
            markersize=8, label='Empirical relationship')

    # Highlight key regions with reduced opacity and better positioning
    ax.axvspan(4.5, 5.5, alpha=0.12, color=COLOR_TERTIARY)
    ax.axvspan(2.5, 3.5, alpha=0.12, color=COLOR_SECONDARY)
    ax.axvspan(0.5, 1.5, alpha=0.12, color=COLOR_QUATERNARY)

    # Add text labels for regions (moved outside the shaded areas)
    ax.text(5.0, 96, 'Early:\nhigh\nmarginal\nvalue', fontsize=8, ha='center',
            color=COLOR_TERTIARY, weight='bold')
    ax.text(3.0, 96, 'Middle:\nmoderate\nreturns', fontsize=8, ha='center',
            color=COLOR_SECONDARY, weight='bold')
    ax.text(1.0, 96, 'Late:\ndiminishing\nreturns', fontsize=8, ha='center',
            color=COLOR_QUATERNARY, weight='bold')

    # Add improvement annotations with better spacing
    improvements = [
        (4.5, 71.5, '13 pts'),
        (3.5, 81.5, '7 pts'),
        (1.5, 92.25, '1.5 pts'),
    ]

    for x, y, text in improvements:
        ax.annotate(f'{text}', xy=(x, y), fontsize=9,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                            edgecolor='black', alpha=0.9, linewidth=1))

    ax.set_xlabel('Cross Entropy (nats)')
    ax.set_ylabel('Accuracy (\\%)')
    ax.set_title('Non-linear Relationship: Cross Entropy vs Performance', pad=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 5.5])
    ax.set_ylim([60, 100])

    # Invert x-axis so lower cross entropy is to the right (better models)
    ax.invert_xaxis()

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch01_information_compression/fig_1_7_ce_accuracy_nonlinear.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_7_ce_accuracy_nonlinear.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.7 saved")
    plt.close()

# ============================================================================
# FIGURE 1.8: Perplexity as Geometric Mean
# ============================================================================

def figure_1_8():
    """Conceptual visualization of perplexity as geometric mean"""
    np.random.seed(42)

    # Simulate varying uncertainty across 20 predictions
    n_predictions = 20
    positions = np.arange(n_predictions)

    # Create varying inverse probabilities (branching factors)
    inverse_probs = np.concatenate([
        np.random.uniform(1.1, 2, 8),
        np.random.uniform(5, 15, 8),
        np.random.uniform(50, 200, 4)
    ])

    # Calculate geometric mean (this is perplexity)
    geometric_mean = np.exp(np.mean(np.log(inverse_probs)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT*1.4))

    # Panel (a): Varying branching factors
    bars = ax1.bar(positions, inverse_probs, color=COLOR_PRIMARY, alpha=0.7,
                   edgecolor='black', linewidth=0.5)

    # Color bars by magnitude
    for i, bar in enumerate(bars):
        if inverse_probs[i] < 5:
            bar.set_color(COLOR_TERTIARY)
            bar.set_alpha(0.7)
        elif inverse_probs[i] < 30:
            bar.set_color(COLOR_PRIMARY)
            bar.set_alpha(0.7)
        else:
            bar.set_color(COLOR_QUATERNARY)
            bar.set_alpha(0.7)

    # Add geometric mean line
    ax1.axhline(geometric_mean, color='black', linestyle='--', linewidth=2,
                label=f'Perplexity = {geometric_mean:.1f} (geometric mean)')

    ax1.set_ylabel('$1/p_i$ (branching factor)')
    ax1.set_title('(a) Varying Uncertainty Across Predictions', pad=10)
    leg = ax1.legend(loc='upper right', fontsize=9)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 250])
    ax1.set_xticks([])

    # Panel (b): Log scale shows geometric mean
    bars2 = ax2.bar(positions, inverse_probs, color=COLOR_PRIMARY, alpha=0.7,
            edgecolor='black', linewidth=0.5)

    # Color bars same as above
    for i, bar in enumerate(bars2):
        if inverse_probs[i] < 5:
            bar.set_color(COLOR_TERTIARY)
            bar.set_alpha(0.7)
        elif inverse_probs[i] < 30:
            bar.set_color(COLOR_PRIMARY)
            bar.set_alpha(0.7)
        else:
            bar.set_color(COLOR_QUATERNARY)
            bar.set_alpha(0.7)

    ax2.axhline(geometric_mean, color='black', linestyle='--', linewidth=2,
                label=f'Geometric mean = {geometric_mean:.1f}')
    ax2.set_yscale('log')
    ax2.set_xlabel('Prediction index')
    ax2.set_ylabel('$1/p_i$ (log scale)')
    ax2.set_title('(b) Geometric Mean on Log Scale', pad=10)
    leg = ax2.legend(loc='upper right', fontsize=9)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor('white')
    # Remove gridlines on panel (b)
    ax2.grid(False)

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/ch01_information_compression/fig_1_8_perplexity_geometric.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch01_information_compression/fig_1_8_perplexity_geometric.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 1.8 saved")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Chapter 1 Figures: Information as Compression")
    print("=" * 60)
    print()

    figure_1_1()
    figure_1_2()
    figure_1_3()
    figure_1_4()
    figure_1_5()
    figure_1_6()
    figure_1_7()
    figure_1_8()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("Output location: figures/ch01_information_compression/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 60)
    print()
    print("Figure Summary:")
    print("  1.1: Probability and Code Length Relationship")
    print("  1.2: MDL Polynomial Fitting Example")
    print("  1.3: Cross Entropy and KL Divergence")
    print("  1.4: KL Divergence Asymmetry (I vs M projection)")
    print("  1.5: Rate-Distortion Curve")
    print("  1.6: Beta-VAE on Rate-Distortion Curve")
    print("  1.7: Non-linear Cross Entropy to Accuracy")
    print("  1.8: Perplexity as Geometric Mean")
