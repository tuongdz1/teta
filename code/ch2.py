"""
Figure generation utilities for Chapter 2 (Bayesian Predictive Inference).
Produces the eight figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats
from scipy.special import beta as beta_func
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

# Standard figure size
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0
DPI = 300

# Create output directory - CORRECTED PATH
os.makedirs('figures/ch02_bayesian_predictive', exist_ok=True)

# ============================================================================
# FIGURE 2.1: Prior to Posterior Update (Bayesian Coin Flip)
# ============================================================================

def figure_2_1():
    """Visualize Bayesian updating from prior to posterior"""
    theta = np.linspace(0, 1, 1000)

    # Prior: Beta(1,1) = Uniform
    prior = stats.beta.pdf(theta, 1, 1)

    # Posterior after 7 heads, 3 tails: Beta(8,4)
    posterior = stats.beta.pdf(theta, 8, 4)

    # Likelihood (not normalized)
    likelihood = theta**7 * (1-theta)**3
    likelihood = likelihood / np.max(likelihood) * np.max(posterior)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT*1.05))

    # Plot prior, likelihood, posterior
    ax.plot(theta, prior, color=COLOR_GRAY, linewidth=2,
            linestyle='--', label='Prior: $\\mathrm{Beta}(1,1)$')
    ax.plot(theta, likelihood, color=COLOR_SECONDARY, linewidth=2,
            alpha=0.6, linestyle=':', label='Likelihood (scaled)')
    ax.plot(theta, posterior, color=COLOR_PRIMARY, linewidth=2.5,
            label='Posterior: $\\mathrm{Beta}(8,4)$')

    # Mark the posterior mean
    post_mean = 8 / (8 + 4)
    ax.axvline(post_mean, color=COLOR_PRIMARY, linestyle='--',
               alpha=0.5, linewidth=1.5)
    # Place a white callout box centered on the line for readability
    ax.text(post_mean, 2.98, f'Posterior mean = {post_mean:.3f}',
            fontsize=9, color=COLOR_PRIMARY, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.95, edgecolor='none'))

    # Mark MLE - better positioned
    mle = 7/10
    ax.axvline(mle, color=COLOR_QUATERNARY, linestyle='--',
               alpha=0.5, linewidth=1.5)
    # Place a white callout box centered on the line for readability
    ax.text(mle, 3.12, f'MLE = {mle:.2f}',
            fontsize=9, color=COLOR_QUATERNARY, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.95, edgecolor='none'))

    ax.set_xlabel('$\\theta$ (probability of heads)')
    ax.set_ylabel('Density')
    ax.set_title('Bayesian Updating: From Prior to Posterior', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 3.2])

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_1_bayesian_update.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_1_bayesian_update.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.1 saved")
    plt.close()

# ============================================================================
# FIGURE 2.2: Posterior Predictive Distribution
# ============================================================================

def figure_2_2():
    """Compare point estimate vs posterior predictive"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.6, FIGURE_HEIGHT*1.05))

    theta = np.linspace(0, 1, 1000)
    posterior = stats.beta.pdf(theta, 8, 4)

    # Panel (a): Posterior over theta
    ax1.plot(theta, posterior, color=COLOR_PRIMARY, linewidth=2.5)
    ax1.fill_between(theta, 0, posterior, alpha=0.2, color=COLOR_PRIMARY)

    # Mark point estimates
    post_mean = 8/12
    mle = 7/10

    ax1.axvline(post_mean, color=COLOR_PRIMARY, linestyle='--', linewidth=2,
                label=f'Posterior mean: {post_mean:.3f}')
    ax1.axvline(mle, color=COLOR_QUATERNARY, linestyle='--', linewidth=2,
                label=f'MLE: {mle:.2f}')

    ax1.set_xlabel('$\\theta$ (probability of heads)')
    ax1.set_ylabel('Posterior density $p(\\theta \\mid D)$')
    ax1.set_title('(a) Posterior Distribution', pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])

    # Panel (b): Predictive probability - improved spacing
    outcomes = ['Tails', 'Heads']

    # MLE approach
    mle_probs = [1-mle, mle]

    # Posterior predictive
    pp_probs = [4/12, 8/12]

    x = np.arange(len(outcomes))
    width = 0.35

    ax2.bar(x - width/2, mle_probs, width, label='MLE (point estimate)',
            color=COLOR_QUATERNARY, alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, pp_probs, width, label='Posterior predictive',
            color=COLOR_PRIMARY, alpha=0.7, edgecolor='black')

    # Add value labels on bars with better spacing
    for i, (m, p) in enumerate(zip(mle_probs, pp_probs)):
        ax2.text(i - width/2, m + 0.025, f'{m:.3f}', ha='center', fontsize=8)
        ax2.text(i + width/2, p + 0.025, f'{p:.3f}', ha='center', fontsize=8)

    ax2.set_xlabel('Next flip outcome')
    ax2.set_ylabel('Probability')
    ax2.set_title('(b) Prediction for Next Flip', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcomes)
    # Place legend inside the axes (upper left) as requested
    ax2.legend(fontsize=9, loc='upper left', frameon=True)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 0.85])

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_2_posterior_predictive.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_2_posterior_predictive.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.2 saved")
    plt.close()

# ============================================================================
# FIGURE 2.3: Different Losses → Different Decisions
# ============================================================================

def figure_2_3():
    """Bimodal distribution showing mode, mean, median"""
    x = np.linspace(-5, 25, 1000)

    # Bimodal mixture: 70% at 20°C, 30% at 5°C
    p1 = 0.7 * stats.norm.pdf(x, loc=20, scale=2)
    p2 = 0.3 * stats.norm.pdf(x, loc=5, scale=2)
    mixture = p1 + p2

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT*1.05))

    # Plot the distribution
    ax.plot(x, mixture, color=COLOR_PRIMARY, linewidth=2.5,
            label='Posterior predictive')
    ax.fill_between(x, 0, mixture, alpha=0.2, color=COLOR_PRIMARY)

    # Calculate key statistics
    mode = 20
    mean = 0.7 * 20 + 0.3 * 5
    median = 18

    # Mark the three decision points
    ax.axvline(mode, color=COLOR_QUATERNARY, linestyle='--', linewidth=2,
               label=f'Mode (0-1 loss): {mode}°C')
    ax.axvline(mean, color=COLOR_SECONDARY, linestyle='--', linewidth=2,
               label=f'Mean (squared loss): {mean:.1f}°C')
    ax.axvline(median, color=COLOR_TERTIARY, linestyle='--', linewidth=2,
               label=f'Median (absolute loss): {median}°C')

    # Add annotation with better positioning (avoid overlapping the dashed line)
    y_mode = mixture[np.argmin(np.abs(x - mode))]
    ax.annotate('Most likely', xy=(mode, y_mode), xytext=(21.2, y_mode + 0.035),
                ha='left', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_ORANGE, alpha=0.7))

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Different Loss Functions $\\Rightarrow$ Different Optimal Decisions', pad=12)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-5, 25])
    ax.set_ylim([0, 0.19])

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_3_loss_decisions.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_3_loss_decisions.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.3 saved")
    plt.close()

# ============================================================================
# FIGURE 2.4: Exchangeability and Permutation Invariance
# ============================================================================

def figure_2_4():
    """Visualize that predictions should be invariant to data order"""
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.6, FIGURE_HEIGHT*1.05))

    # Panel (a): Conceptual view of exchangeability
    orderings = [
        ['A', 'B', 'C', 'D'],
        ['B', 'A', 'D', 'C'],
        ['D', 'C', 'B', 'A'],
        ['C', 'D', 'A', 'B']
    ]

    # Ideal Bayesian: same prediction regardless of order
    bayesian_preds = [0.65] * len(orderings)

    # Non-exchangeable: different predictions
    non_exch_preds = [0.65, 0.68, 0.61, 0.67]

    y_pos = np.arange(len(orderings))

    # Show as horizontal bars
    ax1.barh(y_pos - 0.18, bayesian_preds, 0.35,
             label='Ideal Bayesian', color=COLOR_PRIMARY, alpha=0.7, edgecolor='black')
    ax1.barh(y_pos + 0.18, non_exch_preds, 0.35,
             label='Order-sensitive model', color=COLOR_SECONDARY, alpha=0.7, edgecolor='black')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([''.join(o) for o in orderings], fontsize=8, family='monospace')
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Data ordering')
    ax1.set_title('(a) Prediction Under\nDifferent Orderings', pad=10)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim([0.55, 0.75])

    # Panel (b): Variance across orderings
    categories = ['Exchangeable\n(ideal)', 'Order-sensitive\n(transformer)']
    variances = [0.0, np.var(non_exch_preds)]

    bars = ax2.bar(categories, variances, color=[COLOR_PRIMARY, COLOR_SECONDARY],
                   alpha=0.7, edgecolor='black', width=0.5)

    # Add value labels with better spacing
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.00015,
                f'{var:.4f}', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Variance across orderings')
    ax2.set_title('(b) Exchangeability Gap', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 0.0012])

    # Add annotation with better positioning
    ax2.annotate('Exchangeability gap\n= departure from\nBayesian ideal',
                xy=(1, variances[1]), xytext=(0.5, 0.00085),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3))

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_4_exchangeability.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_4_exchangeability.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.4 saved")
    plt.close()

# ============================================================================
# FIGURE 2.5: I-projection vs M-projection
# ============================================================================

def figure_2_5():
    """Demonstrate mode-covering vs mode-seeking"""
    x = np.linspace(-6, 8, 1000)

    # True posterior: bimodal mixture
    p_true = 0.4 * stats.norm.pdf(x, loc=-2, scale=0.8) + \
             0.6 * stats.norm.pdf(x, loc=3, scale=0.8)

    # I-projection (moment matching, mode covering)
    q_inclusive = stats.norm.pdf(x, loc=0.8, scale=2.8)

    # M-projection (mode seeking)
    q_exclusive = stats.norm.pdf(x, loc=3, scale=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.6, FIGURE_HEIGHT*1.05))

    # Panel (a): I-projection
    ax1.plot(x, p_true, color='black', linewidth=2.5, linestyle='--',
             label='True posterior $p(\\theta \\mid D)$')
    ax1.plot(x, q_inclusive, color=COLOR_PRIMARY, linewidth=2.5,
             label='$q^*$ (I-projection)')
    ax1.fill_between(x, 0, q_inclusive, alpha=0.2, color=COLOR_PRIMARY)

    ax1.set_xlabel('$\\theta$')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) $\\min_q \\mathrm{KL}(p \\| q)$\n(I-projection)\nMode covering, honest uncertainty',
                  fontsize=10, pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.37])

    # Panel (b): M-projection
    ax2.plot(x, p_true, color='black', linewidth=2.5, linestyle='--',
             label='True posterior $p(\\theta \\mid D)$')
    ax2.plot(x, q_exclusive, color=COLOR_SECONDARY, linewidth=2.5,
             label='$q^*$ (M-projection)')
    ax2.fill_between(x, 0, q_exclusive, alpha=0.2, color=COLOR_SECONDARY)

    # Highlight the ignored mode with better positioning
    ax2.annotate('Ignored mode!', xy=(-2, 0.15), xytext=(-4.5, 0.30),
                arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=2),
                fontsize=9, color=COLOR_QUATERNARY,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax2.set_xlabel('$\\theta$')
    ax2.set_title('(b) $\\min_q \\mathrm{KL}(q \\| p)$\n(M-projection)\nMode seeking, overconfident',
                  fontsize=10, pad=10)
    # Move legend to bottom-right to avoid overlapping the callout box
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.37])

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_5_projections.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_5_projections.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.5 saved")
    plt.close()

# ============================================================================
# FIGURE 2.6: Coverage-Accuracy Tradeoff
# ============================================================================

def figure_2_6():
    """Show selective prediction tradeoff"""
    thresholds = np.linspace(0.5, 1.0, 100)

    # Coverage decreases as threshold increases
    coverage = (1.0 - thresholds) / 0.5
    coverage = np.clip(coverage, 0, 1)

    # Accuracy increases as threshold increases
    base_acc = 0.85
    max_acc = 0.98
    accuracy = base_acc + (max_acc - base_acc) * (1 - coverage)**0.6

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT*1.05))

    # Plot the curve
    ax.plot(coverage, accuracy, color=COLOR_PRIMARY, linewidth=3,
            label='Coverage-Accuracy frontier')

    # Mark operating points with better positioning
    operating_points = [
        (1.0, base_acc, 'No selection\n(predict all)', (0.78, 0.84)),
        (0.6, 0.95, 'Balanced', (0.72, 0.955)),
        (0.2, 0.97, 'High precision\n(abstain often)', (0.35, 0.973))
    ]

    for cov, acc, label, textpos in operating_points:
        idx = np.argmin(np.abs(coverage - cov))
        ax.plot(coverage[idx], accuracy[idx], 'o', markersize=10,
                color=COLOR_SECONDARY, zorder=5, markeredgecolor='black',
                markeredgewidth=1.5)

        ax.annotate(label, xy=(coverage[idx], accuracy[idx]), xytext=textpos,
                   fontsize=9, ha='center',
                   arrowprops=dict(arrowstyle='->', color='black', lw=1))

    ax.set_xlabel('Coverage (fraction of examples predicted)')
    ax.set_ylabel('Accuracy (on predicted examples)')
    ax.set_title('Selective Prediction: Coverage-Accuracy Tradeoff', pad=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0.82, 1.0])

    # Add shaded region
    ax.axhspan(0.82, base_acc, alpha=0.1, color='red',
               label='Below baseline')

    ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_6_coverage_accuracy.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_6_coverage_accuracy.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.6 saved")
    plt.close()

# ============================================================================
# FIGURE 2.7: Model Misspecification
# ============================================================================

def figure_2_7():
    """Show polynomial approximations to sine with confidence bands"""
    np.random.seed(42)

    # True function: sine
    x = np.linspace(0, 2*np.pi, 100)
    y_true = np.sin(x)

    # Generate noisy data
    n_data = 15
    x_data = np.linspace(0.5, 2*np.pi - 0.5, n_data)
    y_data = np.sin(x_data) + np.random.normal(0, 0.15, n_data)

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*1.9, FIGURE_HEIGHT*1.05))

    degrees = [3, 7, 15]
    titles = ['(a) Cubic: Underfit', '(b) Degree 7: Better', '(c) Degree 15: Overfit']

    for ax, degree, title in zip(axes, degrees, titles):
        # Fit polynomial
        coeffs = np.polyfit(x_data, y_data, degree)
        y_fit = np.polyval(coeffs, x)

        # Estimate uncertainty
        y_fit_data = np.polyval(coeffs, x_data)
        residuals = y_data - y_fit_data
        sigma = np.std(residuals)

        # Plot true function
        ax.plot(x, y_true, 'k--', linewidth=2, label='True: $\\sin(x)$', alpha=0.6)

        # Plot fit with confidence band
        ax.plot(x, y_fit, color=COLOR_PRIMARY, linewidth=2.5, label=f'Degree {degree} fit')
        ax.fill_between(x, y_fit - 2*sigma, y_fit + 2*sigma,
                        alpha=0.2, color=COLOR_PRIMARY, label='95\\% interval')

        # Plot data
        ax.scatter(x_data, y_data, color=COLOR_SECONDARY, s=40, zorder=5,
                  edgecolor='black', linewidth=0.5, label='Data')

        # Highlight extrapolation region for overfitting case
        if degree == 15:
            extrap_x = np.linspace(2*np.pi, 2.5*np.pi, 20)
            extrap_y = np.polyval(coeffs, extrap_x)
            ax.plot(extrap_x, extrap_y, color=COLOR_QUATERNARY, linewidth=2,
                   linestyle=':', label='Extrapolation')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(title, pad=10)
        # Keep legends off the most salient features
        legend_loc = 'upper left' if degree == 15 else 'upper right'
        ax.legend(loc=legend_loc, fontsize=7.5, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-2.2, 2.2])

        if degree == 15:
            ax.set_xlim([0, 2.5*np.pi])
        else:
            ax.set_xlim([0, 2*np.pi])

    plt.tight_layout(pad=1.5)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_7_misspecification.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_7_misspecification.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.7 saved")
    plt.close()

# ============================================================================
# FIGURE 2.8: Posterior Concentration as n Increases
# ============================================================================

def figure_2_8():
    """Show posterior narrowing with more data"""
    theta = np.linspace(0, 1, 1000)

    # Different sample sizes
    true_p = 0.65

    # Prior: Beta(1,1)
    prior = stats.beta.pdf(theta, 1, 1)

    # Posteriors with increasing data
    data_sizes = [10, 50, 200]
    colors = [COLOR_QUATERNARY, COLOR_SECONDARY, COLOR_PRIMARY]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT*1.05))

    # Plot prior
    ax.plot(theta, prior, color=COLOR_GRAY, linewidth=2, linestyle='--',
            label='Prior: $\\mathrm{Beta}(1,1)$')

    # Plot posteriors
    for n, color in zip(data_sizes, colors):
        n_heads = int(0.65 * n)
        n_tails = n - n_heads

        posterior = stats.beta.pdf(theta, 1+n_heads, 1+n_tails)

        label = f'$n={n}$: $\\mathrm{{Beta}}({1+n_heads},{1+n_tails})$'
        ax.plot(theta, posterior, color=color, linewidth=2.5, label=label)

    # Mark the true value
    ax.axvline(true_p, color='black', linestyle=':', linewidth=2, alpha=0.5)
    # Place label to the left of the vertical true-θ line so arrow doesn't cover it
    ax.text(true_p - 0.16, 15.8, f'True $\\theta = {true_p}$',
            fontsize=9, rotation=0, ha='right',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

    ax.set_xlabel('$\\theta$ (probability of heads)')
    ax.set_ylabel('Posterior density $p(\\theta \\mid D)$')
    ax.set_title('Posterior Concentration: More Data $\\Rightarrow$ Less Uncertainty', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 19])

    # Add annotation with better positioning
    ax.annotate('Posterior\nconcentrates\non truth',
                xy=(true_p, 16.5), xytext=(0.8, 12),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3))

    plt.tight_layout(pad=1.2)
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_8_posterior_concentration.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig('figures/ch02_bayesian_predictive/fig_2_8_posterior_concentration.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 2.8 saved")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Chapter 2 Figures: Bayesian Predictive Inference")
    print("=" * 60)
    print()

    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()
    figure_2_7()
    figure_2_8()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("Output location: figures/ch02_bayesian_predictive/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 60)
    print()
    print("Figure Summary:")
    print("  2.1: Prior to Posterior Update (Bayesian coin flip)")
    print("  2.2: Posterior Predictive vs Point Estimate")
    print("  2.3: Different Loss Functions → Different Decisions")
    print("  2.4: Exchangeability and Permutation Invariance")
    print("  2.5: I-projection vs M-projection")
    print("  2.6: Coverage-Accuracy Tradeoff (selective prediction)")
    print("  2.7: Model Misspecification (polynomials vs sine)")
    print("  2.8: Posterior Concentration with More Data")
