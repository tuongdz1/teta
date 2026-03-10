"""
Figure generation utilities for Chapter 8 (Variational Inference, Bits-Back, and Flows).
Produces the figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch, Ellipse
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.ndimage import gaussian_filter
import os

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# LaTeX font configuration for publication quality
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
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
COLOR_PURPLE = '#9467bd'

# Standard figure size for book (width in inches)
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0

# DPI for high quality
DPI = 300

# Create output directory (always relative to this script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures', 'ch08_variational_flows')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=0):
    """2D Gaussian with optional correlation."""
    sigma_matrix = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                            [rho * sigma_x * sigma_y, sigma_y**2]])
    inv_sigma = np.linalg.inv(sigma_matrix)
    det_sigma = np.linalg.det(sigma_matrix)

    dx = x - mu_x
    dy = y - mu_y

    exponent = -0.5 * (inv_sigma[0, 0] * dx**2 +
                       2 * inv_sigma[0, 1] * dx * dy +
                       inv_sigma[1, 1] * dy**2)

    return np.exp(exponent) / (2 * np.pi * np.sqrt(det_sigma))

# ============================================================================
# FIGURE 8.1: ELBO Decomposition and Gap
# ============================================================================

def figure_8_1():
    """Visualize ELBO as lower bound on log p(x)."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Create a conceptual diagram showing the relationship
    y_log_p = 1.0
    y_elbo = 0.6
    y_kl = y_log_p - y_elbo

    x_left = 0.2
    x_right = 0.8
    bar_width = 0.15

    # Log p(x) bar
    ax.barh([3], [x_right - x_left], left=x_left, height=0.3,
            color=COLOR_GRAY, alpha=0.5, edgecolor='black', linewidth=2,
            label='$\\log p_\\theta(x)$ (intractable)')
    ax.text((x_left + x_right)/2, 3, '$\\log p_\\theta(x)$',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # ELBO bar (shorter)
    elbo_width = (x_right - x_left) * 0.7
    ax.barh([2], [elbo_width], left=x_left, height=0.3,
            color=COLOR_PRIMARY, alpha=0.7, edgecolor='black', linewidth=2,
            label='ELBO (tractable lower bound)')
    ax.text(x_left + elbo_width/2, 2, 'ELBO',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white')

    # KL gap
    gap_start = x_left + elbo_width
    gap_width = (x_right - x_left) - elbo_width
    ax.barh([2], [gap_width], left=gap_start, height=0.3,
            color=COLOR_QUATERNARY, alpha=0.5, edgecolor='black',
            linewidth=2, linestyle='--')
    ax.text(gap_start + gap_width/2, 2, 'Gap',
            ha='center', va='center', fontsize=10)

    # Show the KL divergence annotation
    ax.annotate('', xy=(gap_start + gap_width/2, 2.35),
                xytext=(gap_start + gap_width/2, 2.65),
                arrowprops=dict(arrowstyle='<->', lw=2, color=COLOR_QUATERNARY))
    ax.text(gap_start + gap_width/2 + 0.05, 2.5,
            '$D_{\\mathrm{KL}}(q_\\phi(z|x) \\| p_\\theta(z|x))$',
            fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.4',
                     facecolor=COLOR_LIGHT_ORANGE, alpha=0.8))

    # Decomposition below (increase bar height so multi-line labels don't cramp)
    recon_width = elbo_width * 0.6
    kl_width = elbo_width * 0.4

    components_y = 1.1
    components_h = 0.32
    ax.barh([components_y], [recon_width], left=x_left, height=components_h,
            color=COLOR_TERTIARY, alpha=0.7, edgecolor='black', linewidth=1)
    ax.text(
        x_left + recon_width / 2,
        components_y,
        'Reconstruction\n$\\mathbb{E}_q[\\log p(x|z)]$',
        ha='center',
        va='center',
        fontsize=8.5,
    )

    ax.barh([components_y], [kl_width], left=x_left + recon_width, height=components_h,
            color=COLOR_SECONDARY, alpha=0.7, edgecolor='black', linewidth=1)
    ax.text(x_left + recon_width + kl_width/2, components_y,
            'Prior KL\n$-D_{\\mathrm{KL}}$',
            ha='center', va='center', fontsize=8.5)

    # Connect ELBO to decomposition
    ax.plot([x_left + elbo_width/2, x_left + elbo_width/2],
            [1.85, components_y + 0.2], 'k--', linewidth=1, alpha=0.5)

    # Labels
    ax.text(0.1, 3, 'Target:', ha='right', va='center', fontsize=10)
    ax.text(0.1, 2, 'Optimize:', ha='right', va='center', fontsize=10)
    ax.text(0.1, components_y, 'Components:', ha='right', va='center', fontsize=10)

    ax.set_xlim([0, 1])
    ax.set_ylim([0.6, 3.6])
    ax.axis('off')
    ax.set_title('The ELBO: A Tractable Lower Bound on Log-Likelihood',
                 fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_1_elbo_decomposition.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_1_elbo_decomposition.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.1 saved")
    plt.close()

# ============================================================================
# FIGURE 8.2: Reconstruction-Regularization Tradeoff (2D)
# ============================================================================

def figure_8_2():
    """Show the tension between reconstruction and KL regularization."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT))

    # Create grid
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    # Prior: standard Gaussian
    prior = stats.multivariate_normal.pdf(np.dstack([X, Y]), mean=[0, 0],
                                          cov=[[1, 0], [0, 1]])

    # True posterior (what we want): offset Gaussian
    true_posterior = stats.multivariate_normal.pdf(np.dstack([X, Y]),
                                                   mean=[2, 1],
                                                   cov=[[0.5, 0.2], [0.2, 0.8]])

    # Panel 1: Prior p(z)
    ax1.contour(X, Y, prior, levels=8, colors=COLOR_GRAY, alpha=0.6)
    ax1.contourf(X, Y, prior, levels=8, cmap='Greys', alpha=0.3)
    ax1.plot(0, 0, 'k*', markersize=15, label='Prior mean')
    ax1.set_xlabel('$z_1$')
    ax1.set_ylabel('$z_2$')
    ax1.set_title('(a) Prior $p(z) = \\mathcal{N}(0, I)$')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    ax1.set_aspect('equal')

    # Panel 2: True posterior p(z|x)
    ax2.contour(X, Y, true_posterior, levels=8, colors=COLOR_PRIMARY, alpha=0.8)
    ax2.contourf(X, Y, true_posterior, levels=8, cmap='Blues', alpha=0.4)
    ax2.plot(2, 1, 'b*', markersize=15, label='Posterior mean')
    # Also show prior faintly
    ax2.contour(X, Y, prior, levels=8, colors=COLOR_GRAY,
                alpha=0.3, linestyles='--')
    ax2.set_xlabel('$z_1$')
    ax2.set_title('(b) True Posterior $p(z|x)$')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_aspect('equal')

    # Panel 3: Show the tradeoff - FIXED with line styles for accessibility
    # Good reconstruction but high KL
    q_good_recon = stats.multivariate_normal.pdf(np.dstack([X, Y]),
                                                 mean=[2, 1],
                                                 cov=[[0.4, 0.15], [0.15, 0.7]])
    # Low KL but poor reconstruction
    q_low_kl = stats.multivariate_normal.pdf(np.dstack([X, Y]),
                                             mean=[0.5, 0.3],
                                             cov=[[0.9, 0], [0, 0.9]])

    ax3.contour(X, Y, true_posterior, levels=8, colors=COLOR_GRAY,
                alpha=0.3, linestyles='--', linewidths=1)
    ax3.contour(X, Y, prior, levels=8, colors=COLOR_GRAY,
                alpha=0.3, linestyles=':', linewidths=1)

    # FIXED: Add solid lines for green, dashed for red (colorblind friendly)
    ax3.contour(X, Y, q_good_recon, levels=5, colors=COLOR_TERTIARY,
                alpha=0.8, linewidths=2, linestyles='-')
    ax3.contour(X, Y, q_low_kl, levels=5, colors=COLOR_QUATERNARY,
                alpha=0.8, linewidths=2, linestyles='--')

    ax3.plot(2, 1, 'g*', markersize=12, label='Good recon, high KL (solid)')
    ax3.plot(0.5, 0.3, 'r*', markersize=12, label='Low KL, poor recon (dashed)')

    ax3.set_xlabel('$z_1$')
    ax3.set_title('(c) Approximate Posteriors $q_\\phi(z|x)$')
    ax3.legend(fontsize=8, loc='upper left')  # FIXED: moved to upper left
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-4, 4])
    ax3.set_ylim([-4, 4])
    ax3.set_aspect('equal')

    plt.suptitle('Reconstruction vs Regularization: The ELBO Tradeoff',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_2_recon_regularization.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_2_recon_regularization.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.2 saved")
    plt.close()

# ============================================================================
# FIGURE 8.3: Reparameterization Trick
# ============================================================================

def figure_8_3():
    """Visualize the reparameterization trick."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Panel 1: Naive approach (gradients don't flow)
    ax1.text(0.5, 0.9, 'Naive Sampling (No Gradient Flow)',
             ha='center', fontsize=11, fontweight='bold',
             transform=ax1.transAxes)

    # x -> encoder -> q_phi(z|x)
    ax1.add_patch(Rectangle((0.1, 0.65), 0.2, 0.15,
                           facecolor=COLOR_LIGHT_BLUE, edgecolor='black', linewidth=2))
    ax1.text(0.2, 0.725, '$x$', ha='center', va='center', fontsize=11)

    ax1.arrow(0.32, 0.725, 0.15, 0, head_width=0.03, head_length=0.02,
              fc='black', ec='black')

    ax1.add_patch(Ellipse((0.55, 0.725), 0.15, 0.15,
                         facecolor=COLOR_PRIMARY, alpha=0.7,
                         edgecolor='black', linewidth=2))
    ax1.text(0.55, 0.725, '$q_\\phi$', ha='center', va='center',
             fontsize=10, color='white', fontweight='bold')

    # Sample from q_phi
    ax1.arrow(0.63, 0.725, 0.15, 0, head_width=0.03, head_length=0.02,
              fc='black', ec='black', linestyle='--')
    ax1.text(0.705, 0.77, 'sample', ha='center', fontsize=8, style='italic')

    ax1.add_patch(Circle((0.85, 0.725), 0.05,
                        facecolor=COLOR_SECONDARY, edgecolor='black', linewidth=2))
    ax1.text(0.85, 0.725, '$z$', ha='center', va='center', fontsize=10)

    # No gradient flow
    ax1.plot([0.78, 0.78], [0.6, 0.48], 'r--', linewidth=2, alpha=0.7)
    ax1.text(0.78, 0.54, r'$\times$', ha='center', va='center', fontsize=20,
             color='red', fontweight='bold')
    ax1.text(0.5, 0.35, 'Gradient blocked by sampling!',
             ha='center', fontsize=9, color='red',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='red', linewidth=2))

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.axis('off')

    # Panel 2: Reparameterization (gradients flow)
    ax2.text(0.5, 0.9, 'Reparameterization (Gradient Flows)',
             ha='center', fontsize=11, fontweight='bold',
             transform=ax2.transAxes)

    # x -> encoder -> mu, sigma
    ax2.add_patch(Rectangle((0.1, 0.65), 0.2, 0.15,
                           facecolor=COLOR_LIGHT_BLUE, edgecolor='black', linewidth=2))
    ax2.text(0.2, 0.725, '$x$', ha='center', va='center', fontsize=11)

    ax2.arrow(0.32, 0.725, 0.13, 0, head_width=0.03, head_length=0.02,
              fc='black', ec='black')

    ax2.add_patch(Ellipse((0.53, 0.725), 0.15, 0.15,
                         facecolor=COLOR_PRIMARY, alpha=0.7,
                         edgecolor='black', linewidth=2))
    ax2.text(0.53, 0.725, '$q_\\phi$', ha='center', va='center',
             fontsize=10, color='white', fontweight='bold')

    # Output mu and sigma
    ax2.arrow(0.61, 0.75, 0.08, 0.08, head_width=0.02, head_length=0.015,
              fc='black', ec='black')
    ax2.text(0.73, 0.84, '$\\mu_\\phi(x)$', ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

    ax2.arrow(0.61, 0.70, 0.08, -0.08, head_width=0.02, head_length=0.015,
              fc='black', ec='black')
    ax2.text(0.73, 0.61, '$\\sigma_\\phi(x)$', ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

    # Epsilon (parameter-free noise)
    ax2.add_patch(Circle((0.73, 0.48), 0.04,
                        facecolor='white', edgecolor='black', linewidth=1.5))
    ax2.text(0.73, 0.48, '$\\epsilon$', ha='center', va='center', fontsize=9)
    ax2.text(0.73, 0.41, '$\\sim \\mathcal{N}(0,I)$', ha='center', fontsize=7)

    # Combine to get z
    ax2.arrow(0.73, 0.52, 0, 0.08, head_width=0.02, head_length=0.015,
              fc='black', ec='black')

    ax2.add_patch(Circle((0.85, 0.725), 0.06,
                        facecolor=COLOR_TERTIARY, alpha=0.7,
                        edgecolor='black', linewidth=2))
    ax2.text(0.85, 0.725, '$z$', ha='center', va='center', fontsize=10,
             fontweight='bold')

    ax2.text(0.79, 0.725, '$+$', ha='center', va='center', fontsize=12)
    ax2.text(0.79, 0.78, '$\\times$', ha='center', va='center', fontsize=10)

    # Show gradient flow
    ax2.annotate('', xy=(0.2, 0.6), xytext=(0.85, 0.67),
                arrowprops=dict(arrowstyle='->', lw=2.5,
                              color=COLOR_TERTIARY, alpha=0.7,
                              connectionstyle='arc3,rad=0.3'))
    ax2.text(0.5, 0.4, r'$\checkmark$ Gradients flow through deterministic path!',
             ha='center', fontsize=9, color=COLOR_TERTIARY,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=COLOR_TERTIARY, linewidth=2))

    # Formula
    ax2.text(0.5, 0.15, '$z = \\mu_\\phi(x) + \\sigma_\\phi(x) \\odot \\epsilon$',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_BLUE,
                      alpha=0.8, edgecolor='black', linewidth=1.5))

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_3_reparameterization.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_3_reparameterization.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.3 saved")
    plt.close()

# ============================================================================
# FIGURE 8.4: Bits-Back Coding Protocol
# ============================================================================

def figure_8_4():
    """Step-by-step visualization of bits-back coding."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.2, FIGURE_HEIGHT * 1.5))

    steps = [
        ('1. Sample $z \\sim q_\\phi(z|x)$', 'Consume bits from shared coder state',
         '$-\\log q_\\phi(z|x)$ bits', COLOR_PRIMARY),
        ('2. Send $z$ using prior', 'Entropy coding with $p(z)$',
         '$-\\log p(z)$ bits', COLOR_SECONDARY),
        ('3. Send $x$ given $z$', 'Entropy coding with $p(x|z)$',
         '$-\\log p(x|z)$ bits', COLOR_TERTIARY),
        ('4. Reclaim bits', 'Decode under $q_\\phi(z|x)$ to recover bits',
         '$+\\log q_\\phi(z|x)$ bits back', COLOR_QUATERNARY),
    ]

    y_start = 0.85
    y_step = 0.18

    for i, (step, desc, bits, color) in enumerate(steps):
        y = y_start - i * y_step

        # Step number and description
        ax.add_patch(FancyBboxPatch((0.05, y - 0.05), 0.9, 0.12,
                                   boxstyle='round,pad=0.01',
                                   facecolor=color, alpha=0.3,
                                   edgecolor=color, linewidth=2))

        ax.text(0.08, y + 0.03, step, fontsize=11, fontweight='bold',
                va='top')
        ax.text(0.08, y - 0.02, desc, fontsize=9, style='italic',
                va='top')

        # Cost annotation
        ax.text(0.88, y + 0.005, bits, fontsize=10, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', linewidth=1))

        # Arrow to next step (except last)
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y - 0.08), xytext=(0.5, y - 0.02),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Net cost calculation
    ax.add_patch(FancyBboxPatch((0.1, 0.02), 0.8, 0.08,
                               boxstyle='round,pad=0.01',
                               facecolor=COLOR_LIGHT_ORANGE, alpha=0.8,
                               edgecolor='black', linewidth=2.5))

    formula = ('Net cost: $-\\log p(z) - \\log p(x|z) + \\log q_\\phi(z|x)$\n'
              '$= -\\log p(x,z) + \\log q_\\phi(z|x) = -\\mathrm{ELBO}$')
    ax.text(0.5, 0.06, formula, fontsize=11, ha='center', va='center',
            fontweight='bold')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    ax.set_title('Bits-Back Coding: Why the ELBO is Exactly Right',
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_4_bitsback_protocol.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_4_bitsback_protocol.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.4 saved")
    plt.close()

# ============================================================================
# FIGURE 8.5: Three Gaps Visualization - FIXED
# ============================================================================

def figure_8_5():
    """Show approximation, amortization, and optimization gaps."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT * 1.2))

    # Define the hierarchy of objectives
    y_positions = [0.85, 0.65, 0.45, 0.25, 0.05]
    labels = [
        'True log-likelihood\n$\\log p_\\theta(x)$',
        'Best in family\n$\\max_{q \\in \\mathcal{Q}} \\mathrm{ELBO}(q, \\theta; x)$',
        'Best amortized\n$\\max_\\phi \\mathrm{ELBO}(q_\\phi, \\theta; x)$',
        'Optimal training\n$\\max_{\\phi,\\theta} \\mathrm{ELBO}(q_\\phi, \\theta; x)$',
        'Actual result\n$\\mathrm{ELBO}(q_\\phi^{\\mathrm{trained}}, \\theta^{\\mathrm{trained}}; x)$',
    ]

    values = [1.0, 0.85, 0.72, 0.65, 0.58]
    colors = [COLOR_GRAY, COLOR_PRIMARY, COLOR_SECONDARY,
              COLOR_TERTIARY, COLOR_QUATERNARY]

    label_x = 1.05
    for i, (y, label, value, color) in enumerate(zip(y_positions, labels, values, colors)):
        # Bar
        ax.barh([y], [value], height=0.12, left=0,
               color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Label: use a fixed x position so gap callouts don't collide
        ax.text(label_x, y, label, fontsize=9, va='center', ha='left', linespacing=1.1)

        # Gap annotations - FIXED: distinguish the two optimization gaps
        if i > 0:
            gap_y = (y_positions[i-1] + y) / 2
            gap_size = values[i-1] - values[i]

            # Arrow showing gap
            ax.annotate('', xy=(values[i], gap_y + 0.04),
                       xytext=(values[i-1], gap_y + 0.04),
                       arrowprops=dict(arrowstyle='<->', lw=2,
                                     color='red', alpha=0.7))

            # Gap label - FIXED: last gap is now "Training Gap"
            if i == 1:
                gap_name = 'Approximation\nGap'
            elif i == 2:
                gap_name = 'Amortization\nGap'
            elif i == 3:
                gap_name = 'Optimization\nGap'
            else:
                gap_name = 'Training Gap\n(finite sample)'

            ax.text((values[i-1] + values[i])/2, gap_y - 0.05,
                   gap_name, ha='center', fontsize=9,
                   color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.9,
                            edgecolor='red', linewidth=1))

    ax.set_xlim([0, 1.35])
    ax.set_ylim([-0.05, 0.95])
    ax.set_xlabel('Objective Value (higher is better)', fontsize=10)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_title('Three Sources of Suboptimality in VAE Training',
                 fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_5_three_gaps.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_5_three_gaps.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.5 saved")
    plt.close()

# ============================================================================
# FIGURE 8.6: Coupling Layer Transformation
# ============================================================================

def figure_8_6():
    """Visualize how a coupling layer transforms space."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                        figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT))

    # Create a grid in input space
    x = np.linspace(0.5, 3.5, 7)
    y = np.linspace(0.5, 2.5, 5)
    X, Y = np.meshgrid(x, y)

    # Panel 1: Input space
    for i in range(len(x)):
        ax1.plot(X[:, i], Y[:, i], 'b-', alpha=0.5, linewidth=1)
    for j in range(len(y)):
        ax1.plot(X[j, :], Y[j, :], 'b-', alpha=0.5, linewidth=1)

    ax1.set_xlabel('$x_a$ (unchanged)', fontsize=10)
    ax1.set_ylabel('$x_b$ (to transform)', fontsize=10)
    ax1.set_title('(a) Input Space', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 4])
    ax1.set_ylim([0, 3])
    ax1.set_aspect('equal')

    # Panel 2: Coupling transformation
    # z_a = x_a (unchanged)
    # z_b = x_b * exp(s(x_a)) + t(x_a)
    # Simple example: s(x_a) = 0.3*x_a, t(x_a) = 0.5*x_a

    def coupling_transform(x_a, x_b):
        s = 0.3 * x_a
        t = 0.5 * x_a
        z_a = x_a
        z_b = x_b * np.exp(s) + t
        return z_a, z_b

    Z_A, Z_B = coupling_transform(X, Y)

    for i in range(len(x)):
        ax2.plot(Z_A[:, i], Z_B[:, i], 'g-', alpha=0.5, linewidth=1)
    for j in range(len(y)):
        ax2.plot(Z_A[j, :], Z_B[j, :], 'g-', alpha=0.5, linewidth=1)

    ax2.set_xlabel('$z_a = x_a$', fontsize=10)
    ax2.set_ylabel('$z_b = x_b \\cdot e^{s(x_a)} + t(x_a)$', fontsize=10)
    ax2.set_title('(b) After Coupling Layer', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 4])
    ax2.set_ylim([0, 8])

    # Add annotation
    ax2.text(2, 7, 'Vertical lines transformed\nHorizontal unchanged',
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_LIGHT_BLUE,
                     alpha=0.8))

    # Panel 3: Jacobian structure
    ax3.text(0.5, 0.85, 'Jacobian Structure', ha='center', fontsize=11,
            fontweight='bold', transform=ax3.transAxes)

    # Draw block matrix
    ax3.add_patch(Rectangle((0.2, 0.4), 0.25, 0.25,
                           facecolor='white', edgecolor='black', linewidth=2))
    ax3.text(0.325, 0.525, '$I$', ha='center', va='center', fontsize=14)

    ax3.add_patch(Rectangle((0.5, 0.4), 0.25, 0.25,
                           facecolor=COLOR_LIGHT_BLUE, edgecolor='black',
                           linewidth=2, alpha=0.3))
    ax3.text(0.625, 0.525, '$0$', ha='center', va='center', fontsize=14)

    ax3.add_patch(Rectangle((0.2, 0.1), 0.25, 0.25,
                           facecolor=COLOR_LIGHT_ORANGE, edgecolor='black',
                           linewidth=2, alpha=0.5))
    ax3.text(0.325, 0.225, '$\\ast$', ha='center', va='center', fontsize=14)

    ax3.add_patch(Rectangle((0.5, 0.1), 0.25, 0.25,
                           facecolor=COLOR_TERTIARY, edgecolor='black',
                           linewidth=3, alpha=0.6))
    ax3.text(0.625, 0.225, '$\\mathrm{diag}$', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Clean axis labels for the Jacobian blocks (avoid tiny, overlapping text)
    label_bbox = dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.9, edgecolor='none')
    ax3.text(0.325, 0.70, '$x_a$', ha='center', va='center', fontsize=12, bbox=label_bbox)
    ax3.text(0.625, 0.70, '$x_b$', ha='center', va='center', fontsize=12, bbox=label_bbox)
    ax3.text(0.09, 0.525, '$z_a$', ha='center', va='center', fontsize=12,
            bbox=label_bbox, rotation=90)
    ax3.text(0.09, 0.225, '$z_b$', ha='center', va='center', fontsize=12,
            bbox=label_bbox, rotation=90)

    # Formula for log det (note complexity inline to avoid extra callouts)
    formula = '$\\log|\\det J| = \\sum_j s_j(x_a) \\quad (\\mathcal{O}(d))$'
    ax3.text(0.5, 0.75, formula, ha='center', fontsize=11,
            transform=ax3.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=1.5))

    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.axis('off')

    plt.suptitle('Coupling Layers: Efficient Invertible Transformations',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_6_coupling_layer.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_6_coupling_layer.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.6 saved")
    plt.close()

# ============================================================================
# FIGURE 8.7: Change of Variables (Density Transformation)
# ============================================================================

def figure_8_7():
    """Show how flows transform densities via change of variables."""
    np.random.seed(42)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                        figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT))

    # Generate base samples
    n_samples = 1000
    z = np.random.randn(n_samples, 2)

    # Base distribution (standard Gaussian)
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z_base = stats.multivariate_normal.pdf(np.dstack([X, Y]),
                                           mean=[0, 0],
                                           cov=[[1, 0], [0, 1]])

    # Panel 1: Base distribution
    ax1.contourf(X, Y, Z_base, levels=15, cmap='Blues', alpha=0.6)
    ax1.contour(X, Y, Z_base, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax1.scatter(z[:100, 0], z[:100, 1], s=10, alpha=0.5, color='blue')
    ax1.set_xlabel('$z_1$', fontsize=10)
    ax1.set_ylabel('$z_2$', fontsize=10)
    ax1.set_title('(a) Base $p_0(z) = \\mathcal{N}(0,I)$', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_aspect('equal')

    # Panel 2: Show the transformation
    ax2.text(0.5, 0.9, '(b) Invertible Map $f$', ha='center', fontsize=11,
            fontweight='bold', transform=ax2.transAxes)

    # Draw flow diagram
    ax2.add_patch(Circle((0.25, 0.5), 0.12, facecolor=COLOR_LIGHT_BLUE,
                        edgecolor='black', linewidth=2,
                        transform=ax2.transAxes))
    ax2.text(0.25, 0.5, '$z$', ha='center', va='center', fontsize=14,
            transform=ax2.transAxes)

    ax2.annotate('', xy=(0.55, 0.5), xytext=(0.38, 0.5),
                transform=ax2.transAxes,
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax2.text(0.465, 0.56, '$f$', ha='center', fontsize=11,
            transform=ax2.transAxes, fontweight='bold')

    ax2.add_patch(Circle((0.75, 0.5), 0.12, facecolor=COLOR_LIGHT_ORANGE,
                        edgecolor='black', linewidth=2,
                        transform=ax2.transAxes))
    ax2.text(0.75, 0.5, '$x$', ha='center', va='center', fontsize=14,
            transform=ax2.transAxes)

    # Formula
    formula = ('$\\log p(x) = \\log p_0(f^{-1}(x))$\n'
              '$+ \\log|\\det J_{f^{-1}}(x)|$')
    ax2.text(0.5, 0.25, formula, ha='center', fontsize=10,
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=1.5))

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.axis('off')

    # Panel 3: Transformed distribution
    # Apply a simple nonlinear transformation
    def transform(z):
        x1 = z[:, 0]
        x2 = z[:, 1] + 0.5 * z[:, 0]**2  # Nonlinear coupling
        return np.column_stack([x1, x2])

    x = transform(z)

    # Compute transformed density approximately via samples
    from scipy.stats import gaussian_kde
    if len(x) > 0:
        kde = gaussian_kde(x.T)
        Z_transformed = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    ax3.contourf(X, Y, Z_transformed, levels=15, cmap='Oranges', alpha=0.6)
    ax3.contour(X, Y, Z_transformed, levels=10, colors='black',
                alpha=0.3, linewidths=0.5)
    ax3.scatter(x[:100, 0], x[:100, 1], s=10, alpha=0.5, color='orange')
    ax3.set_xlabel('$x_1$', fontsize=10)
    ax3.set_ylabel('$x_2$', fontsize=10)
    ax3.set_title('(c) Transformed $p(x) = f_\\sharp p_0$', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-3, 3])
    ax3.set_ylim([-3, 3])
    ax3.set_aspect('equal')

    plt.suptitle('Change of Variables: How Flows Transform Distributions',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_7_change_of_variables.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_7_change_of_variables.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.7 saved")
    plt.close()

# ============================================================================
# FIGURE 8.8: VAE vs Flow Architecture Comparison
# ============================================================================

def figure_8_8():
    """Compare VAE and flow architectures."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH * 1.35, FIGURE_HEIGHT * 0.9)
    )

    # Local styling for a quieter, more consistent visual language
    text_color = '#1f1f1f'
    muted_text = '#666666'
    line_color = '#3a3a3a'
    good_color = '#4F7A5C'
    bad_color = '#9B4F4F'

    def setup_ax(ax):
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')

    def add_node(ax, center, label, width, height, facecolor,
                 text_col=text_color, alpha=1.0, subtext=None):
        patch = FancyBboxPatch(
            (center[0] - width / 2, center[1] - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=line_color,
            facecolor=facecolor,
            alpha=alpha,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            center[0],
            center[1],
            label,
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            color=text_col,
            transform=ax.transAxes,
        )
        if subtext:
            ax.text(
                center[0],
                center[1] - height / 2 - 0.045,
                subtext,
                ha='center',
                va='top',
                fontsize=8,
                color=muted_text,
                transform=ax.transAxes,
            )

    def add_arrow(ax, start, end):
        ax.annotate(
            '',
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle='-|>', lw=1.2, color=line_color),
        )

    def add_bullets(ax, items, start_y):
        icon_x = 0.08
        text_x = 0.12
        step = 0.085
        for i, (kind, text, emphasis) in enumerate(items):
            y = start_y - i * step
            if kind == 'check':
                ax.text(icon_x, y, r'$\checkmark$', color=good_color,
                        fontsize=9, ha='left', va='center', transform=ax.transAxes)
            elif kind == 'cross':
                ax.text(icon_x, y, r'$\times$', color=bad_color,
                        fontsize=9, ha='left', va='center', transform=ax.transAxes)
            ax.text(text_x, y, text, color=text_color, fontsize=9,
                    fontweight='bold' if emphasis else 'normal',
                    ha='left', va='center', transform=ax.transAxes)

    for ax in (ax1, ax2):
        setup_ax(ax)

    # Panel 1: VAE
    ax1.text(0.5, 0.93, 'Variational Autoencoder (VAE)', ha='center',
             fontsize=11, fontweight='bold', color=text_color,
             transform=ax1.transAxes)

    y_diag = 0.68
    node_w = 0.20
    node_h = 0.12
    x_in = 0.15
    x_latent = 0.50
    x_out = 0.85

    add_node(ax1, (x_in, y_diag), '$x$', node_w, node_h, COLOR_LIGHT_BLUE)
    add_node(ax1, (x_latent, y_diag), '$z$', node_w, node_h, COLOR_PRIMARY,
             text_col='white', subtext='low-dim bottleneck')
    add_node(ax1, (x_out, y_diag), '$\\hat{x}$', node_w, node_h, COLOR_LIGHT_ORANGE)

    add_arrow(ax1, (x_in + node_w / 2, y_diag), (x_latent - node_w / 2, y_diag))
    ax1.text((x_in + x_latent) / 2, y_diag + 0.09, 'encoder', ha='center',
             fontsize=8, style='italic', color=muted_text, transform=ax1.transAxes)

    add_arrow(ax1, (x_latent + node_w / 2, y_diag), (x_out - node_w / 2, y_diag))
    ax1.text((x_latent + x_out) / 2, y_diag + 0.09, 'decoder', ha='center',
             fontsize=8, style='italic', color=muted_text, transform=ax1.transAxes)

    ax1.hlines(0.50, 0.06, 0.94, transform=ax1.transAxes,
               colors=muted_text, linewidth=0.6, alpha=0.4)

    properties_vae = [
        ('check', 'Compressed representation', False),
        ('check', 'Controllable generation', False),
        ('cross', 'Not exact likelihood', False),
        ('cross', 'Lossy bottleneck', False),
        ('', r'Objective: $\mathrm{ELBO}$', True),
    ]
    add_bullets(ax1, properties_vae, start_y=0.42)

    # Panel 2: Normalizing Flow
    ax2.text(0.5, 0.93, 'Normalizing Flow', ha='center',
             fontsize=11, fontweight='bold', color=text_color,
             transform=ax2.transAxes)

    flow_w = 0.11
    flow_h = 0.10
    x_flow_in = 0.15
    x_flow_layers = [0.40, 0.55, 0.70]
    x_flow_out = 0.88

    add_node(ax2, (x_flow_in, y_diag), '$z$', node_w, node_h, 'white',
             subtext=r'$\mathcal{N}(0, I)$')
    for x_pos, label in zip(x_flow_layers, ['$f_1$', '$f_2$', '$f_3$']):
        add_node(ax2, (x_pos, y_diag), label, flow_w, flow_h,
                 COLOR_LIGHT_BLUE, alpha=0.85)

    add_node(ax2, (x_flow_out, y_diag), '$x$', node_w, node_h,
             COLOR_LIGHT_ORANGE, subtext='same dim')

    add_arrow(ax2, (x_flow_in + node_w / 2, y_diag),
              (x_flow_layers[0] - flow_w / 2, y_diag))
    for left, right in zip(x_flow_layers[:-1], x_flow_layers[1:]):
        add_arrow(ax2, (left + flow_w / 2, y_diag),
                  (right - flow_w / 2, y_diag))
    add_arrow(ax2, (x_flow_layers[-1] + flow_w / 2, y_diag),
              (x_flow_out - node_w / 2, y_diag))

    ax2.text(0.55, 0.58, 'invertible layers', ha='center', fontsize=8.5,
             color=muted_text, transform=ax2.transAxes)

    ax2.hlines(0.50, 0.06, 0.94, transform=ax2.transAxes,
               colors=muted_text, linewidth=0.6, alpha=0.4)

    properties_flow = [
        ('check', 'Exact likelihood', False),
        ('check', 'Invertible (encode/decode)', False),
        ('cross', 'No dimensionality reduction', False),
        ('cross', 'Architectural constraints', False),
        ('', r'Objective: $\max \log p(x)$', True),
    ]
    add_bullets(ax2, properties_flow, start_y=0.42)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08, wspace=0.12)
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_8_vae_vs_flow.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_8_vae_vs_flow.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.8 saved")
    plt.close()

# ============================================================================
# FIGURE 8.9: Posterior Collapse Over Training - FIXED
# ============================================================================

def figure_8_9():
    """Show what posterior collapse looks like during training."""
    np.random.seed(42)

    # Simulate training curves
    epochs = np.linspace(0, 100, 200)

    # Healthy training
    recon_healthy = 200 * np.exp(-epochs/30) + 50
    kl_healthy = 5 + 10 * (1 - np.exp(-epochs/20))
    elbo_healthy = -(recon_healthy + kl_healthy)

    # Collapsed training
    recon_collapsed = 200 * np.exp(-epochs/25) + 45
    kl_collapsed = 15 * np.exp(-epochs/15) + 0.1  # KL goes to zero
    elbo_collapsed = -(recon_collapsed + kl_collapsed)

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT * 1.4))

    # Panel 1: Reconstruction loss
    # FIXED: Add line styles for accessibility
    axes[0, 0].plot(epochs, recon_healthy, color=COLOR_TERTIARY, linewidth=2.5,
                    linestyle='-', label='Healthy training (solid)')
    axes[0, 0].plot(epochs, recon_collapsed, color=COLOR_QUATERNARY,
                    linewidth=2.5, linestyle='--', label='Collapsed (dashed)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reconstruction Loss')
    axes[0, 0].set_title('(a) Reconstruction: $-\\mathbb{E}_q[\\log p(x|z)]$')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 250])

    # Panel 2: KL divergence - FIXED: line styles
    axes[0, 1].plot(epochs, kl_healthy, color=COLOR_TERTIARY, linewidth=2.5,
                    linestyle='-', label='Healthy: KL stabilizes (solid)')
    axes[0, 1].plot(epochs, kl_collapsed, color=COLOR_QUATERNARY,
                    linewidth=2.5, linestyle='--',
                    label='Collapsed: KL $\\to$ 0 (dashed)')
    axes[0, 1].axhline(0.5, color='gray', linestyle=':', linewidth=1.5,
                      label='Collapse threshold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('(b) Prior Matching: $D_{\\mathrm{KL}}(q \\| p)$')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 20])

    # Add warning annotation
    axes[0, 1].text(70, 2, r'$\mathbf{Warning}$' + '\nPosterior Collapse!',
                   ha='center', fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='red', linewidth=2))

    # Panel 3: ELBO - FIXED: line styles
    axes[1, 0].plot(epochs, elbo_healthy, color=COLOR_TERTIARY, linewidth=2.5,
                    linestyle='-', label='Healthy (solid)')
    axes[1, 0].plot(epochs, elbo_collapsed, color=COLOR_QUATERNARY,
                    linewidth=2.5, linestyle='--', label='Collapsed (dashed)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ELBO (higher is better)')
    axes[1, 0].set_title('(c) Overall Objective')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Diagnostic - KL per dimension
    axes[1, 1].set_title(
        r'(d) Diagnostic: Active Dimensions'
        '\n'
        r'Per-dimension KL at epoch 100 (active if $\mathrm{KL} > 0.5$ nats)',
        fontsize=10,
    )

    dims = np.arange(1, 11)
    kl_per_dim_healthy = np.random.uniform(0.5, 2.5, 10)
    kl_per_dim_collapsed = np.random.uniform(0, 0.05, 10)

    x_pos = np.arange(len(dims))
    width = 0.35

    axes[1, 1].bar(x_pos - width/2, kl_per_dim_healthy, width,
                   color=COLOR_TERTIARY, alpha=0.7, label='Healthy',
                   edgecolor='black', linewidth=1)
    axes[1, 1].bar(x_pos + width/2, kl_per_dim_collapsed, width,
                   color=COLOR_QUATERNARY, alpha=0.7, label='Collapsed',
                   edgecolor='black', linewidth=1)

    axes[1, 1].axhline(0.5, color='gray', linestyle='--', linewidth=1,
                      alpha=0.7, label='Active threshold')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('KL (nats)')
    # Title already set above (kept outside plot area to avoid collisions)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f'$z_{i}$' for i in dims])
    # Keep legend inside but away from the highest bars to avoid occlusion.
    axes[1, 1].legend(fontsize=8, loc='upper center', ncol=2, framealpha=0.9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Posterior Collapse: Detection and Diagnosis',
                 fontsize=12, fontweight='bold', y=0.99)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_9_posterior_collapse.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_9_posterior_collapse.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.9 saved")
    plt.close()

# ============================================================================
# FIGURE 8.10: β-VAE Rate-Distortion Curve - FIXED
# ============================================================================

def figure_8_10():
    """Show how β controls the rate-distortion tradeoff in VAEs."""
    # Generate rate-distortion curve
    D = np.linspace(0.1, 2.5, 100)
    R = 3 * np.exp(-D * 0.8) + 0.2

    # β-VAE operating points
    betas = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_points = {
        0.1: (2.2, 3 * np.exp(-2.2 * 0.8) + 0.2),
        0.5: (1.5, 3 * np.exp(-1.5 * 0.8) + 0.2),
        1.0: (0.8, 3 * np.exp(-0.8 * 0.8) + 0.2),
        2.0: (0.4, 3 * np.exp(-0.4 * 0.8) + 0.2),
        5.0: (0.15, 3 * np.exp(-0.15 * 0.8) + 0.2),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Panel 1: Rate-distortion curve
    ax1.plot(D, R, color=COLOR_GRAY, linewidth=2.5, linestyle='--',
            label='Optimal R(D) frontier')

    colors = [COLOR_QUATERNARY, COLOR_SECONDARY, COLOR_PRIMARY,
             COLOR_TERTIARY, COLOR_PURPLE]

    for beta, color in zip(betas, colors):
        d, r = beta_points[beta]
        marker = 'o'
        size = 120 if beta == 1.0 else 80
        label = f'$\\beta={beta}$' + (' (standard)' if beta == 1.0 else '')

        ax1.scatter(d, r, s=size, color=color, marker=marker,
                   edgecolor='black', linewidth=2, label=label, zorder=5)

        # Add annotations for extremes
        if beta == 0.1:
            ax1.annotate('Low compression\nHigh fidelity\n(overfitting risk)',
                        xy=(d, r), xytext=(1.75, 0.35),
                        fontsize=8, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 alpha=0.85, edgecolor='none'),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
        elif beta == 5.0:
            ax1.annotate('High compression\nLow fidelity\n(underfitting)',
                        xy=(d, r), xytext=(d + 0.2, r - 0.4),
                        fontsize=8, ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 alpha=0.85, edgecolor='none'),
                        arrowprops=dict(arrowstyle='->', lw=1.5))

    ax1.set_xlabel('Distortion $D = -\\mathbb{E}_q[\\log p(x|z)]$', fontsize=10)
    ax1.set_ylabel('Rate $R = D_{\\mathrm{KL}}(q \\| p)$', fontsize=10)
    ax1.set_title('(a) $\\beta$-VAE Operating Points', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 2.7])
    ax1.set_ylim([0, 3])

    # Panel 2: What β controls - FIXED: larger text in info box
    ax2.text(0.5, 0.95, 'Effect of $\\beta$ on Objective',
            ha='center', fontsize=11, fontweight='bold',
            transform=ax2.transAxes)

    # Show the objective
    formula = ('$\\mathcal{L}_{\\beta\\mathrm{-VAE}} = '
              '\\mathbb{E}_q[\\log p(x|z)] - \\beta \\cdot D_{\\mathrm{KL}}(q \\| p)$')
    ax2.text(0.5, 0.85, formula, ha='center', fontsize=10,
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_BLUE,
                     alpha=0.8, edgecolor='black', linewidth=1.5))

    # Interpretation table
    table_data = [
        ('$\\beta < 1$', 'Prioritize reconstruction', 'High fidelity'),
        ('$\\beta = 1$', 'Standard VAE (ELBO)', 'Balanced'),
        ('$\\beta > 1$', 'Prioritize compression', 'Disentanglement'),
    ]

    y_start = 0.65
    for i, (beta_val, meaning, result) in enumerate(table_data):
        y = y_start - i * 0.18

        # Row background
        color = COLOR_LIGHT_BLUE if i == 1 else 'white'
        ax2.add_patch(Rectangle((0.05, y - 0.06), 0.9, 0.12,
                               facecolor=color, edgecolor='black',
                               linewidth=1, alpha=0.3,
                               transform=ax2.transAxes))

        ax2.text(0.15, y, beta_val, ha='center', va='center', fontsize=10,
                transform=ax2.transAxes, fontweight='bold')
        ax2.text(0.45, y, meaning, ha='center', va='center', fontsize=9,
                transform=ax2.transAxes)
        ax2.text(0.78, y, result, ha='center', va='center', fontsize=9,
                transform=ax2.transAxes, style='italic')

    # FIXED: Add information bottleneck connection with larger text
    ax2.text(0.5, 0.08, 'Connection to Information Bottleneck:\n'
                       '$\\beta$ controls how much to compress $X$ into $Z$',
            ha='center', fontsize=9, transform=ax2.transAxes,  # Increased from 8 to 9
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_ORANGE,
                     alpha=0.7, edgecolor='black', linewidth=1.5))

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_10_beta_vae.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_10_beta_vae.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.10 saved")
    plt.close()

# ============================================================================
# FIGURE 8.11: I-projection vs M-projection
# ============================================================================

def figure_8_11():
    """Geometric view of I-projection vs M-projection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Create true distribution (bimodal)
    x = np.linspace(-5, 5, 200)
    p_true = (0.4 * stats.norm.pdf(x, -2, 0.6) +
             0.6 * stats.norm.pdf(x, 2, 0.7))

    # Convention (common in the VI vs EP discussion):
    #   I-projection: min KL(q || p) (reverse/exclusive KL) – tends to be mode-seeking
    #   M-projection: min KL(p || q) (forward/inclusive KL) – tends to be mode-covering
    q_i_proj = stats.norm.pdf(x, 2, 0.7)    # Picks the dominant mode
    q_m_proj = stats.norm.pdf(x, 0.4, 2.2)  # Broadly covers both modes

    y_max = 1.1 * max(p_true.max(), q_i_proj.max(), q_m_proj.max())

    # Panel 1: I-projection (what VAEs do)
    ax1.fill_between(x, 0, p_true, alpha=0.3, color=COLOR_GRAY,
                     label='True $p(z|x)$ (bimodal)')
    ax1.plot(x, p_true, color='black', linewidth=2.5, linestyle='--')

    ax1.fill_between(x, 0, q_i_proj, alpha=0.4, color=COLOR_PRIMARY)
    ax1.plot(x, q_i_proj, color=COLOR_PRIMARY, linewidth=3,
            label='$q^*(z|x)$ (I-projection)')

    ax1.set_xlabel('$z$', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('(a) I-Projection: $\\min_q D_{\\mathrm{KL}}(q \\| p)$\n'
                 '(Variational Inference)', fontsize=10)
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, y_max])

    # Add properties
    props = [
        r'$\checkmark$ Mode seeking',
        r'$\times$ Misses modes',
        r'$\times$ Underestimates uncertainty',
    ]
    for i, prop in enumerate(props):
        color = COLOR_TERTIARY if r'\checkmark' in prop else COLOR_QUATERNARY
        ax1.text(0.05, 0.85 - i*0.08, prop, transform=ax1.transAxes,
                fontsize=9, color=color)

    # Panel 2: M-projection (for comparison)
    ax2.fill_between(x, 0, p_true, alpha=0.3, color=COLOR_GRAY,
                     label='True $p(z|x)$ (bimodal)')
    ax2.plot(x, p_true, color='black', linewidth=2.5, linestyle='--')

    ax2.fill_between(x, 0, q_m_proj, alpha=0.4, color=COLOR_SECONDARY)
    ax2.plot(x, q_m_proj, color=COLOR_SECONDARY, linewidth=3,
            label='$q^*(z|x)$ (M-projection)')

    ax2.set_xlabel('$z$', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('(b) M-Projection: $\\min_q D_{\\mathrm{KL}}(p \\| q)$\n'
                 '(Not used in VAEs)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, y_max])

    # Add properties
    props = [
        r'$\checkmark$ Moment matching',
        r'$\checkmark$ Covers both modes',
        r'$\times$ May be too broad',
    ]
    for i, prop in enumerate(props):
        color = COLOR_TERTIARY if r'\checkmark' in prop else COLOR_QUATERNARY
        ax2.text(0.05, 0.85 - i*0.08, prop, transform=ax2.transAxes,
                fontsize=9, color=color)

    plt.suptitle('KL Divergence Direction: I-projection (VAEs) vs M-projection',
                 fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_11_i_vs_m_projection.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_11_i_vs_m_projection.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.11 saved")
    plt.close()

# ============================================================================
# FIGURE 8.12: Latent Space Learned Structure - FIXED
# ============================================================================

def figure_8_12():
    """Show what a well-trained VAE learns in latent space."""
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Simulate learned latent space with structure
    # Generate synthetic data with classes
    n_per_class = 100

    # Class 1: e.g., "digit 0"
    z1 = np.random.multivariate_normal([-2, -1.5], [[0.3, 0.1], [0.1, 0.3]],
                                       n_per_class)
    # Class 2: e.g., "digit 1"
    z2 = np.random.multivariate_normal([0, 2], [[0.4, 0], [0, 0.2]],
                                       n_per_class)
    # Class 3: e.g., "digit 2"
    z3 = np.random.multivariate_normal([2.5, 0], [[0.25, 0.15], [0.15, 0.35]],
                                       n_per_class)
    # Class 4: e.g., "digit 3"
    z4 = np.random.multivariate_normal([0, -2], [[0.3, -0.1], [-0.1, 0.25]],
                                       n_per_class)

    # Panel 1: Learned latent space
    ax1.scatter(z1[:, 0], z1[:, 1], s=30, alpha=0.6,
               color=COLOR_PRIMARY, label='Class 0', edgecolors='black', linewidth=0.5)
    ax1.scatter(z2[:, 0], z2[:, 1], s=30, alpha=0.6,
               color=COLOR_SECONDARY, label='Class 1', edgecolors='black', linewidth=0.5)
    ax1.scatter(z3[:, 0], z3[:, 1], s=30, alpha=0.6,
               color=COLOR_TERTIARY, label='Class 2', edgecolors='black', linewidth=0.5)
    ax1.scatter(z4[:, 0], z4[:, 1], s=30, alpha=0.6,
               color=COLOR_QUATERNARY, label='Class 3', edgecolors='black', linewidth=0.5)

    # Show prior as contours
    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    prior = stats.multivariate_normal.pdf(np.dstack([X, Y]), mean=[0, 0],
                                         cov=[[1, 0], [0, 1]])
    ax1.contour(X, Y, prior, levels=5, colors='gray', alpha=0.3, linewidths=1)

    ax1.set_xlabel('Latent dimension $z_1$', fontsize=10)
    ax1.set_ylabel('Latent dimension $z_2$', fontsize=10)
    ax1.set_title('(a) Well-Trained VAE Latent Space', fontsize=11)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    ax1.set_aspect('equal')

    # Add annotations
    ax1.text(0.95, 0.95, 'Key properties:\n'
                        '• Clusters by class\n'
                        '• Smooth transitions\n'
                        '• Respects prior mass',
            transform=ax1.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     alpha=0.9, edgecolor='black', linewidth=1))

    # Panel 2: Show interpolation paths - FIXED: better visual hierarchy
    # Reduced opacity of background points
    ax2.scatter(z1[:, 0], z1[:, 1], s=15, alpha=0.15, color=COLOR_PRIMARY)
    ax2.scatter(z3[:, 0], z3[:, 1], s=15, alpha=0.15, color=COLOR_TERTIARY)

    # Interpolation path
    start = np.mean(z1, axis=0)
    end = np.mean(z3, axis=0)

    # Linear interpolation
    t = np.linspace(0, 1, 20)
    interp_points = np.outer(1 - t, start) + np.outer(t, end)

    # Thicker line for interpolation path
    ax2.plot(interp_points[:, 0], interp_points[:, 1], 'k-',
            linewidth=3, label='Interpolation path', zorder=4)  # Increased from 2 to 3
    ax2.scatter(interp_points[:, 0], interp_points[:, 1],
               s=50, c=t, cmap='viridis', edgecolors='black',  # Increased from 40 to 50
               linewidth=1.5, zorder=5)  # Increased edge width

    # Mark start and end
    ax2.scatter(*start, s=200, color=COLOR_PRIMARY, marker='*',
               edgecolors='black', linewidth=2, zorder=6, label='Start (Class 0)')
    ax2.scatter(*end, s=200, color=COLOR_TERTIARY, marker='*',
               edgecolors='black', linewidth=2, zorder=6, label='End (Class 2)')

    ax2.set_xlabel('Latent dimension $z_1$', fontsize=10)
    ax2.set_ylabel('Latent dimension $z_2$', fontsize=10)
    ax2.set_title('(b) Smooth Interpolation in Latent Space', fontsize=11)
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_aspect('equal')

    # Add annotation
    ax2.text(0.5, 0.05, 'Decoding intermediate points\n'
                       'produces smooth morphs between classes',
            transform=ax2.transAxes, ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_LIGHT_BLUE,
                     alpha=0.8, edgecolor='black', linewidth=1))

    plt.suptitle('VAE Latent Space: Learned Structure and Interpolation',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_12_latent_structure.pdf'),
                dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_8_12_latent_structure.png'),
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 8.12 saved")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Generating Chapter 8 Figures: Variational Inference, Bits-Back, and Flows")
    print("Fixed version with improved accessibility and clarity")
    print("=" * 70)
    print()

    figure_8_1()
    figure_8_2()
    figure_8_3()
    figure_8_4()
    figure_8_5()
    figure_8_6()
    figure_8_7()
    figure_8_8()
    figure_8_9()
    figure_8_10()
    figure_8_11()
    figure_8_12()

    print()
    print("=" * 70)
    print("All figures generated successfully!")
    print(f"Output location: {OUTPUT_DIR}/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 70)
    print()
    print("Figure Summary:")
    print("  8.1:  ELBO Decomposition and Gap")
    print("  8.2:  Reconstruction-Regularization Tradeoff (FIXED: line styles)")
    print("  8.3:  Reparameterization Trick")
    print("  8.4:  Bits-Back Coding Protocol")
    print("  8.5:  Three Gaps (FIXED: distinguished optimization gaps)")
    print("  8.6:  Coupling Layer Transformation")
    print("  8.7:  Change of Variables (Density Transformation)")
    print("  8.8:  VAE vs Flow Architecture Comparison")
    print("  8.9:  Posterior Collapse (FIXED: line styles + better caption)")
    print("  8.10: β-VAE Rate-Distortion (FIXED: larger text in info box)")
    print("  8.11: I-projection vs M-projection")
    print("  8.12: Latent Space Structure (FIXED: better visual hierarchy)")
    print()
