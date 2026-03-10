"""
Figure generation utilities for Chapter 9 (PAC-Bayes and MDL).
Produces the figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
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

# Color palette
COLOR_PRIMARY = '#1f77b4'      # Blue
COLOR_SECONDARY = '#ff7f0e'    # Orange
COLOR_TERTIARY = '#2ca02c'     # Green
COLOR_QUATERNARY = '#d62728'   # Red
COLOR_GRAY = '#7f7f7f'
COLOR_LIGHT_BLUE = '#aec7e8'
COLOR_LIGHT_ORANGE = '#ffbb78'
COLOR_PURPLE = '#9467bd'

# Figure dimensions
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0
DPI = 300

# Output directory (align with repo layout under code/figures)
OUTPUT_DIR = 'figures/ch09_pacbayes_mdl'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FIGURE 9.1: The Generalization Gap
# ============================================================================

def figure_9_1():
    """Visualize training vs test error and overfitting."""
    np.random.seed(42)

    epochs = np.linspace(0, 100, 200)

    # Well-regularized model
    train_good = 0.5 * np.exp(-epochs/20) + 0.05
    test_good = 0.5 * np.exp(-epochs/25) + 0.08 + 0.02 * np.random.randn(len(epochs)) * 0.3

    # Overfitting model
    train_overfit = 0.5 * np.exp(-epochs/15) + 0.001
    test_overfit = 0.5 * np.exp(-epochs/30) + 0.15 + 0.05 * (epochs/100)**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Panel 1: Good generalization
    ax1.plot(epochs, train_good, color=COLOR_PRIMARY, linewidth=2.5,
            label='Training error')
    ax1.plot(epochs, test_good, color=COLOR_SECONDARY, linewidth=2.5,
            label='Test error')
    ax1.fill_between(epochs, train_good, test_good, alpha=0.2, color=COLOR_TERTIARY)

    gap_epoch = 80
    gap_idx = np.argmin(np.abs(epochs - gap_epoch))
    ax1.plot([gap_epoch, gap_epoch], [train_good[gap_idx], test_good[gap_idx]],
            'k--', linewidth=2, alpha=0.7)
    ax1.text(gap_epoch + 5, (train_good[gap_idx] + test_good[gap_idx])/2,
            'Small gap\n(good)', fontsize=9, va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_BLUE, alpha=0.8))

    ax1.set_xlabel('Training Epochs', fontsize=10)
    ax1.set_ylabel('Error', fontsize=10)
    ax1.set_title('(a) Well-Regularized Model', fontsize=11)
    # Place legend above axes to avoid covering curves
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.6])

    # Panel 2: Overfitting
    ax2.plot(epochs, train_overfit, color=COLOR_PRIMARY, linewidth=2.5,
            label='Training error')
    ax2.plot(epochs, test_overfit, color=COLOR_QUATERNARY, linewidth=2.5,
            label='Test error (overfitting!)')
    ax2.fill_between(epochs, train_overfit, test_overfit, alpha=0.2, color=COLOR_QUATERNARY)

    gap_epoch = 80
    gap_idx = np.argmin(np.abs(epochs - gap_epoch))
    ax2.plot([gap_epoch, gap_epoch], [train_overfit[gap_idx], test_overfit[gap_idx]],
            'k--', linewidth=2, alpha=0.7)
    ax2.text(gap_epoch + 5, (train_overfit[gap_idx] + test_overfit[gap_idx])/2,
            'Large gap\n(overfitting)', fontsize=9, va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_ORANGE, alpha=0.8))

    best_epoch = np.argmin(test_overfit[:100])
    ax2.axvline(epochs[best_epoch], color=COLOR_TERTIARY, linestyle=':',
               linewidth=2, alpha=0.7, label='Early stopping')

    ax2.set_xlabel('Training Epochs', fontsize=10)
    ax2.set_ylabel('Error', fontsize=10)
    ax2.set_title('(b) Overfitting Model', fontsize=11)
    # Place legend above axes to keep plot area clean
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=1,
               frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.6])

    plt.suptitle('The Generalization Gap: Training vs Test Performance',
                 fontsize=12, fontweight='bold', y=0.98)

    # Leave room for legends placed above axes
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{OUTPUT_DIR}/fig_9_1_generalization_gap.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_1_generalization_gap.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.1 saved")
    plt.close()

# ============================================================================
# FIGURE 9.2: PAC-Bayes Bound Decomposition
# ============================================================================

def figure_9_2():
    """Visualize the PAC-Bayes bound structure with tighter, non-overlapping layout."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.05, FIGURE_HEIGHT * 0.95))

    # True risk bar (move lower; shorten height to reduce whitespace)
    ax.barh([2.25], [0.70], left=0.16, height=0.24,
            color=COLOR_GRAY, alpha=0.45, edgecolor='black', linewidth=1.6)
    ax.text(0.51, 2.25, 'True Risk (unknown)', ha='center', va='center', fontsize=10)

    # Bound components (make text smaller to fit neatly inside the bars)
    # Taller bars and white sub-boxes to ensure text contrast
    y_comp = 1.85
    bar_h = 0.30
    ax.barh([y_comp], [0.41], left=0.16, height=bar_h,
            color=COLOR_PRIMARY, alpha=0.85, edgecolor='black', linewidth=1.6)
    # White sub-box overlay for text readability
    ax.add_patch(FancyBboxPatch((0.19, y_comp - bar_h/2 + 0.03), 0.33, bar_h - 0.06,
                                boxstyle='round,pad=0.02', facecolor='white',
                                edgecolor='none', alpha=0.95, zorder=2))
    ax.text(0.36, y_comp, '$\\mathbb{E}_Q[\\mathrm{emp\\ risk}]$',
            ha='center', va='center', fontsize=9, color='black', zorder=3)

    ax.barh([y_comp], [0.28], left=0.57, height=bar_h,
            color=COLOR_SECONDARY, alpha=0.9, edgecolor='black', linewidth=1.6)
    ax.add_patch(FancyBboxPatch((0.585, y_comp - bar_h/2 + 0.03), 0.245, bar_h - 0.06,
                                boxstyle='round,pad=0.02', facecolor='white',
                                edgecolor='none', alpha=0.95, zorder=2))
    ax.text(0.70, y_comp, '$+\\, \\sqrt{(\\mathrm{KL}+\\log(n/\\delta))/(2n)}$',
            ha='center', va='center', fontsize=8.5, color='black', zorder=3)

    # Inequality arrow
    ax.annotate('', xy=(0.5, 2.10), xytext=(0.5, 2.00),
                arrowprops=dict(arrowstyle='<-', lw=2.0, color='black'))
    ax.text(0.54, 2.04, '$\\leq$', fontsize=12, fontweight='bold')

    # Breakdown box: compact 2-column layout (no overlaps)
    box_x, box_y, box_w, box_h = 0.10, 0.92, 0.80, 0.62
    ax.add_patch(FancyBboxPatch((box_x, box_y), box_w, box_h,
                                boxstyle='round,pad=0.02',
                                facecolor='white', alpha=0.96,
                                edgecolor=COLOR_SECONDARY, linewidth=1.8))

    ax.text(box_x + 0.03, box_y + box_h - 0.10,
            'Complexity term pieces:', ha='left', va='center',
            fontsize=9.2, fontweight='bold', color=COLOR_SECONDARY)

    rows = [
        (r'$\mathrm{KL}(Q\|P)$', 'compression / information gain from data'),
        (r'$\log(n/\delta)$', 'confidence term ($\\delta$ = failure prob.)'),
        (r'$1/\sqrt{n}$', 'sample size scaling (more data $\\Rightarrow$ tighter)')
    ]

    y0 = box_y + box_h - 0.22
    dy = 0.17
    for i, (term, explanation) in enumerate(rows):
        y = y0 - i * dy
        if i == 0:
            ax.add_patch(FancyBboxPatch((box_x + 0.02, y - 0.06), box_w - 0.04, 0.12,
                                        boxstyle='round,pad=0.01',
                                        facecolor=COLOR_LIGHT_ORANGE, alpha=0.35,
                                        edgecolor='none', zorder=0))
        ax.text(box_x + 0.05, y, term, fontsize=9.6, fontweight='bold',
                ha='left', va='center')
        ax.text(box_x + 0.28, y, explanation, fontsize=9.0,
                ha='left', va='center')

    ax.text(0.5, box_y + 0.07,
            'Key insight: smaller $\\mathrm{KL}(Q\\|P)$ $\\Rightarrow$ tighter generalization certificate',
            ha='center', va='center', fontsize=8.6, color=COLOR_SECONDARY, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COLOR_SECONDARY, linewidth=1.4))

    ax.set_xlim([0, 1])
    ax.set_ylim([0.85, 2.45])
    ax.axis('off')
    ax.set_title('PAC-Bayes Bound: Decomposition and Interpretation',
                 fontsize=12, fontweight='bold', pad=6)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_2_pacbayes_bound.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_2_pacbayes_bound.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.2 saved")
    plt.close()

# ============================================================================
# FIGURE 9.3: KL as Information/Codelength
# ============================================================================

def figure_9_3():
    """Visualize KL divergence as codelength difference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Panel 1: Prior vs Posterior
    theta = np.linspace(-4, 6, 200)
    prior = stats.norm.pdf(theta, loc=0, scale=1.5)
    posterior = stats.norm.pdf(theta, loc=2, scale=0.5)

    ax1.fill_between(theta, 0, prior, alpha=0.3, color=COLOR_GRAY,
                     label='Prior $P$ (before data)')
    ax1.plot(theta, prior, color=COLOR_GRAY, linewidth=2.5, linestyle='--')

    ax1.fill_between(theta, 0, posterior, alpha=0.4, color=COLOR_PRIMARY,
                     label='Posterior $Q$ (after data)')
    ax1.plot(theta, posterior, color=COLOR_PRIMARY, linewidth=2.5)

    ax1.axvline(0, color=COLOR_GRAY, linestyle=':', linewidth=2, alpha=0.7)
    ax1.axvline(2, color=COLOR_PRIMARY, linestyle=':', linewidth=2, alpha=0.7)

    ax1.annotate('', xy=(2, 0.05), xytext=(0, 0.05),
                arrowprops=dict(arrowstyle='<->', lw=2, color=COLOR_SECONDARY))
    ax1.text(1, 0.08, 'Information from data\n$\\mathrm{KL}(Q \\| P)$ nats',
            ha='center', fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_ORANGE, alpha=0.9, edgecolor='black'))

    ax1.set_xlabel('Parameter $\\theta$', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('(a) Prior → Posterior: Learning', fontsize=11)
    # Keep legend outside to avoid covering density curves
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2,
               fontsize=9, frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.9])

    # Panel 2: Codelength interpretation
    ax2.text(0.5, 0.95, 'KL as Codelength', ha='center', fontsize=12,
            fontweight='bold', transform=ax2.transAxes)

    scenarios = [
        ('$-\\log P(h)$', 'Cost to send hypothesis\nusing prior code', 0.75, COLOR_GRAY, 15),
        ('$-\\log Q(h)$', 'Cost using posterior code\n(if receiver knows $Q$)', 0.55, COLOR_PRIMARY, 8),
        ('', 'Difference = $\\mathrm{KL}(Q \\| P)$\n= Info extracted from data', 0.35, COLOR_SECONDARY, 7)
    ]

    for term, desc, y, color, cost in scenarios:
        if term:
            # Slightly taller boxes and slightly lower top to align with text baseline
            ax2.add_patch(Rectangle((0.1, y - 0.055), 0.25, 0.095,
                                   facecolor=color, alpha=0.5,
                                   edgecolor='black', linewidth=1.5,
                                   transform=ax2.transAxes))
            ax2.text(0.225, y, term, ha='center', va='center', fontsize=10,
                    transform=ax2.transAxes, fontweight='bold')
            # Bring descriptor text closer and slightly lower for nicer alignment
            ax2.text(0.40, y-0.01, desc, ha='left', va='center', fontsize=8,
                    transform=ax2.transAxes)
            # Move bits label leftwards to avoid overlap with descriptors
            ax2.text(0.78, y, f'{cost} bits', ha='right', va='center', fontsize=9,
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))
        else:
            ax2.annotate('', xy=(0.3, y + 0.03), xytext=(0.15, y + 0.03),
                        transform=ax2.transAxes,
                        arrowprops=dict(arrowstyle='<->', lw=2.5, color=color))
            ax2.text(0.40, y-0.01, desc, ha='left', va='center', fontsize=9,
                    transform=ax2.transAxes, color=color, fontweight='bold')

    formula = ('$\\mathrm{KL}(Q \\| P) = \\mathbb{E}_Q[\\log Q - \\log P]$\n'
              '$= \\mathbb{E}_Q[-\\log P] - H(Q)$')
    ax2.text(0.5, 0.12, formula, ha='center', fontsize=10, transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_BLUE,
                     alpha=0.8, edgecolor='black', linewidth=1.5))

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.axis('off')

    # Leave room for external legend on panel (a)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{OUTPUT_DIR}/fig_9_3_kl_codelength.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_3_kl_codelength.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.3 saved")
    plt.close()

# ============================================================================
# FIGURE 9.4: Two-Part MDL Code
# ============================================================================

def figure_9_4():
    """Visualize the two-part MDL code structure."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.2, FIGURE_HEIGHT))

    models = [
        ('Simple', 20, 500, COLOR_TERTIARY),
        ('Optimal', 40, 200, COLOR_PRIMARY),
        ('Complex', 200, 50, COLOR_QUATERNARY)
    ]

    y_positions = [0.7, 0.45, 0.2]

    for (name, model_cost, data_cost, color), y in zip(models, y_positions):
        total = model_cost + data_cost

        # Model cost
        ax.add_patch(Rectangle((0.1, y - 0.05), model_cost/600, 0.08,
                               facecolor=COLOR_LIGHT_BLUE, alpha=0.7,
                               edgecolor='black', linewidth=1.5))
        ax.text(0.1 + model_cost/1200, y, f'{model_cost}', ha='center', va='center', fontsize=9)

        # Data|Model cost
        ax.add_patch(Rectangle((0.1 + model_cost/600, y - 0.05), data_cost/600, 0.08,
                               facecolor=COLOR_LIGHT_ORANGE, alpha=0.7,
                               edgecolor='black', linewidth=1.5))
        ax.text(0.1 + model_cost/600 + data_cost/1200, y, f'{data_cost}',
               ha='center', va='center', fontsize=9)

        # Labels
        ax.text(0.05, y, name, ha='right', va='center', fontsize=10, fontweight='bold')

        total_color = COLOR_TERTIARY if name == 'Optimal' else COLOR_GRAY
        total_weight = 'bold' if name == 'Optimal' else 'normal'
        ax.text(0.75, y, f'Total: {total} bits', ha='left', va='center', fontsize=9,
               color=total_color, fontweight=total_weight,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=total_color, linewidth=2 if name == 'Optimal' else 1))

    # Legend
    ax.add_patch(Rectangle((0.15, 0.9), 0.12, 0.05, facecolor=COLOR_LIGHT_BLUE,
                           alpha=0.7, edgecolor='black', linewidth=1,
                           transform=ax.transAxes))
    ax.text(0.28, 0.925, 'Model: $L(M)$', ha='left', va='center', fontsize=9,
           transform=ax.transAxes)

    ax.add_patch(Rectangle((0.5, 0.9), 0.12, 0.05, facecolor=COLOR_LIGHT_ORANGE,
                           alpha=0.7, edgecolor='black', linewidth=1,
                           transform=ax.transAxes))
    ax.text(0.63, 0.925, 'Data$|$Model: $L(D|M)$', ha='left', va='center',
           fontsize=9, transform=ax.transAxes)

    ax.text(0.5, 0.05, 'MDL Principle: Choose model minimizing $L(M) + L(D|M)$',
           ha='center', fontsize=10, color=COLOR_TERTIARY, fontweight='bold',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_BLUE,
                    alpha=0.6, edgecolor=COLOR_TERTIARY, linewidth=2))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    ax.set_title('Two-Part MDL: Balancing Model Complexity and Fit',
                fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_4_mdl_twopart.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_4_mdl_twopart.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.4 saved")
    plt.close()

# ============================================================================
# FIGURE 9.5: Flat vs Sharp Minima (3D Loss Landscape)
# ============================================================================

def figure_9_5():
    """Visualize flat and sharp minima in a 3D loss landscape.
    Improves star visibility with white borders and raised markers."""
    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT * 1.2))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Flat minimum (left) and sharp minimum (right)
    Z_flat = 0.1 * (X + 2)**2 + 0.1 * (Y)**2 + 0.5
    Z_sharp = 2 * (X - 2)**2 + 2 * (Y)**2 + 0.5
    Z = np.where(X < 0, Z_flat, Z_sharp)

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.82,
                           linewidth=0, antialiased=True)

    # Compute z at marker locations and raise well above the surface for visibility
    flat_x, flat_y = -2, 1.3
    sharp_x, sharp_y = 2, 0.8
    z_flat_star = 0.1 * (flat_x + 2)**2 + 0.1 * (flat_y)**2 + 0.5 + 1.2
    z_sharp_star = 2 * (sharp_x - 2)**2 + 2 * (sharp_y)**2 + 0.5 + 1.2

    # Draw white-underlay stars first for clear borders
    ax.scatter([flat_x], [flat_y], [z_flat_star], color='white', s=520, marker='*',
               depthshade=False, zorder=100)
    ax.scatter([sharp_x], [sharp_y], [z_sharp_star], color='white', s=520, marker='*',
               depthshade=False, zorder=100)

    # Colored stars on top
    ax.scatter([flat_x], [flat_y], [z_flat_star], color=COLOR_TERTIARY, s=380, marker='*',
               edgecolors='white', linewidths=2.0, label='Flat minimum',
               depthshade=False, zorder=101)
    ax.scatter([sharp_x], [sharp_y], [z_sharp_star], color=COLOR_QUATERNARY, s=380, marker='*',
               edgecolors='white', linewidths=2.0, label='Sharp minimum',
               depthshade=False, zorder=101)

    # Perturbation circles
    theta_circle = np.linspace(0, 2*np.pi, 50)
    r = 0.5
    x_flat = -2 + r * np.cos(theta_circle)
    y_flat = 0 + r * np.sin(theta_circle)
    z_flat = 0.1 * (x_flat + 2)**2 + 0.1 * (y_flat)**2 + 0.5
    ax.plot(x_flat, y_flat, z_flat, color=COLOR_TERTIARY, linewidth=2.5, alpha=0.9)

    x_sharp = 2 + r * np.cos(theta_circle)
    y_sharp = 0 + r * np.sin(theta_circle)
    z_sharp = 2 * (x_sharp - 2)**2 + 2 * (y_sharp)**2 + 0.5
    ax.plot(x_sharp, y_sharp, z_sharp, color=COLOR_QUATERNARY, linewidth=2.5, alpha=0.9)

    ax.set_xlabel('$\\theta_1$', fontsize=11)
    ax.set_ylabel('$\\theta_2$', fontsize=11)
    ax.set_zlabel('Loss', fontsize=11)
    ax.set_title('Flat vs Sharp Minima: Loss Landscape Curvature', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_5_flat_sharp_minima.pdf', dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_5_flat_sharp_minima.png', dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.5 saved")
    plt.close()

# ============================================================================
# FIGURE 9.6: Hessian Eigenvalues and Flatness Measures
# ============================================================================

def figure_9_6():
    """Show Hessian eigenvalue spectrum and flatness metrics."""
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Panel 1: Eigenvalue spectra
    n_params = 100
    eigs_flat = np.sort(np.abs(np.random.randn(n_params) * 0.5))
    eigs_sharp = np.sort(np.abs(np.random.randn(n_params) * 0.5))
    eigs_sharp[-10:] = np.linspace(5, 15, 10)

    ax1.semilogy(range(n_params), eigs_flat, 'o-', color=COLOR_TERTIARY,
                linewidth=2, markersize=4, alpha=0.7, label='Flat minimum')
    ax1.semilogy(range(n_params), eigs_sharp, 's-', color=COLOR_QUATERNARY,
                linewidth=2, markersize=4, alpha=0.7, label='Sharp minimum')

    ax1.set_xlabel('Eigenvalue index (sorted)', fontsize=10)
    ax1.set_ylabel('Eigenvalue magnitude (log scale)', fontsize=10)
    ax1.set_title('(a) Hessian Eigenvalue Spectrum', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Move callouts away from data and use arrows to point at regions
    ax1.annotate('Large eigenvalues\n$\\rightarrow$ sharp directions',
                 xy=(95, eigs_sharp[-1]), xycoords='data',
                 xytext=(65, 6), textcoords='data', ha='center', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_ORANGE, alpha=0.85),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_QUATERNARY))
    ax1.annotate('Small eigenvalues\n$\\rightarrow$ flat directions',
                 xy=(5, eigs_flat[2]), xycoords='data',
                 xytext=(25, 0.04), textcoords='data', ha='center', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_BLUE, alpha=0.85),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_TERTIARY))

    # Panel 2: Flatness metrics
    metrics = ['Max\nEigenvalue', 'Trace\n(sum)', 'Sharpness\n(max loss\nin ball)']
    flat_values = [np.max(eigs_flat), np.sum(eigs_flat), 0.3]
    sharp_values = [np.max(eigs_sharp), np.sum(eigs_sharp), 2.5]

    x_pos = np.arange(len(metrics))
    width = 0.35
    flat_norm = np.array(flat_values) / np.array(sharp_values)
    sharp_norm = np.ones(len(metrics))

    bars1 = ax2.bar(x_pos - width/2, flat_norm, width, label='Flat minimum',
                    color=COLOR_TERTIARY, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width/2, sharp_norm, width, label='Sharp minimum',
                    color=COLOR_QUATERNARY, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Relative magnitude (sharp $= 1$)\n(lower $\\Rightarrow$ flatter)', fontsize=10)
    ax2.set_title('(b) Flatness Metrics', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics, fontsize=9)
    # Legend inside the axis (prevents title overlap/clipping when saving with tight bbox).
    ax2.legend(loc='upper center', ncol=2, frameon=True, framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    # Add headroom so legend doesn't cover the sharp-min bars at 1.0.
    ax2.set_ylim([0, 1.35])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_6_hessian_flatness.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_6_hessian_flatness.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.6 saved")
    plt.close()

# ============================================================================
# FIGURE 9.7: Connection Between Flatness and PAC-Bayes
# ============================================================================

def figure_9_7():
    """Show how flat minima correspond to low KL divergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    theta = np.linspace(-4, 4, 200)
    prior = stats.norm.pdf(theta, loc=0, scale=1.2)
    posterior_flat = stats.norm.pdf(theta, loc=1, scale=0.8)
    posterior_sharp = stats.norm.pdf(theta, loc=1, scale=0.3)

    # Panel 1: Flat minimum
    ax1.fill_between(theta, 0, prior, alpha=0.2, color=COLOR_GRAY)
    ax1.plot(theta, prior, color=COLOR_GRAY, linewidth=2, linestyle='--', label='Prior $P$')
    ax1.fill_between(theta, 0, posterior_flat, alpha=0.4, color=COLOR_TERTIARY)
    ax1.plot(theta, posterior_flat, color=COLOR_TERTIARY, linewidth=2.5,
            label='Posterior $Q$ (flat min)')

    kl_flat = 0.5 * (0.8**2 / 1.2**2 + 1**2 / 1.2**2 - 1 - np.log(0.8**2 / 1.2**2))

    ax1.set_xlabel('Parameter $\\theta$', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title(f'(a) Flat Minimum\n$\\mathrm{{KL}}(Q \\| P) \\approx {kl_flat:.2f}$ nats',
                 fontsize=11)
    # Keep legend away from the summary callout at the top.
    ax1.legend(loc='lower left', frameon=True, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.6])

    ax1.text(0.5, 0.76, '$\\checkmark$ Broad posterior\n$\\checkmark$ Low KL\n$\\checkmark$ Better generalization',
            transform=ax1.transAxes, ha='center', fontsize=9,
            color=COLOR_TERTIARY, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_LIGHT_BLUE,
                     alpha=0.8, edgecolor=COLOR_TERTIARY, linewidth=2))

    # Panel 2: Sharp minimum
    ax2.fill_between(theta, 0, prior, alpha=0.2, color=COLOR_GRAY)
    ax2.plot(theta, prior, color=COLOR_GRAY, linewidth=2, linestyle='--', label='Prior $P$')
    ax2.fill_between(theta, 0, posterior_sharp, alpha=0.4, color=COLOR_QUATERNARY)
    ax2.plot(theta, posterior_sharp, color=COLOR_QUATERNARY, linewidth=2.5,
            label='Posterior $Q$ (sharp min)')

    kl_sharp = 0.5 * (0.3**2 / 1.2**2 + 1**2 / 1.2**2 - 1 - np.log(0.3**2 / 1.2**2))

    ax2.set_xlabel('Parameter $\\theta$', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title(f'(b) Sharp Minimum\n$\\mathrm{{KL}}(Q \\| P) \\approx {kl_sharp:.2f}$ nats',
                 fontsize=11)
    # Keep legend away from the summary callout at the top.
    ax2.legend(loc='lower left', frameon=True, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.5])

    ax2.text(0.5, 0.76, '$\\times$ Narrow posterior\n$\\times$ High KL\n$\\times$ Worse generalization',
            transform=ax2.transAxes, ha='center', fontsize=9,
            color=COLOR_QUATERNARY, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_LIGHT_ORANGE,
                     alpha=0.8, edgecolor=COLOR_QUATERNARY, linewidth=2))

    plt.suptitle('Flatness $\\leftrightarrow$ Low KL $\\leftrightarrow$ Good Generalization',
                 fontsize=12, fontweight='bold', y=0.97)

    # Legends are inside; standard margins suffice
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'{OUTPUT_DIR}/fig_9_7_flatness_pacbayes.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_7_flatness_pacbayes.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.7 saved")
    plt.close()

# ============================================================================
# FIGURE 9.8: SGD Noise and Implicit Regularization
# ============================================================================

def figure_9_8():
    """Show how SGD noise favors flat minima."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    theta = np.linspace(-3, 3, 300)
    loss_flat = 0.5 * (theta + 1)**2 + 1
    loss_sharp = 10 * (theta - 1.5)**2 + 0.5
    loss = np.minimum(loss_flat, loss_sharp)

    # Panel 1: Large batch (deterministic)
    ax1.plot(theta, loss, 'k-', linewidth=2, alpha=0.5)
    ax1.fill_between(theta, 0, loss, alpha=0.1, color=COLOR_GRAY)

    traj_large = np.array([[-2.5, 12], [-1.8, 6], [-1.2, 3], [0, 1.5],
                          [1.0, 2], [1.4, 0.8], [1.5, 0.5]])
    ax1.plot(traj_large[:, 0], traj_large[:, 1], 'o-', color=COLOR_QUATERNARY,
            linewidth=2.5, markersize=8, label='Large batch trajectory')
    ax1.plot(1.5, 0.5, '*', color=COLOR_QUATERNARY, markersize=20,
            markeredgecolor='black', markeredgewidth=1.5,
            label='Converges to sharp minimum')

    ax1.set_xlabel('Parameter $\\theta$', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('(a) Large Batch: Low Noise\n(Deterministic → Sharp Min)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 15])

    # Panel 2: Small batch (noisy)
    ax2.plot(theta, loss, 'k-', linewidth=2, alpha=0.5)
    ax2.fill_between(theta, 0, loss, alpha=0.1, color=COLOR_GRAY)

    np.random.seed(42)
    traj_small = np.array([[-2.5, 12]])
    current_pos = -2.5
    for i in range(15):
        gradient_dir = -np.sign(current_pos + 1) if current_pos < 0 else np.sign(current_pos - 1.5)
        noise = np.random.randn() * 0.3
        current_pos = current_pos - 0.3 * gradient_dir + noise
        current_pos = np.clip(current_pos, -3, 3)
        loss_val = min(0.5 * (current_pos + 1)**2 + 1, 10 * (current_pos - 1.5)**2 + 0.5)
        traj_small = np.vstack([traj_small, [current_pos, loss_val]])

    ax2.plot(traj_small[:, 0], traj_small[:, 1], 'o-', color=COLOR_TERTIARY,
            linewidth=1.5, markersize=5, alpha=0.7, label='Small batch trajectory')
    ax2.plot(-1, 1, '*', color=COLOR_TERTIARY, markersize=20,
            markeredgecolor='black', markeredgewidth=1.5,
            label='Converges to flat minimum')

    ax2.add_patch(Ellipse((-1, 1), 1.5, 3, alpha=0.2, color=COLOR_TERTIARY,
                         edgecolor=COLOR_TERTIARY, linewidth=2, linestyle='--'))
    ax2.text(-1, 4.5, 'Noise explores\nflat basin', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_BLUE, alpha=0.8))

    ax2.set_xlabel('Parameter $\\theta$', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_title('(b) Small Batch: High Noise\n(Stochastic → Flat Min)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 15])

    plt.suptitle('SGD Noise as Implicit Regularization Toward Flat Minima',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_8_sgd_noise.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_8_sgd_noise.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.8 saved")
    plt.close()

# ============================================================================
# FIGURE 9.9: Risk-Coverage Tradeoff
# ============================================================================

def figure_9_9():
    """Visualize selective prediction risk-coverage curves."""
    np.random.seed(42)

    coverage = np.linspace(0.1, 1.0, 100)
    risk_calibrated = 0.02 + 0.25 * (coverage - 0.1)**1.5
    risk_uncalibrated = 0.05 + 0.3 * (coverage - 0.1)**1.2
    risk_oracle = 0.01 + 0.15 * (coverage - 0.1)**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Panel 1: Risk-coverage curves
    ax1.plot(coverage, risk_calibrated, color=COLOR_PRIMARY, linewidth=2.5,
            label='Calibrated model')
    ax1.plot(coverage, risk_uncalibrated, color=COLOR_QUATERNARY, linewidth=2.5,
            linestyle='--', label='Miscalibrated model')
    ax1.plot(coverage, risk_oracle, color=COLOR_TERTIARY, linewidth=2.5,
            linestyle=':', label='Oracle (best possible)')

    op_points = [(0.5, 0.02 + 0.25 * 0.4**1.5, 'High\nConfidence'),
                 (0.8, 0.02 + 0.25 * 0.7**1.5, 'Balanced'),
                 (1.0, 0.02 + 0.25 * 0.9**1.5, 'Full\nCoverage')]

    for cov, risk, label in op_points:
        ax1.plot(cov, risk, 'o', markersize=10, color=COLOR_SECONDARY,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax1.annotate(label, xy=(cov, risk), xytext=(cov - 0.15, risk + 0.05),
                    fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', lw=1.5))

    target_risk = 0.08
    ax1.axhline(target_risk, color='gray', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Target risk = {target_risk}')

    idx_target = np.argmin(np.abs(risk_calibrated - target_risk))
    cov_target = coverage[idx_target]
    ax1.axvline(cov_target, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(cov_target, 0.01, f'$c^* = {cov_target:.2f}$', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

    ax1.set_xlabel('Coverage (fraction of predictions made)', fontsize=10)
    ax1.set_ylabel('Selective Risk', fontsize=10)
    ax1.set_title('(a) Risk-Coverage Curves', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 0.35])

    # Panel 2: Decision boundaries
    # Remove redundant header to prevent overlap with the subplot title

    confidence = np.linspace(0, 1, 200)
    density = 2 * confidence
    density = density / np.trapezoid(density, confidence)

    ax2.fill_between(confidence, 0, density, alpha=0.3, color=COLOR_PRIMARY)
    ax2.plot(confidence, density, color=COLOR_PRIMARY, linewidth=2,
            label='Confidence distribution')

    threshold = 0.6
    ax2.axvline(threshold, color=COLOR_QUATERNARY, linestyle='--', linewidth=2.5,
               label=f'Threshold $\\tau = {threshold}$')

    ax2.fill_between(confidence[confidence < threshold], 0, 1, alpha=0.2,
                     color=COLOR_QUATERNARY, transform=ax2.get_xaxis_transform(),
                     label='Abstain')
    ax2.fill_between(confidence[confidence >= threshold], 0, 1, alpha=0.2,
                     color=COLOR_TERTIARY, transform=ax2.get_xaxis_transform(),
                     label='Predict')

    ax2.text(0.3, 0.7, 'ABSTAIN\n(uncertain)', ha='center', fontsize=10,
            color=COLOR_QUATERNARY, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.75, 0.7, 'PREDICT\n(confident)', ha='center', fontsize=10,
            color=COLOR_TERTIARY, fontweight='bold', transform=ax2.transAxes)

    ax2.set_xlabel('Model Confidence', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title(f'(b) Abstention Rule: $g(x) = \\mathbb{{1}}[\\mathrm{{conf}}(x) \\geq {threshold}]$',
                 fontsize=10)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_9_risk_coverage.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_9_risk_coverage.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.9 saved")
    plt.close()

# ============================================================================
# FIGURE 9.10: Conformal Prediction Sets
# ============================================================================

def figure_9_10():
    """Visualize conformal prediction sets and coverage guarantee."""
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    # Panel 1: Prediction set sizes
    n_examples = 50
    easy_sizes = np.random.poisson(1, 15)
    medium_sizes = np.random.poisson(3, 20)
    hard_sizes = np.random.poisson(8, 15)
    set_sizes = np.concatenate([easy_sizes, medium_sizes, hard_sizes])
    colors = ['green'] * 15 + ['orange'] * 20 + ['red'] * 15

    ax1.bar(range(n_examples), set_sizes, color=colors, alpha=0.6,
           edgecolor='black', linewidth=0.5)

    ax1.axvspan(-0.5, 14.5, alpha=0.1, color=COLOR_TERTIARY)
    ax1.text(7, 13, 'Easy\n(confident)', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

    ax1.axvspan(14.5, 34.5, alpha=0.1, color=COLOR_SECONDARY)
    ax1.text(24, 13, 'Medium', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

    ax1.axvspan(34.5, 49.5, alpha=0.1, color=COLOR_QUATERNARY)
    ax1.text(42, 13, 'Hard\n(uncertain)', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white'))

    ax1.set_xlabel('Test Example Index', fontsize=10)
    ax1.set_ylabel('Prediction Set Size', fontsize=10)
    ax1.set_title('(a) Conformal Prediction Set Sizes', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 15])

    ax1.text(0.5, 0.95, 'Set size = model uncertainty', transform=ax1.transAxes,
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_LIGHT_BLUE, alpha=0.8))

    # Panel 2: Coverage guarantee
    alpha_levels = np.array([0.05, 0.10, 0.20])
    theoretical_coverage = 1 - alpha_levels
    empirical_coverage = theoretical_coverage + np.random.randn(3) * 0.01

    x_pos = np.arange(len(alpha_levels))
    width = 0.35

    bars1 = ax2.bar(x_pos - width/2, theoretical_coverage, width,
                    label='Guaranteed: $1 - \\alpha$', color=COLOR_PRIMARY,
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width/2, empirical_coverage, width,
                    label='Empirical', color=COLOR_TERTIARY,
                    alpha=0.7, edgecolor='black', linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Coverage (P[true label in set])', fontsize=10)
    ax2.set_title('(b) Coverage Guarantee', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'$\\alpha={a:.2f}$' for a in alpha_levels])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.7, 1.05])

    guarantee_text = ('Distribution-free guarantee:\n'
                     '$P(y \\in C_\\alpha(x)) \\geq 1 - \\alpha$')
    ax2.text(0.5, 0.15, guarantee_text, ha='center', fontsize=9,
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT_ORANGE,
                     alpha=0.8, edgecolor='black', linewidth=1.5))

    plt.suptitle('Conformal Prediction: Rigorous Coverage Guarantees',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_10_conformal_prediction.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_10_conformal_prediction.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.10 saved")
    plt.close()

# ============================================================================
# FIGURE 9.11: SAM (Sharpness-Aware Minimization)
# ============================================================================

def figure_9_11():
    """Visualize how SAM works."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.6, FIGURE_HEIGHT))

    theta = np.linspace(-2, 3, 300)
    loss = (theta - 0.5)**2 + 0.5 * np.sin(5 * theta) + 2

    theta_curr = 0.3
    loss_curr = (theta_curr - 0.5)**2 + 0.5 * np.sin(5 * theta_curr) + 2

    # Panel 1: Standard SGD
    ax1.plot(theta, loss, 'k-', linewidth=2, alpha=0.7)
    ax1.fill_between(theta, 0, loss, alpha=0.1, color=COLOR_GRAY)

    ax1.plot(theta_curr, loss_curr, 'o', markersize=12, color=COLOR_QUATERNARY,
            markeredgecolor='black', markeredgewidth=2, label='Current $\\theta$')

    gradient = 2 * (theta_curr - 0.5) + 0.5 * 5 * np.cos(5 * theta_curr)
    theta_next = theta_curr - 0.15 * gradient
    loss_next = (theta_next - 0.5)**2 + 0.5 * np.sin(5 * theta_next) + 2

    ax1.annotate('', xy=(theta_next, loss_next), xytext=(theta_curr, loss_curr),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_QUATERNARY))
    ax1.plot(theta_next, loss_next, 's', markersize=10, color=COLOR_QUATERNARY,
            markeredgecolor='black', markeredgewidth=1.5)

    ax1.text(theta_curr - 0.3, loss_curr + 0.5, 'Standard\nSGD step', fontsize=9,
            ha='center', bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor=COLOR_LIGHT_ORANGE, alpha=0.8))

    ax1.set_xlabel('Parameter $\\theta$', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('(a) Standard SGD', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 5])

    # Panel 2: SAM
    ax2.plot(theta, loss, 'k-', linewidth=2, alpha=0.7)
    ax2.fill_between(theta, 0, loss, alpha=0.1, color=COLOR_GRAY)

    ax2.plot(theta_curr, loss_curr, 'o', markersize=12, color=COLOR_TERTIARY,
            markeredgecolor='black', markeredgewidth=2, label='Current $\\theta$')

    # Step 1: Adversarial perturbation
    rho = 0.1
    theta_perturb = theta_curr + rho * gradient / (abs(gradient) + 1e-10)
    loss_perturb = (theta_perturb - 0.5)**2 + 0.5 * np.sin(5 * theta_perturb) + 2

    ax2.annotate('', xy=(theta_perturb, loss_perturb), xytext=(theta_curr, loss_curr),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_SECONDARY,
                              linestyle='--'))
    ax2.plot(theta_perturb, loss_perturb, '^', markersize=10, color=COLOR_SECONDARY,
            markeredgecolor='black', markeredgewidth=1.5,
            label='Perturbed $\\theta + \\rho \\nabla \\mathcal{L}$')
    ax2.text(theta_perturb + 0.5, loss_perturb + 0.5, '1. Ascent', fontsize=8, ha='center')

    # Step 2: Gradient at perturbed point
    gradient_perturb = 2 * (theta_perturb - 0.5) + 0.5 * 5 * np.cos(5 * theta_perturb)
    theta_sam = theta_curr - 0.15 * gradient_perturb
    loss_sam = (theta_sam - 0.5)**2 + 0.5 * np.sin(5 * theta_sam) + 2

    ax2.annotate('', xy=(theta_sam, loss_sam), xytext=(theta_curr, loss_curr),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_TERTIARY))
    ax2.plot(theta_sam, loss_sam, 's', markersize=10, color=COLOR_TERTIARY,
            markeredgecolor='black', markeredgewidth=1.5, label='SAM update')
    # Place the step-2 textbox away from the ascent label to avoid overlap
    ax2.text(theta_curr - 0.9, (loss_curr + loss_sam)/2 + 0.6,
            '2. Descent\nusing worst-case\ngradient', fontsize=7.5, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLOR_LIGHT_BLUE, alpha=0.85))

    ax2.set_xlabel('Parameter $\\theta$', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.set_title('(b) Sharpness-Aware Minimization (SAM)', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 5])

    plt.suptitle('SAM: Explicitly Seeking Flat Minima',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_11_sam.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_11_sam.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.11 saved")
    plt.close()

# ============================================================================
# FIGURE 9.12: Compression-Generalization Unifying View
# ============================================================================

def figure_9_12():
    """Unifying view: all roads lead to compression."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.15, FIGURE_HEIGHT * 1.2))

    # Professional, minimalist color scheme (local to this figure)
    base = '#2c3e50'        # dark slate
    accent = '#2a9d8f'      # teal accent
    mid_gray = '#6c7a89'
    light_fill = '#f5f7fa'

    # Center: Generalization (draw arrows below text; draw text last)
    center_x, center_y = 0.5, 0.5
    ax.add_patch(Circle((center_x, center_y), 0.11, facecolor=accent,
                        alpha=0.9, edgecolor=base, linewidth=2.5,
                        transform=ax.transAxes, zorder=3))

    # Surrounding concepts
    concepts = [
        ('PAC-Bayes\nLow KL', 0.25, 0.78, base, '$\\mathrm{KL}(Q\\|P)$ small'),
        ('MDL\nShort Code', 0.75, 0.78, base, '$L(M) + L(D|M)$ small'),
        ('Flat Minima\nBroad Posterior', 0.17, 0.34, base, 'Small Hessian eigenvalues'),
        ('SGD Noise\nImplicit Reg', 0.83, 0.34, base, 'Escapes sharp minima'),
        ('Compression', 0.5, 0.22, base, 'All frameworks converge')
    ]

    for label, x, y, color, detail in concepts:
        ax.add_patch(FancyBboxPatch((x - 0.085, y - 0.055), 0.17, 0.11,
                                    boxstyle='round,pad=0.01',
                                    facecolor=light_fill, alpha=1.0,
                                    edgecolor=base, linewidth=1.8,
                                    transform=ax.transAxes, zorder=1))
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color=base, transform=ax.transAxes, zorder=2)

        if label != 'Compression':
            # Aim arrowheads to just outside the green circle to avoid overlapping its text
            dx, dy = (center_x - x), (center_y - y)
            L = (dx**2 + dy**2) ** 0.5 + 1e-6
            margin = 0.13  # circle radius (0.11) + small gap
            end_x = center_x - (dx / L) * margin
            end_y = center_y - (dy / L) * margin
            ax.annotate('', xy=(end_x, end_y), xytext=(x, y),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=2.0, color=mid_gray,
                                        alpha=0.65, connectionstyle='arc3,rad=0.2'),
                        zorder=0)

        if label != 'Compression':
            detail_y = y - 0.11 if y > 0.5 else y + 0.11
            ax.text(x, detail_y, detail, ha='center', fontsize=7, style='italic',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85),
                    color=base, zorder=2)

    # Arrow from compression to generalization
    ax.annotate('', xy=(center_x, center_y - 0.12), xytext=(0.5, 0.22 + 0.08),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2.4, color=mid_gray),
                zorder=0)

    # Draw center text last so arrows do not cover it
    ax.text(center_x, center_y, 'Good\nGeneralization', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', transform=ax.transAxes,
            zorder=4)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    ax.set_title('Compression $\\leftrightarrow$ Generalization: The Unifying Principle',
                 fontsize=12, fontweight='bold', pad=16)

    insight = ('All frameworks converge:\n'
              'Models that compress well generalize well')
    ax.text(0.5, 0.02, insight, ha='center', fontsize=10, transform=ax.transAxes,
            color=base, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.95, edgecolor=base, linewidth=1.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_9_12_unifying_view.pdf',
                dpi=DPI, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_9_12_unifying_view.png',
                dpi=DPI, bbox_inches='tight')
    print("✓ Figure 9.12 saved")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chapter 9: Compression and Generalization - Figure Generation")
    print("=" * 70)
    print()

    figure_9_1()
    figure_9_2()
    figure_9_3()
    figure_9_4()
    figure_9_5()
    figure_9_6()
    figure_9_7()
    figure_9_8()
    figure_9_9()
    figure_9_10()
    figure_9_11()
    figure_9_12()

    print()
    print("=" * 70)
    print("✓ All figures generated successfully!")
    print("Output location: figures/ch09_pacbayes_mdl/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 70)
    print()
    print("Figure Summary:")
    print("  9.1:  The Generalization Gap")
    print("  9.2:  PAC-Bayes Bound Decomposition")
    print("  9.3:  KL as Information/Codelength")
    print("  9.4:  Two-Part MDL Code")
    print("  9.5:  Flat vs Sharp Minima (3D Loss Landscape)")
    print("  9.6:  Hessian Eigenvalues and Flatness Measures")
    print("  9.7:  Connection Between Flatness and PAC-Bayes")
    print("  9.8:  SGD Noise and Implicit Regularization")
    print("  9.9:  Risk-Coverage Tradeoff")
    print("  9.10: Conformal Prediction Sets")
    print("  9.11: SAM (Sharpness-Aware Minimization)")
    print("  9.12: Compression-Generalization Unifying View")
    print()
