"""
Figure generation utilities for Chapter 11 (Information-Sound Systems).
Produces the figures for the chapter.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse, FancyArrowPatch, Circle, Patch
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from scipy.special import softmax
import os

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# LaTeX font configuration for publication quality
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans']})
rc('text', usetex=True)
# Keep text in sans-serif for book consistency, but avoid forcing sans-serif math
# (it can drop glyphs like \Delta depending on the TeX font setup).
rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}')
rc('axes', labelsize=12, titlesize=12)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)

# Color palette - colorblind friendly
COLOR_PRIMARY = '#1b9e77'     # Teal (neutral/informational)
COLOR_SECONDARY = '#d95f02'   # Burnt orange (caution / additional safeguards)
COLOR_TERTIARY = '#7570b3'    # Indigo (trusted / positive outcome)
COLOR_QUATERNARY = '#e7298a'  # Magenta (escalation / risk)
COLOR_GRAY = '#7f7f7f'
COLOR_LIGHT_PRIMARY = '#ccece6'
COLOR_LIGHT_SECONDARY = '#fdd0a2'
COLOR_LIGHT_TERTIARY = '#dadaeb'
COLOR_LIGHT_BLUE = COLOR_LIGHT_PRIMARY
COLOR_LIGHT_ORANGE = COLOR_LIGHT_SECONDARY
COLOR_PURPLE = '#9467bd'

# Standard figure size for book (width in inches)
FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0

# DPI for high quality
DPI = 300

# Create output directory (relative to this script so LaTeX includes are stable)
# LaTeX expects: code/figures/ch11_optimization_geometry/...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures', 'ch11_optimization_geometry')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Utility to add consistent subfigure labels."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='left', va='bottom')

# ============================================================================
# FIGURE 11.1: Retrieval Decision - Prior vs Posterior Entropy
# ============================================================================

def figure_11_1():
    """Show how retrieval reduces entropy for different query types"""
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*2.0, FIGURE_HEIGHT*0.95))
    plt.subplots_adjust(wspace=0.38, bottom=0.15)
    decision_text_size = 10
    alpha_prior = 0.7
    alpha_post = 0.8

    # Panel (a): High confidence - "Capital of France"
    ax1 = axes[0]
    cities = ['Paris', 'Lyon', 'Marseille', 'Nice', 'Other']
    prior_a = np.array([0.99, 0.003, 0.003, 0.002, 0.002])
    posterior_a = np.array([0.995, 0.002, 0.001, 0.001, 0.001])

    x_pos = np.arange(len(cities))
    width = 0.35

    H_prior_a = -np.sum(prior_a * np.log(prior_a + 1e-10))
    H_post_a = -np.sum(posterior_a * np.log(posterior_a + 1e-10))
    bars1 = ax1.bar(x_pos - width/2, prior_a, width, label='Prior',
                    color=COLOR_GRAY, alpha=alpha_prior, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, posterior_a, width, label='Posterior',
                    color=COLOR_TERTIARY, alpha=alpha_post, edgecolor='black')

    ax1.set_ylabel('Probability')
    ax1.set_title('(a) High Confidence\n$\\Delta H = 0.032$ nats', fontsize=10.5, weight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=9, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.05])
    ax1.margins(x=0.08)


    # Panel (b): Uncertain - "Population of Paris"
    ax2 = axes[1]

    # Use discrete ranges for consistency with bar charts
    population_edges = np.array([-np.inf, 3, 5, 7, 9, np.inf])
    population_labels = [
        r'$<3\,\mathrm{M}$',
        r'$3\text{--}5\,\mathrm{M}$',
        r'$5\text{--}7\,\mathrm{M}$',
        r'$7\text{--}9\,\mathrm{M}$',
        r'$>9\,\mathrm{M}$',
    ]
    x_pos_b = np.arange(len(population_labels))

    prior_cdf = stats.norm.cdf(population_edges, loc=7, scale=3)
    posterior_cdf = stats.norm.cdf(population_edges, loc=2.16, scale=0.3)
    prior_b = np.diff(prior_cdf)
    posterior_b = np.diff(posterior_cdf)

    H_prior_b = -np.sum(prior_b * np.log(prior_b + 1e-10))
    H_post_b = -np.sum(posterior_b * np.log(posterior_b + 1e-10))

    bars_prior_b = ax2.bar(x_pos_b - width/2, prior_b, width, label='Prior',
                           color=COLOR_GRAY, alpha=alpha_prior, edgecolor='black')
    bars_post_b = ax2.bar(x_pos_b + width/2, posterior_b, width, label='Posterior',
                          color=COLOR_SECONDARY, alpha=alpha_post, edgecolor='black')

    ax2.set_xlabel('Population (millions)')
    ax2.set_ylabel('Probability')
    ax2.set_title('(b) Uncertain Query\n$\\Delta H = 1.533$ nats', fontsize=10.5, weight='bold')
    ax2.set_xticks(x_pos_b)
    ax2.set_xticklabels(population_labels, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=9, framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 0.65])



    # Panel (c): No prior - "Order status"
    ax3 = axes[2]

    statuses = ['Pending', 'Shipped', 'Delivered', 'Cancelled', 'Other']
    prior_c = np.ones(5) / 5  # Uniform - no information
    posterior_c = np.array([0, 1, 0, 0, 0])  # Deterministic after retrieval

    x_pos = np.arange(len(statuses))

    H_prior_c = -np.sum(prior_c * np.log(prior_c + 1e-10))
    H_post_c = 0  # Deterministic

    bars3 = ax3.bar(x_pos - width/2, prior_c, width, label='Prior',
                    color=COLOR_GRAY, alpha=alpha_prior, edgecolor='black')
    bars4 = ax3.bar(x_pos + width/2, posterior_c, width, label='Posterior',
                    color=COLOR_QUATERNARY, alpha=alpha_post, edgecolor='black')

    ax3.set_ylabel('Probability')
    ax3.set_title('(c) No Prior Knowledge\n$\\Delta H = 1.609$ nats', fontsize=10.5, weight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(statuses, rotation=45, ha='right', fontsize=9)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=9, framealpha=0.9, ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])


    # Add decision boxes inside each subplot at the top
    ax1.text(0.5, 0.95, 'Skip retrieval', transform=ax1.transAxes, ha='center', va='top',
             fontsize=8.5, weight='bold', color=COLOR_TERTIARY,
             bbox=dict(boxstyle='round,pad=0.25', facecolor=COLOR_LIGHT_TERTIARY, edgecolor=COLOR_TERTIARY, linewidth=1.5))
    ax2.text(0.5, 0.95, 'Offer retrieval', transform=ax2.transAxes, ha='center', va='top',
             fontsize=8.5, weight='bold', color=COLOR_SECONDARY,
             bbox=dict(boxstyle='round,pad=0.25', facecolor=COLOR_LIGHT_SECONDARY, edgecolor=COLOR_SECONDARY, linewidth=1.5))
    ax3.text(0.5, 0.95, 'Require retrieval', transform=ax3.transAxes, ha='center', va='top',
             fontsize=8.5, weight='bold', color=COLOR_QUATERNARY,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fbe0ec', edgecolor=COLOR_QUATERNARY, linewidth=1.5))

    fig.text(0.5, 0.01,
             'Entropy reduction ($\\Delta H$) guides retrieval decision: low $\\Delta H$ suggests skipping, high $\\Delta H$ suggests retrieval adds value.',
             ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_1_retrieval_entropy.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_1_retrieval_entropy.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.1 saved")
    plt.close()

# ============================================================================
# FIGURE 11.2A: Unconstrained Generation
# ============================================================================

def figure_11_2a():
    """Show unconstrained generation space"""
    np.random.seed(42)

    fig, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH*0.9, FIGURE_HEIGHT*0.9))

    # Panel (a): Unconstrained generation - valid vs everything else
    valid_x = np.random.normal(0, 0.6, 55)
    valid_y = np.random.normal(0, 0.6, 55)

    waste_x = np.concatenate([
        np.random.uniform(-3.5, 3.5, 60),
        np.random.normal(2.4, 0.7, 30),
        np.random.normal(-2.4, 0.7, 30)
    ])
    waste_y = np.concatenate([
        np.random.uniform(-3.5, 3.5, 60),
        np.random.normal(-2.2, 0.7, 30),
        np.random.normal(2.3, 0.7, 30)
    ])

    # Plot invalid cluster first so valid points sit on top
    ax1.scatter(waste_x, waste_y, color=COLOR_GRAY, s=45, alpha=0.6,
                label='Invalid/irrelevant', zorder=1)
    ax1.scatter(valid_x, valid_y, color=COLOR_TERTIARY, s=70, alpha=0.85,
                edgecolor='black', linewidth=0.6, label='Valid outputs', zorder=3)

    circle = Circle((0, 0), 3.5, fill=False, edgecolor='black', linewidth=2.0, linestyle=(0, (6, 3)), alpha=1.0)
    ax1.add_patch(circle)
    ax1.set_xlabel('Semantic relevance', fontsize=11)
    ax1.set_ylabel('Syntactic correctness', fontsize=11)
    ax1.set_title('Unconstrained Generation: High Entropy', fontsize=11.5, weight='bold')
    ax1.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.95)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim([-4, 4])
    ax1.set_ylim([-4, 4])
    ax1.set_aspect('equal')

    ax1.text(0, -3.6, '27\\% useful — utility diluted',
             ha='center', fontsize=9, style='italic', color=COLOR_QUATERNARY,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor=COLOR_QUATERNARY, linewidth=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_2a_unconstrained.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_2a_unconstrained.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.2a saved")
    plt.close()

# ============================================================================
# FIGURE 11.2B: Constrained Generation
# ============================================================================

def figure_11_2b():
    """Show constrained generation space"""
    np.random.seed(42)

    fig, ax2 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH*0.9, FIGURE_HEIGHT*0.9))

    constrained_x = np.random.normal(0, 0.45, 110)
    constrained_y = np.random.normal(0, 0.45, 110)

    ax2.scatter(constrained_x, constrained_y, color=COLOR_TERTIARY, s=70, alpha=0.8,
                edgecolor='black', linewidth=0.6, label='Valid outputs', zorder=2)

    limited_region = Circle((0, 0), 1.25, fill=False, edgecolor=COLOR_PRIMARY, linewidth=3,
                            label='Schema constraint')
    ax2.add_patch(limited_region)

    ax2.set_xlabel('Semantic relevance', fontsize=11)
    ax2.set_ylabel('Syntactic correctness', fontsize=11)
    ax2.set_title('Constrained Generation: Low Entropy', fontsize=11.5, weight='bold')
    ax2.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.95)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_aspect('equal')

    ax2.text(0, -3.6, '100\\% of outputs are useful — schema collapses search space',
             ha='center', fontsize=10, style='italic', color=COLOR_TERTIARY,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.95, edgecolor=COLOR_TERTIARY, linewidth=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_2b_constrained.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_2b_constrained.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.2b saved")
    plt.close()

# ============================================================================
# FIGURE 11.3: Three Types of Drift
# ============================================================================

def figure_11_3():
    """Visualize distribution and calibration drift with clearer weighting"""
    np.random.seed(42)

    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH*1.3, FIGURE_HEIGHT*1.6), sharex=True)
    plt.subplots_adjust(hspace=0.28)

    days = np.arange(0, 90)

    # Panel (a): Input Distribution Drift
    ax1 = axes[0]

    # Mahalanobis distance - stable then drifts
    mmd = np.ones(90) * 0.5
    mmd[30:] = 0.5 + 0.03 * (days[30:] - 30) + np.random.normal(0, 0.05, 60)
    mmd = np.clip(mmd, 0, 3)

    threshold_input = 1.5

    ax1.plot(days, mmd, color=COLOR_PRIMARY, linewidth=2.8, label='Distribution distance')
    ax1.fill_between(days, 0, mmd, alpha=0.15, color=COLOR_PRIMARY)
    ax1.axhline(threshold_input, color=COLOR_QUATERNARY, linestyle='--', linewidth=2.5,
                label='Alert threshold')

    # Mark alert point
    alert_day = np.where(mmd > threshold_input)[0][0]
    ax1.plot(alert_day, mmd[alert_day], 'o', color=COLOR_QUATERNARY, markersize=9, zorder=5)
    ax1.annotate('Alert triggered', xy=(alert_day, mmd[alert_day]),
                 xytext=(alert_day - 8, 2.5),
                 fontsize=9.5,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=COLOR_QUATERNARY, linewidth=1.5),
                 arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=1.5))

    ax1.set_ylabel('Distribution Distance')
    ax1.set_title('(a) Input Distribution Drift', fontsize=11, weight='bold')
    legend_input = ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 90])
    ax1.set_ylim([0, 3])
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Panel (b): Calibration Drift
    ax2 = axes[1]

    # ECE increases over time
    ece = (0.02 + np.random.normal(0, 0.003, 30)).tolist()
    ece += (0.02 + 0.003 * (days[30:60] - 30) + np.random.normal(0, 0.005, 30)).tolist()
    ece += (0.11 + 0.002 * (days[60:] - 60) + np.random.normal(0, 0.008, 30)).tolist()
    ece = np.clip(np.array(ece), 0, 0.2)

    ax2.plot(days, ece, color=COLOR_SECONDARY, linewidth=2.5, label='Calibration error')
    ax2.fill_between(days, 0, ece, alpha=0.18, color=COLOR_SECONDARY)
    ax2.axhline(0.05, color=COLOR_QUATERNARY, linestyle='--', linewidth=2.5,
                label='Threshold')

    guardrail_pause_day = 60
    ax2.axvline(guardrail_pause_day, color='black', linestyle=':', linewidth=1.6, alpha=0.6)
    ax2.annotate('Model refresh paused', xy=(guardrail_pause_day, 0.11), xytext=(guardrail_pause_day - 15, 0.16),
                 fontsize=9.5, ha='right', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='black', linewidth=1),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Expected Calibration Error')
    ax2.set_title('(b) Calibration Drift', fontsize=11, weight='bold')
    legend_calib = ax2.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 90])
    ax2.set_ylim([0, 0.2])
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.text(0.5, 0.02,
             'Monitor distribution shifts and calibration degradation to trigger model updates.',
             ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_3_drift_monitoring.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_3_drift_monitoring.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.3 saved")
    plt.close()

# ============================================================================
# FIGURE 11.4: Confidence-Based Routing Flow
# ============================================================================

def figure_11_4():
    """Confidence-based routing flow - clean vertical design"""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*0.9, FIGURE_HEIGHT*1.4))
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 14])
    ax.axis('off')

    # Query arrives
    start_box = FancyBboxPatch((2, 12.5), 6, 1, boxstyle="round,pad=0.15",
                               edgecolor='black', facecolor=COLOR_LIGHT_PRIMARY, linewidth=2.5)
    ax.add_patch(start_box)
    ax.text(5, 13, 'Query Arrives', ha='center', va='center', fontsize=12, weight='bold')

    # Vertical arrow
    ax.annotate('', xy=(5, 11.5), xytext=(5, 12.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # Decision diamond
    ax.text(5, 11, 'Estimate Confidence', ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2.5))

    # Vertical spacing for 4 paths
    paths = [
        {'y': 8.5, 'name': 'Primary', 'conf': '$c \\geq 0.8$', 'action': 'Direct answer',
         'color': COLOR_LIGHT_TERTIARY, 'edgecolor': COLOR_TERTIARY, 'metrics': '98\\% acc, 1$\\times$ cost'},
        {'y': 6.0, 'name': 'Fallback 1', 'conf': '$0.5 \\leq c < 0.8$', 'action': 'Add schema hints',
         'color': COLOR_LIGHT_SECONDARY, 'edgecolor': COLOR_SECONDARY, 'metrics': '94\\% acc, 3$\\times$ cost'},
        {'y': 3.5, 'name': 'Fallback 2', 'conf': '$0.3 \\leq c < 0.5$', 'action': 'Retrieve evidence',
         'color': COLOR_LIGHT_SECONDARY, 'edgecolor': COLOR_SECONDARY, 'metrics': '95\\% acc, 5$\\times$ cost'},
        {'y': 1.0, 'name': 'Abstain', 'conf': '$c < 0.3$', 'action': 'Human review',
         'color': '#fbe0ec', 'edgecolor': COLOR_QUATERNARY, 'metrics': 'Human oversight'}
    ]

    for path in paths:
        # Draw box
        box = FancyBboxPatch((1, path['y']), 8, 1.8, boxstyle="round,pad=0.15",
                             edgecolor=path['edgecolor'], facecolor=path['color'], linewidth=2.5)
        ax.add_patch(box)

        # Path name
        ax.text(1.5, path['y'] + 1.4, f"\\textbf{{{path['name']}}}", ha='left', va='top',
                fontsize=11, weight='bold')

        # Confidence range
        ax.text(1.5, path['y'] + 0.9, path['conf'], ha='left', va='center', fontsize=10.5)

        # Action
        ax.text(1.5, path['y'] + 0.3, path['action'], ha='left', va='center', fontsize=10)

        # Metrics
        ax.text(8.5, path['y'] + 0.9, path['metrics'], ha='right', va='center',
                fontsize=9, style='italic', color=COLOR_GRAY)

        # Arrow from decision
        ax.annotate('', xy=(5, path['y'] + 1.8), xytext=(5, 10),
                    arrowprops=dict(arrowstyle='->', lw=2, color=path['edgecolor'], alpha=0.7))

    fig.text(0.5, 0.02, 'Confidence-based routing allocates queries to appropriate handling paths.',
             ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_4_routing_flow.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_4_routing_flow.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.4 saved")
    plt.close()

# ============================================================================
# FIGURE 11.5: Token Budget Allocation
# ============================================================================

def figure_11_5():
    """Show token budget allocation strategies for different task types"""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.2, FIGURE_HEIGHT))

    # Categories
    categories = ['Retrieval-heavy\\\\ (Q\\&A)', 'Balanced\\\\ (Chat assistant)', 'Generation-heavy\\\\ (Creative writing)']

    # Components
    components = ['System prompt', 'Retrieved context', 'Examples/History', 'Query', 'Response']
    colors_comp = [COLOR_GRAY, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_PURPLE]

    # Data for each scenario (percentages that sum to 100)
    scenario1 = [6, 68, 12, 3, 11]   # Retrieval-heavy
    scenario2 = [5, 10, 40, 5, 40]   # Balanced
    scenario3 = [8, 0, 15, 2, 75]    # Generation-heavy

    data = [scenario1, scenario2, scenario3]

    # Create stacked horizontal bars
    y_pos = np.arange(len(categories))
    left = np.zeros(len(categories))

    for i, (component, color) in enumerate(zip(components, colors_comp)):
        values = [d[i] for d in data]
        bars = ax.barh(y_pos, values, left=left, height=0.6, color=color, edgecolor='black',
                       linewidth=0.8, label=component, alpha=0.8)

        # Add percentage labels for segments > 5%
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val >= 8:
                token_count = int(round(val * 20))
                if val >= 40:
                    label = f'{int(val)}\\%\n(~{token_count} tokens)'
                else:
                    label = f'{int(val)}\\%'
                ax.text(left[j] + val/2, bar.get_y() + bar.get_height()/2,
                        label, ha='center', va='center', fontsize=10.5, weight='bold')

        left += values

    # Add total budget line
    ax.axvline(100, color='black', linestyle='--', linewidth=2.5, alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Percentage of Token Budget (\\%)')
    ax.set_title('Token Budget Allocation Strategies', fontsize=12, weight='bold')
    legend_tokens = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.95)
    legend_tokens.get_frame().set_facecolor('white')
    legend_tokens.get_frame().set_edgecolor('#bbbbbb')
    legend_tokens.get_frame().set_linewidth(0.8)
    ax.grid(True, alpha=0.2, axis='x')
    for marker in (25, 50, 75):
        ax.axvline(marker, color='#d0d0d0', linestyle='--', linewidth=0.8, zorder=0)
    ax.set_xlim([0, 110])
    ax.set_ylim([-0.6, len(categories) - 0.4])

    fig.text(0.5, 0.02,
             'Percentages reflect a 2000-token cap across three example deployments (customer support, agentic assistant, creative writing).',
             ha='center', fontsize=10.5)

    plt.tight_layout(rect=[0, 0.16, 0.98, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_5_token_budget.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_5_token_budget.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.5 saved")
    plt.close()


# ============================================================================
# FIGURE 11.6: Routing Performance Metrics
# ============================================================================

def figure_11_6():
    """Summarize cost, latency, and accuracy for each routing path"""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.3, FIGURE_HEIGHT*1.1))
    ax.axis('off')

    headers = ['Path', 'Confidence', 'Safeguard action', 'Latency', 'Accuracy', 'Cost']
    rows = [
        ['Primary', '$\\geq 0.8$', 'Direct answers', '0.18 s', '98\\%', '$1\\times$'],
        ['Fallback 1', '$0.5{-}0.8$', 'Schema + retry', '0.55 s', '94\\%', '$3\\times$'],
        ['Fallback 2', '$0.3{-}0.5$', 'Retrieve + retry', '0.95 s', '95\\%', '$5\\times$'],
        ['Abstain', '$< 0.3$', 'Human review', 'Variable', 'Human', 'Human']
    ]

    # Adjust column widths for better fit
    col_widths = [0.12, 0.14, 0.26, 0.14, 0.13, 0.11]
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center',
                     colLoc='center', loc='center', colWidths=col_widths, bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.35
        if row == 0:
            cell.set_fontsize(10)
            cell.set_facecolor(COLOR_LIGHT_PRIMARY)
            cell.set_text_props(weight='bold', color='black')
            cell.set_edgecolor('#aaaaaa')
            cell.set_linewidth(0.9)
        else:
            cell.set_fontsize(9.5)
            cell.set_edgecolor('#b5b5b5')
            cell.set_linewidth(0.8)
            if col == 0:
                colors = [COLOR_TERTIARY, COLOR_SECONDARY, COLOR_SECONDARY, COLOR_QUATERNARY]
                fill_colors = [COLOR_LIGHT_TERTIARY, COLOR_LIGHT_SECONDARY, COLOR_LIGHT_SECONDARY, '#fde2f2']
                cell.set_facecolor(fill_colors[row-1])
                cell.set_text_props(color=colors[row-1], weight='bold')
            else:
                cell.set_facecolor('white')
            if col in (3, 4, 5):
                cell.get_text().set_ha('right')
            elif col == 2:
                cell.get_text().set_ha('left')
            else:
                cell.get_text().set_ha('center')

    ax.text(0.5, 1.06, 'Routing Performance Summary', ha='center', fontsize=12, weight='bold',
            transform=ax.transAxes)
    fig.text(0.5, 0.02,
             'Blended outcome: 95\\% accuracy at 2.8$\\times$ baseline cost.',
             ha='center', fontsize=10, color=COLOR_GRAY)

    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_6_routing_metrics.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_6_routing_metrics.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.6 saved")
    plt.close()


# ============================================================================
# FIGURE 11.7: Feedback Loop Error Amplification
# ============================================================================

def figure_11_7():
    """Show how errors amplify through feedback loops over iterations"""
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH*1.7, FIGURE_HEIGHT*1.05))
    plt.subplots_adjust(wspace=0.28, hspace=0.24)
    axes = axes.flatten()

    answers = ['Correct', 'Error A', 'Error B', 'Error C', 'Other']
    x_pos = np.arange(len(answers))

    distributions = [
        (0, np.array([0.45, 0.25, 0.12, 0.08, 0.10])),
        (1, np.array([0.30, 0.40, 0.12, 0.08, 0.10])),
        (2, np.array([0.22, 0.55, 0.10, 0.07, 0.06])),
        (3, np.array([0.15, 0.70, 0.06, 0.04, 0.05])),
        (4, np.array([0.08, 0.85, 0.03, 0.02, 0.02])),
        (5, np.array([0.03, 0.95, 0.01, 0.005, 0.005]))
    ]
    notes = {
        0: 'Initial cache: correct answer leads',
        1: 'Recent retrievals highlight Error A',
        2: 'Context reuse reinforces Error A',
        3: 'Fallbacks now favor Error A',
        4: 'Correct answer rarely resurfacing',
        5: 'System converges on the wrong answer'
    }
    accent_colors = {
        0: COLOR_TERTIARY,
        1: COLOR_SECONDARY,
        2: COLOR_SECONDARY,
        3: COLOR_QUATERNARY,
        4: COLOR_QUATERNARY,
        5: COLOR_QUATERNARY
    }

    legend_added = False
    for ax, (t, probs) in zip(axes, distributions):
        colors = [COLOR_TERTIARY if label == 'Correct' else COLOR_QUATERNARY if label == 'Error A' else COLOR_GRAY
                  for label in answers]
        bars = ax.bar(x_pos, probs, color=colors, alpha=0.82, edgecolor='black', linewidth=1.1)

        for bar, prob in zip(bars, probs):
            if prob > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{prob*100:.0f}\\%', ha='center', fontsize=9.5, weight='bold')

        if t in (0, 3):
            ax.set_ylabel('Probability')
        # Remove redundant per-panel titles to reduce clutter
        ax.set_xticks(x_pos)
        ax.set_xticklabels(answers, rotation=0, ha='center', fontsize=9)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_ylim([0, 1.05])
        ax.set_yticks([0.0, 0.5, 1.0])
        # Put annotation as title
        ax.set_title(f'$t={t}$: {notes[t]}', fontsize=9.5, pad=8,
                    color=accent_colors[t], weight='bold')
        if not legend_added and t == 0:
            legend_handles = [Patch(facecolor=COLOR_TERTIARY, edgecolor='black', label='Correct'),
                               Patch(facecolor=COLOR_QUATERNARY, edgecolor='black', label='Error A'),
                               Patch(facecolor=COLOR_GRAY, edgecolor='black', label='Other')]
            legend = ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                             framealpha=0.95, fontsize=8.8, ncol=3)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('#bbbbbb')
            legend_added = True

    fig.text(0.5, 0.02,
             'Feedback loops amplify errors: probability mass shifts from correct answer to Error A over iterations.',
             ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(
        os.path.join(OUTPUT_DIR, 'fig_11_7_feedback_amplification.pdf'),
        dpi=DPI,
        bbox_inches='tight',
        pad_inches=0.08,
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, 'fig_11_7_feedback_amplification.png'),
        dpi=DPI,
        bbox_inches='tight',
        pad_inches=0.08,
    )
    print("✓ Figure 11.7 saved")
    plt.close()

# ============================================================================
# FIGURE 11.8: The Four-Stage Architecture
# ============================================================================

def figure_11_8():
    """Schematic diagram of the four-stage pattern with information flow"""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH*1.6, FIGURE_HEIGHT*1.1))
    ax.set_xlim([0, 16])
    ax.set_ylim([0, 10])
    ax.axis('off')

    # Stage boxes
    stages = [
        {'x': 0.5, 'name': 'ENCODE', 'color': COLOR_PRIMARY,
         'ops': ['Tokenize', 'Embed', 'Classify intent', 'Extract entities'],
         'info': 'Extract key information\nfrom input'},
        {'x': 4.2, 'name': 'BUDGET', 'color': COLOR_SECONDARY,
         'ops': ['Estimate difficulty', 'Route fast/slow', 'Decide retrieval', 'Allocate tokens'],
         'info': 'Allocate effort based on\npredicted difficulty'},
        {'x': 7.9, 'name': 'VERIFY', 'color': COLOR_TERTIARY,
         'ops': ['Schema validation', 'Type/range checks', 'Confidence check', 'Trigger fallbacks'],
         'info': 'Validate outputs before\nrelease'},
        {'x': 11.6, 'name': 'ACT', 'color': COLOR_PURPLE,
         'ops': ['Return to user', 'Update database', 'Call APIs', 'Log metrics'],
         'info': 'Deliver actions with\nmonitored impact'}
    ]

    for i, stage in enumerate(stages):
        # Main stage box (increased height for readability)
        box_y = 4.1
        box_w, box_h = 3.2, 5.6
        box = FancyBboxPatch((stage['x'], box_y), box_w, box_h, boxstyle="round,pad=0.15",
                             edgecolor='black', facecolor='white', linewidth=2.5)
        ax.add_patch(box)
        # Colored header band for identification (stick to top of box)
        header = Rectangle((stage['x'], box_y + box_h - 0.95), box_w, 0.45,
                           facecolor=stage['color'], edgecolor='none', alpha=0.8)
        ax.add_patch(header)
        # Stage number badge
        badge = Circle((stage['x'] + 0.35, box_y + box_h + 0.3), 0.24, facecolor='white', edgecolor='black', linewidth=1.2)
        ax.add_patch(badge)
        ax.text(stage['x'] + 0.35, box_y + box_h + 0.3, f'{i + 1}', ha='center', va='center', fontsize=10, weight='bold')

        # Stage name
        ax.text(stage['x'] + 1.6, box_y + box_h - 0.7, f"\\textbf{{{stage['name']}}}",
                ha='center', fontsize=12, weight='bold')

        # Operations - left-aligned for better readability
        for j, op in enumerate(stage['ops']):
            ax.text(stage['x'] + 0.3, box_y + box_h - 1.3 - j*0.72, f'• {op}',
                    ha='left', fontsize=10.5, va='top', color='black')

        # Information theory note
        info_box = FancyBboxPatch((stage['x'] + 0.2, box_y + 0.4), 2.8, 1.4,
                                  boxstyle="round,pad=0.08",
                                  edgecolor=stage['color'], facecolor='white',
                                  linewidth=2.0, alpha=0.98)
        ax.add_patch(info_box)
        ax.text(stage['x'] + 1.6, box_y + 1.1, stage['info'],
                ha='center', va='center', fontsize=10.5, style='italic', color='black')

        # Forward arrow (except for last stage)
        if i < len(stages) - 1:
            # add small gaps so arrows don't touch boxes
            connector_y = box_y + box_h - 0.25  # keep connectors away from bullet text
            ax.annotate(
                '',
                xy=(stages[i + 1]['x'] + 0.06, connector_y),
                xytext=(stage['x'] + box_w - 0.06, connector_y),
                arrowprops=dict(arrowstyle='->', lw=2.8, color='black'),
            )

    # Feedback arrow from VERIFY to BUDGET (fallback loop)
    # Draw well below the stage boxes to avoid overlapping with information text
    # Information boxes are at y=4.5-5.9, so place arrow at y=2.0 for clear separation
    arrow_y = 2.0
    ax.annotate('', xy=(stages[1]['x'] + box_w/2, arrow_y), xytext=(stages[2]['x'] + box_w/2, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=3.0, color=COLOR_QUATERNARY,
                                linestyle='dashed'))
    ax.text((stages[1]['x'] + stages[2]['x'] + box_w) / 2, arrow_y - 0.5, 'Fallback loop',
            ha='center', fontsize=10.5, style='italic', color=COLOR_QUATERNARY)

    # Input/Output labels
    ax.text(0.5, 2.8, '\\textbf{Input:}\\\\ Raw query', ha='left', fontsize=11.5,
           bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_LIGHT_BLUE,
                    edgecolor='black', linewidth=1.5))

    ax.text(14.8, 2.8, '\\textbf{Output:}\\\\ Effects in world', ha='right', fontsize=11.5,
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6d5ff',
                    edgecolor='black', linewidth=1.5))

    # Title at top
    fig.suptitle('The Four-Stage Architecture for Robust AI Systems', fontsize=13, weight='bold', y=0.98)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_8_four_stage_architecture.pdf'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_8_four_stage_architecture.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Figure 11.8 saved")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Chapter 11 Figures: Designing Information-Sound Systems")
    print("=" * 60)
    print()

    figure_11_1()
    figure_11_2a()
    figure_11_2b()
    figure_11_3()
    figure_11_4()
    figure_11_5()
    figure_11_6()
    figure_11_7()
    figure_11_8()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("Output location: code/figures/ch11_optimization_geometry/")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 60)
    print()
    print("Figure Summary:")
    print("  11.1: Retrieval Decisions and Entropy Reduction")
    print("  11.2a: Unconstrained Generation")
    print("  11.2b: Constrained Generation")
    print("  11.3: Drift Monitoring (Distribution & Calibration)")
    print("  11.4: Confidence-Based Routing Flow")
    print("  11.5: Token Budget Allocation Strategies")
    print("  11.6: Routing Performance Metrics")
    print("  11.7: Feedback Amplification")
    print("  11.8: Four-Stage Architecture")
