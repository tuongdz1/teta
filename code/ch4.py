"""
Figure generation utilities for Chapter 4 (High-Dimensional Representations).
Produces the ten figures used in the book.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge, Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from scipy import stats
import os
import shutil

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Warm scientific palette
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

FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 4.0
DPI = 300

OUTPUT_DIR = Path(__file__).resolve().parent / 'figures' / 'ch04_highdim_representations'
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


def style_3d_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANEL)
        axis.pane.set_edgecolor(PANEL)


def callout_box(edge: str = COLOR_GRAY, face: str = PAPER) -> dict:
    return dict(boxstyle='round,pad=0.28', facecolor=face, edgecolor=edge, linewidth=1.2, alpha=0.9)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', dpi=DPI)
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=DPI)

# ============================================================================
# FIGURE 4.1: Volume Concentration Near Surface
# ============================================================================

def figure_4_1():
    """Show how volume concentrates in outer shell"""
    dimensions = np.array([2, 10, 100, 1000])
    epsilon = 0.01  # Outer 1% shell

    # Fraction of volume within radius (1-epsilon)
    volume_fraction = (1 - epsilon)**dimensions
    shell_fraction = 1 - volume_fraction

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Visual representation for 2D
    # Draw disk with highlighted shell
    r_outer = 1.0
    r_inner = 1 - epsilon * 10  # Exaggerate for visibility

    ax1.add_patch(Circle((0, 0), r_outer, facecolor=COLOR_LIGHT_BLUE, edgecolor=GRID, linewidth=1.5, alpha=0.35))
    ax1.add_patch(Wedge((0, 0), r_outer, 0, 360, width=r_outer - r_inner, facecolor=COLOR_SECONDARY, alpha=0.35, edgecolor='none'))
    ax1.add_patch(Circle((0, 0), r_inner, facecolor=PANEL, edgecolor='none', alpha=1.0))
    ax1.add_patch(Circle((0, 0), r_outer, fill=False, edgecolor=INK, linewidth=2))

    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_aspect('equal')
    ax1.set_title('(a) 2D: Most area in center\n(exaggerated shell)', fontsize=11)
    # Clarify that the shell is exaggerated for visibility (outer 10% radius => ~19% area in 2D).
    ax1.text(0, 0, 'Exaggerated: outer 10\\% shell\n$\\approx$ 20\\% of area (2D)',
             ha='center', va='center',
             fontsize=10, zorder=10,
             bbox=callout_box(edge=GRID, face=PAPER))
    ax1.axis('off')

    # Panel (b): Fraction of volume in outer shell vs dimension
    line = ax2.semilogy(dimensions, shell_fraction, 'o-', color=COLOR_PRIMARY,
                        linewidth=2.6, markersize=8, markeredgecolor=INK, markeredgewidth=1.2)[0]
    _glow(line, COLOR_PRIMARY, 2.6)

    # Highlight endpoints only
    for d, frac in [(dimensions[0], shell_fraction[0]), (dimensions[-1], shell_fraction[-1])]:
        ax2.text(
            d,
            frac * 1.4,
            f'{frac:.3f}',
            ha='center',
            fontsize=9,
            zorder=10,
            bbox=callout_box(edge=GRID, face=PAPER),
        )

    # Add horizontal line at 1
    ax2.axhline(1, color=COLOR_GRAY, linestyle='--', linewidth=1.6, alpha=0.6)
    ax2.text(220, 1.2, 'All volume', fontsize=9, ha='center', color=COLOR_GRAY)

    ax2.set_xlabel('Dimension $d$', fontsize=11)
    ax2.set_ylabel('Fraction in outer 1\\% shell', fontsize=11)  # Simplified
    ax2.set_title('(b) Volume Concentration vs Dimension', fontsize=11)
    ax2.grid(True, alpha=0.25, which='both')
    ax2.set_xlim([1, 2000])
    ax2.set_ylim([0.01, 2])
    style_axes(ax2)

    # Add annotation
    # Move annotation so it does not overlap the x tick label at 1000
    ax2.annotate('In high dimensions,\nALL volume is near surface',
                xy=(1000, shell_fraction[-1]), xytext=(800, 0.35),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.5),
                fontsize=9, ha='center',
                bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT))

    plt.tight_layout()
    save_figure(fig, 'fig_4_1_volume_concentration')
    print("✓ Figure 4.1 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.2: Random Vector Orthogonality
# ============================================================================

def figure_4_2():
    """Show dot products shrinking with dimension"""
    np.random.seed(42)

    dimensions = [2, 10, 50, 100, 500, 1000]
    n_samples = 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Distribution of dot products for different dimensions
    x_grid = np.linspace(-0.6, 0.6, 400)
    for i, d in enumerate([10, 100, 1000]):
        dot_products = []
        for _ in range(n_samples):
            # Generate random unit vectors
            u = np.random.randn(d)
            v = np.random.randn(d)
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            dot_products.append(np.dot(u, v))

        color = [COLOR_QUATERNARY, COLOR_SECONDARY, COLOR_PRIMARY][i]
        kde = stats.gaussian_kde(dot_products)
        density = kde(x_grid)
        line = ax1.plot(x_grid, density, color=color, linewidth=2.6, label=f'$d={d}$')[0]
        _glow(line, color, 2.6)
        ax1.fill_between(x_grid, 0, density, color=color, alpha=0.15)

    ax1.axvline(0, color=COLOR_GRAY, linestyle='--', linewidth=1.8, alpha=0.7)
    ax1.set_xlabel('Dot product $u^\\top v$', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('(a) Distribution of Random Dot Products', fontsize=11)
    # Pin legend to upper right to keep it off the central bars
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim([-0.5, 0.5])
    style_axes(ax1)

    # Panel (b): Standard deviation vs dimension
    stds = []
    for d in dimensions:
        dot_products = []
        for _ in range(n_samples):
            u = np.random.randn(d)
            v = np.random.randn(d)
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            dot_products.append(np.dot(u, v))
        stds.append(np.std(dot_products))

    # Theoretical: std = 1/sqrt(d)
    theoretical = 1 / np.sqrt(np.array(dimensions))

    line = ax2.loglog(dimensions, stds, 'o-', color=COLOR_PRIMARY, linewidth=2.6,
                      markersize=8, label='Empirical', markeredgecolor=INK, markeredgewidth=1.2)[0]
    _glow(line, COLOR_PRIMARY, 2.6)
    ax2.loglog(dimensions, theoretical, '--', color=COLOR_SECONDARY, linewidth=2.4,
               label='Theory: $1/\\sqrt{d}$')

    ax2.set_xlabel('Dimension $d$', fontsize=11)
    ax2.set_ylabel('Standard deviation of $u^\\top v$', fontsize=11)
    ax2.set_title('(b) Concentration Around Zero', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, which='both')
    style_axes(ax2)

    # Add annotation
    # Move the yellow callout out of the way of the lines
    ax2.annotate('Nearly orthogonal\nfor large $d$',
                xy=(1000, theoretical[-1]), xytext=(25, 0.22),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.5),
                fontsize=9, ha='center',
                bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT))

    plt.tight_layout()
    save_figure(fig, 'fig_4_2_orthogonality')
    print("✓ Figure 4.2 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.3: Distance Uniformity
# ============================================================================

def figure_4_3():
    """Show how distances become uniform in high dimensions"""
    np.random.seed(42)

    dimensions = [2, 10, 50, 100, 500]
    n_points = 100

    # Multi-row layout without empty panels (keeps the figure compact at \linewidth in LaTeX).
    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT * 1.25))
    gs = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    for idx, d in enumerate(dimensions):
        ax = axes[idx]
        # Generate random points
        points = np.random.randn(n_points, d)

        # Compute distances from origin
        distances = np.linalg.norm(points, axis=1)

        # Plot histogram
        ax.hist(
            distances,
            bins=18,
            color=COLOR_PRIMARY,
            alpha=0.65,
            edgecolor=INK,
            linewidth=0.4,
            density=True,
        )

        # Mark mean and std
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Increased linewidth for better visibility
        ax.axvline(mean_dist, color=COLOR_SECONDARY, linewidth=2.2, linestyle='--')
        ax.axvspan(mean_dist - std_dist, mean_dist + std_dist, alpha=0.18, color=COLOR_SECONDARY)

        ax.set_xlabel('Distance', fontsize=10)
        if idx in {0, 3}:
            ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'$d={d}$ (CV={std_dist/mean_dist:.3f})', fontsize=10)
        ax.grid(True, alpha=0.25)
        style_axes(ax)

    fig.tight_layout(pad=0.6)
    save_figure(fig, 'fig_4_3_distance_uniformity')
    print("✓ Figure 4.3 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.4: Gaussian Thin Shell
# ============================================================================

def figure_4_4():
    """Visualize norm concentration for Gaussian samples"""
    np.random.seed(42)

    dimensions = [10, 100, 1000]
    n_samples = 5000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Distribution of norms
    colors = [COLOR_QUATERNARY, COLOR_SECONDARY, COLOR_PRIMARY]
    all_norms = []
    for d in dimensions:
        x = np.random.randn(n_samples, d)
        all_norms.append(np.linalg.norm(x, axis=1))

    # Use a shared x-grid across all dimensions; previously this was based only on the first
    # dimension's samples, which hid the d=100 and d=1000 curves.
    x_min = min(np.min(norms) for norms in all_norms)
    x_max = max(np.max(norms) for norms in all_norms)
    x_grid = np.linspace(x_min, x_max, 600)

    for i, d in enumerate(dimensions):
        norms = all_norms[i]
        kde = stats.gaussian_kde(norms)
        density = kde(x_grid)
        line = ax1.plot(x_grid, density, color=colors[i], linewidth=2.6, label=f'$d={d}$')[0]
        _glow(line, colors[i], 2.6)
        ax1.fill_between(x_grid, 0, density, color=colors[i], alpha=0.15)

        mean_norm = np.sqrt(d)
        ax1.axvline(mean_norm, color=colors[i], linestyle='--', linewidth=2.0, alpha=0.7)

    ax1.set_xlabel('Norm $\\|x\\|$', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('(a) Norm Concentration in Thin Shell', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)
    style_axes(ax1)

    # Panel (b): Relative standard deviation vs dimension
    dims = np.logspace(0.5, 3.5, 20).astype(int)
    rel_stds = []

    for d in dims:
        norms = []
        for _ in range(1000):
            x = np.random.randn(d)
            norms.append(np.linalg.norm(x))

        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        rel_stds.append(std_norm / mean_norm)

    # Theoretical: 1/sqrt(2d)
    theoretical = 1 / np.sqrt(2 * dims)

    line = ax2.loglog(dims, rel_stds, 'o', color=COLOR_PRIMARY, markersize=6,
                      label='Empirical', alpha=0.75, markeredgecolor=INK, markeredgewidth=0.8)[0]
    _glow(line, COLOR_PRIMARY, 2.4)
    ax2.loglog(dims, theoretical, '--', color=COLOR_SECONDARY, linewidth=2.4,
               label='Theory: $1/\\sqrt{2d}$')

    ax2.set_xlabel('Dimension $d$', fontsize=11)
    ax2.set_ylabel('Relative std: $\\sigma(\\|x\\|) / \\mathbb{E}[\\|x\\|]$', fontsize=11)
    ax2.set_title('(b) Shell Thickness Shrinks with $d$', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, which='both')
    style_axes(ax2)

    # Add annotation
    # Move the yellow callout away from the plotted lines
    ax2.annotate('Samples concentrate\nin thin shell',
                xy=(1000, theoretical[-1]), xytext=(25, 0.2),
                arrowprops=dict(arrowstyle='->', color=COLOR_GRAY, lw=1.5),
                fontsize=9, ha='center',
                bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT))

    plt.tight_layout()
    save_figure(fig, 'fig_4_4_gaussian_shell')
    print("✓ Figure 4.4 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.5: Johnson-Lindenstrauss Lemma
# ============================================================================

def figure_4_5():
    """Demonstrate distance preservation under random projection"""
    np.random.seed(42)

    # Generate random points in high dimension
    d_orig = 1000
    n_points = 50
    points = np.random.randn(n_points, d_orig)

    # Compute original pairwise distances
    orig_dists = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            orig_dists.append(np.linalg.norm(points[i] - points[j]))
    orig_dists = np.array(orig_dists)

    # Project to different target dimensions - reduced from 4 to 3 for clarity
    target_dims = [50, 200, 500]

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*2, FIGURE_HEIGHT))

    for idx, k in enumerate(target_dims):
        # Random projection matrix
        R = np.random.randn(k, d_orig) / np.sqrt(k)

        # Project points
        projected = points @ R.T

        # Compute projected distances
        proj_dists = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                proj_dists.append(np.linalg.norm(projected[i] - projected[j]))
        proj_dists = np.array(proj_dists)

        # Plot distance ratio to avoid the "empty plot" effect from distance concentration in high d.
        ratios = proj_dists / orig_dists

        axes[idx].scatter(
            orig_dists,
            ratios,
            alpha=0.45,
            s=26,
            color=COLOR_PRIMARY,
            edgecolor=INK,
            linewidth=0.4,
        )

        # Add ±10% bounds around 1
        epsilon = 0.1
        x = np.linspace(orig_dists.min(), orig_dists.max(), 200)
        axes[idx].axhline(1.0, color=COLOR_GRAY, linewidth=2.0, alpha=0.7, linestyle='--')
        axes[idx].axhline(1.0 - epsilon, color=COLOR_QUATERNARY, linewidth=1.4, alpha=0.5, linestyle='--')
        axes[idx].axhline(1.0 + epsilon, color=COLOR_QUATERNARY, linewidth=1.4, alpha=0.5, linestyle='--')
        axes[idx].fill_between(x, 1.0 - epsilon, 1.0 + epsilon, alpha=0.10, color=COLOR_QUATERNARY)

        # Compute distortion
        mean_ratio = np.mean(ratios)
        max_distortion = np.max(np.abs(ratios - 1))

        axes[idx].set_xlabel('Original distance', fontsize=10)
        axes[idx].set_ylabel('Projected / original', fontsize=10)
        axes[idx].set_title(f'$k={k}$, max $|r-1|$: {max_distortion:.3f}', fontsize=10)
        axes[idx].grid(True, alpha=0.25)
        axes[idx].set_xlim([orig_dists.min() * 0.98, orig_dists.max() * 1.02])
        axes[idx].set_ylim([0.6, 1.4])
        style_axes(axes[idx])

        # Add text with statistics
        axes[idx].text(
            0.05,
            0.95,
            f'Mean ratio: {mean_ratio:.3f}',
            transform=axes[idx].transAxes,
            va='top',
            fontsize=9,
            bbox=callout_box(edge=GRID, face=PAPER),
        )

    # Use fig.suptitle + tight_layout with a top rect to reserve space for title
    fig.suptitle(
        f'Johnson-Lindenstrauss: {d_orig}D $\\to$ $k$D projection (n={n_points} points)',
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.94])
    save_figure(fig, 'fig_4_5_johnson_lindenstrauss')
    print("✓ Figure 4.5 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.6: Anisotropy in Embeddings
# ============================================================================

def figure_4_6():
    """Visualize anisotropic embeddings concentrated in cone"""
    np.random.seed(42)

    # Generate anisotropic embeddings (2D for visualization)
    n_tokens = 200

    # Common direction (anisotropy)
    mu = np.array([1, 0])

    # Generate embeddings with large component along mu
    embeddings = []
    for _ in range(n_tokens):
        # Large component along mu, small perpendicular component
        parallel = 0.8 + 0.1 * np.random.randn()
        perp = 0.3 * np.random.randn()

        # Perpendicular direction
        mu_perp = np.array([0, 1])

        e = parallel * mu + perp * mu_perp
        # Normalize
        e = e / np.linalg.norm(e)
        embeddings.append(e)

    embeddings = np.array(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Anisotropic embeddings
    ax1.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.55, s=22,
               color=COLOR_PRIMARY, edgecolor=INK, linewidth=0.3)

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), '--', color=GRID, linewidth=1.2, alpha=0.8)

    # Highlight cone of concentration
    cone = Wedge((0, 0), 1.05, -20, 20, facecolor=COLOR_LIGHT_ORANGE, alpha=0.35, edgecolor='none')
    ax1.add_patch(cone)

    # Mark mean direction
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)
    ax1.arrow(0, 0, mean_emb[0]*0.8, mean_emb[1]*0.8,
             head_width=0.1, head_length=0.1, fc=COLOR_QUATERNARY,
             ec=COLOR_QUATERNARY, linewidth=2.5, alpha=0.9)
    ax1.text(mean_emb[0]*0.9, mean_emb[1]*0.9 + 0.2,
            'Mean direction $\\mu$', fontsize=10, color=COLOR_QUATERNARY,
            weight='bold', ha='center', bbox=callout_box(edge=COLOR_QUATERNARY, face=PAPER))

    ax1.set_xlabel('Dimension 1', fontsize=11)
    ax1.set_ylabel('Dimension 2', fontsize=11)
    ax1.set_title('(a) Anisotropic Embeddings\n(concentrated in cone)', fontsize=11)
    ax1.grid(True, alpha=0.25)
    ax1.set_aspect('equal')
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    style_axes(ax1)

    # Panel (b): After centering
    centered = embeddings - np.mean(embeddings, axis=0)
    # Renormalize
    centered = centered / np.linalg.norm(centered, axis=1, keepdims=True)

    ax2.scatter(centered[:, 0], centered[:, 1], alpha=0.55, s=22,
               color=COLOR_TERTIARY, edgecolor=INK, linewidth=0.3)

    # Draw unit circle
    ax2.plot(np.cos(theta), np.sin(theta), '--', color=GRID, linewidth=1.2, alpha=0.8)

    # Mark new mean (should be near zero) - adjusted position
    mean_centered = np.mean(centered, axis=0)
    ax2.plot(0, 0, 'o', markersize=10, color=COLOR_QUATERNARY,
            markeredgecolor=INK, markeredgewidth=1.6)
    ax2.text(0, -0.2, 'Mean $\\approx 0$', fontsize=10, color=COLOR_QUATERNARY,
            weight='bold', ha='center', bbox=callout_box(edge=COLOR_QUATERNARY, face=PAPER))

    ax2.set_xlabel('Dimension 1', fontsize=11)
    ax2.set_ylabel('Dimension 2', fontsize=11)
    ax2.set_title('(b) After Centering\n(more isotropic)', fontsize=11)
    ax2.grid(True, alpha=0.25)
    ax2.set_aspect('equal')
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    style_axes(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig_4_6_anisotropy')
    print("✓ Figure 4.6 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.7: Whitening Transformation
# ============================================================================

def figure_4_7():
    """Show effect of whitening on embedding distribution"""
    np.random.seed(42)

    # Generate correlated 2D data
    n = 300

    # Covariance matrix (elongated in one direction)
    cov = np.array([[4, 2], [2, 1.2]])

    # Generate data
    data = np.random.multivariate_normal([0, 0], cov, n)

    # Whiten: X_white = Sigma^{-1/2} X
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sigma_inv_sqrt = eigenvectors @ np.diag(1/np.sqrt(eigenvalues)) @ eigenvectors.T
    data_whitened = (data @ sigma_inv_sqrt.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT))

    # Panel (a): Original data
    ax1.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20,
               color=COLOR_PRIMARY, edgecolor=INK, linewidth=0.3)

    # Draw covariance ellipse
    # Eigenvalues and eigenvectors for ellipse
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width, height = 2 * np.sqrt(eigenvalues) * 2  # 2 std deviations

    ellipse = Ellipse((0, 0), width, height, angle=angle,
                     facecolor='none', edgecolor=COLOR_QUATERNARY,
                     linewidth=2.2, linestyle='--', label='Covariance')
    ax1.add_patch(ellipse)

    # Draw principal axes
    for i in range(2):
        ax1.arrow(0, 0,
                 eigenvectors[0, i] * np.sqrt(eigenvalues[i]) * 2,
                 eigenvectors[1, i] * np.sqrt(eigenvalues[i]) * 2,
                 head_width=0.3, head_length=0.3,
                 fc=COLOR_SECONDARY, ec=COLOR_SECONDARY,
                 linewidth=2.0, alpha=0.75, zorder=5)

    ax1.set_xlabel('Dimension 1', fontsize=11)
    ax1.set_ylabel('Dimension 2', fontsize=11)
    ax1.set_title('(a) Original Embeddings\n(anisotropic, correlated)', fontsize=11)
    ax1.grid(True, alpha=0.25)
    ax1.set_aspect('equal')
    ax1.set_xlim([-6, 6])
    ax1.set_ylim([-6, 6])
    ax1.legend(fontsize=9)
    style_axes(ax1)

    # Panel (b): Whitened data
    ax2.scatter(data_whitened[:, 0], data_whitened[:, 1], alpha=0.55, s=20,
               color=COLOR_TERTIARY, edgecolor=INK, linewidth=0.3)

    # Draw unit circle (identity covariance)
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(2*np.cos(theta), 2*np.sin(theta), '--', color=COLOR_QUATERNARY, linewidth=2.2,
            label='Identity covariance')

    ax2.set_xlabel('Dimension 1', fontsize=11)
    ax2.set_ylabel('Dimension 2', fontsize=11)
    ax2.set_title('(b) After Whitening\n(isotropic, uncorrelated)', fontsize=11)
    ax2.grid(True, alpha=0.25)
    ax2.set_aspect('equal')
    ax2.set_xlim([-6, 6])
    ax2.set_ylim([-6, 6])
    ax2.legend(fontsize=9)
    style_axes(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig_4_7_whitening')
    print("✓ Figure 4.7 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.8: Spherical Geometry (Angular vs Euclidean)
# ============================================================================

def figure_4_8():
    """Illustrate angular vs Euclidean distance on the unit sphere (2D slice)."""
    theta1 = np.deg2rad(-35)
    theta2 = np.deg2rad(35)
    p1 = np.array([np.cos(theta1), np.sin(theta1)])
    p2 = np.array([np.cos(theta2), np.sin(theta2)])

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH * 1.1, FIGURE_HEIGHT))

    # Sphere cross-section
    ax.add_patch(Circle((0, 0), 1, facecolor=COLOR_LIGHT_BLUE, edgecolor=INK, linewidth=1.6, alpha=0.28))

    # Geodesic (arc) and Euclidean chord
    arc_theta = np.linspace(theta1, theta2, 200)
    arc_line = ax.plot(np.cos(arc_theta), np.sin(arc_theta), color=COLOR_PRIMARY, linewidth=3.0)[0]
    _glow(arc_line, COLOR_PRIMARY, 3.0)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=COLOR_QUATERNARY, linewidth=2.2)

    # Radial lines to show the angle
    ax.plot([0, p1[0]], [0, p1[1]], color=GRID, linewidth=1.2)
    ax.plot([0, p2[0]], [0, p2[1]], color=GRID, linewidth=1.2)
    arc_small = np.linspace(theta1, theta2, 60)
    ax.plot(0.32 * np.cos(arc_small), 0.32 * np.sin(arc_small), color=COLOR_GRAY, linewidth=1.4)

    theta_mid = 0.5 * (theta1 + theta2)
    ax.text(0.42 * np.cos(theta_mid), 0.42 * np.sin(theta_mid), r'$\theta$',
            fontsize=11, color=COLOR_GRAY, ha='center', va='center', weight='bold')

    # Points on the sphere
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=70, color=COLOR_SECONDARY,
               edgecolor=INK, linewidth=1.1, zorder=5)

    # Callouts
    arc_xy = np.array([np.cos(theta_mid), np.sin(theta_mid)])
    chord_mid = 0.5 * (p1 + p2)
    ax.annotate('geodesic (angular)',
                xy=(arc_xy[0], arc_xy[1]),
                xytext=(0.05, 1.08),
                arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=1.4),
                fontsize=9,
                bbox=callout_box(edge=COLOR_PRIMARY, face=PAPER))
    ax.annotate('chord (Euclidean)',
                xy=(chord_mid[0], chord_mid[1]),
                xytext=(-0.15, -0.95),
                arrowprops=dict(arrowstyle='->', color=COLOR_QUATERNARY, lw=1.4),
                fontsize=9,
                bbox=callout_box(edge=COLOR_QUATERNARY, face=PAPER))

    ax.set_title('Angular vs Euclidean distance on the sphere', fontsize=11)
    ax.set_aspect('equal')
    ax.set_xlim([-1.35, 1.35])
    ax.set_ylim([-1.2, 1.2])
    ax.axis('off')

    plt.tight_layout()
    save_figure(fig, 'fig_4_8_spherical_geometry')
    print("✓ Figure 4.8 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.9: Manifold Structure in High Dimensions
# ============================================================================

def figure_4_9():
    """Show a low-dimensional manifold embedded in a higher-dimensional space."""
    rng = np.random.default_rng(7)

    # Swiss roll manifold parameters
    t_min, t_max = 1.5 * np.pi, 4.5 * np.pi
    u_min, u_max = -1.0, 1.0

    # Surface grid
    t_grid = np.linspace(t_min, t_max, 100)
    u_grid = np.linspace(u_min, u_max, 40)
    T, U = np.meshgrid(t_grid, u_grid)
    X = T * np.cos(T)
    Y = U
    Z = T * np.sin(T)

    # Sample points on the manifold
    # Keep point count modest so the 3D panel remains readable when scaled into the PDF.
    n_samples = 400
    t = rng.uniform(t_min, t_max, n_samples)
    u = rng.uniform(u_min, u_max, n_samples)
    x = t * np.cos(t)
    y = u
    z = t * np.sin(t)

    norm_vals = (t - t_min) / (t_max - t_min)
    colors = WARM_CMAP(norm_vals)
    surface_colors = WARM_CMAP((T - t_min) / (t_max - t_min))

    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.7, FIGURE_HEIGHT))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot_surface(
        X,
        Y,
        Z,
        facecolors=surface_colors,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.45,
    )
    ax1.scatter(x, y, z, c=colors, s=10, alpha=0.65, edgecolor='none')

    # Highlight two points and intrinsic path
    t1, t2 = 2.2 * np.pi, 4.1 * np.pi
    u1 = 0.35
    p1 = np.array([t1 * np.cos(t1), u1, t1 * np.sin(t1)])
    p2 = np.array([t2 * np.cos(t2), u1, t2 * np.sin(t2)])

    t_path = np.linspace(t1, t2, 140)
    x_path = t_path * np.cos(t_path)
    y_path = np.full_like(t_path, u1)
    z_path = t_path * np.sin(t_path)
    geo_line = ax1.plot(x_path, y_path, z_path, color=COLOR_PRIMARY, linewidth=2.8)[0]
    _glow(geo_line, COLOR_PRIMARY, 2.8)

    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
             linestyle='--', color=COLOR_QUATERNARY, linewidth=2.0)
    ax1.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                s=70, color=COLOR_SECONDARY, edgecolor=INK, linewidth=1.1, zorder=6)

    ax1.set_title('(a) Curved manifold in ambient space', fontsize=11)
    ax1.view_init(elev=18, azim=45)
    ax1.set_box_aspect((1.6, 0.7, 1.3))
    style_3d_axes(ax1)

    legend_lines = [
        Line2D([0], [0], color=COLOR_PRIMARY, lw=2.6),
        Line2D([0], [0], color=COLOR_QUATERNARY, lw=2.0, linestyle='--'),
    ]
    ax1.legend(legend_lines, ['geodesic (on manifold)', 'ambient chord'],
               loc='upper left', fontsize=8, framealpha=0.9)

    # Intrinsic coordinates (unrolled)
    ax2.scatter(t, u, c=colors, s=18, alpha=0.6, edgecolor='none')
    ax2.plot([t1, t2], [u1, u1], color=COLOR_PRIMARY, linewidth=2.8)
    ax2.scatter([t1, t2], [u1, u1], s=55, color=COLOR_SECONDARY,
                edgecolor=INK, linewidth=1.1, zorder=5)

    ax2.set_xlabel('Intrinsic coordinate $t$', fontsize=11)
    ax2.set_ylabel('Intrinsic coordinate $u$', fontsize=11)
    ax2.set_title('(b) Unrolled intrinsic coordinates', fontsize=11)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim([t_min - 0.4, t_max + 0.4])
    ax2.set_ylim([u_min - 0.2, u_max + 0.2])
    style_axes(ax2)

    ax2.annotate('intrinsic path is straight',
                 xy=((t1 + t2) / 2, u1),
                 xytext=(t1, u1 + 0.55),
                 arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=1.4),
                 fontsize=9,
                 bbox=callout_box(edge=COLOR_PRIMARY, face=PAPER))

    plt.tight_layout()
    save_figure(fig, 'fig_4_9_manifold_structure')
    print("✓ Figure 4.9 saved")
    plt.close(fig)

# ============================================================================
# FIGURE 4.10: Product Quantization Concept
# ============================================================================

def figure_4_10():
    """Illustrate product quantization with subspace decomposition"""
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH*1.5, FIGURE_HEIGHT*1.5))

    # Panel (a): Original vector
    d = 32
    x = np.random.randn(d)

    axes[0, 0].bar(range(d), x, color=COLOR_PRIMARY, alpha=0.75, edgecolor=INK, linewidth=0.4)
    axes[0, 0].set_xlabel('Dimension', fontsize=10)
    axes[0, 0].set_ylabel('Value', fontsize=10)
    axes[0, 0].set_title('(a) Original Vector ($d=32$)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    style_axes(axes[0, 0])

    # Panel (b): Split into subspaces
    m = 4  # Number of subspaces
    subspace_dim = d // m

    colors_sub = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_QUATERNARY]

    for i in range(m):
        start = i * subspace_dim
        end = (i + 1) * subspace_dim
        axes[0, 1].bar(range(start, end), x[start:end], color=colors_sub[i],
                      alpha=0.75, edgecolor=INK, linewidth=0.4,
                      label=f'Subspace {i+1}')

    # Draw vertical lines to separate subspaces - more prominent
    for i in range(1, m):
        axes[0, 1].axvline(i * subspace_dim - 0.5, color=GRID,
                          linewidth=2.0, linestyle='--')

    axes[0, 1].set_xlabel('Dimension', fontsize=10)
    axes[0, 1].set_ylabel('Value', fontsize=10)
    axes[0, 1].set_title(f'(b) Split into $m={m}$ Subspaces', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].legend(fontsize=8.8, loc='upper right')
    style_axes(axes[0, 1])

    # Panel (c): Quantize each subspace
    # Generate centroids for each subspace
    k = 8  # Number of centroids per subspace

    # For visualization, show one subspace
    subspace_idx = 0
    subvector = x[subspace_idx*subspace_dim:(subspace_idx+1)*subspace_dim]

    # Generate random centroids
    centroids = np.random.randn(k, subspace_dim) * 0.5

    # Find nearest centroid
    distances = [np.linalg.norm(subvector - c) for c in centroids]
    nearest_idx = np.argmin(distances)

    # Plot subvector and centroids
    axes[1, 0].bar(range(subspace_dim), subvector, color=COLOR_PRIMARY,
                  alpha=0.75, edgecolor=INK, linewidth=0.4, label='Original subvector')
    axes[1, 0].bar(range(subspace_dim), centroids[nearest_idx], color=COLOR_QUATERNARY,
                  alpha=0.55, edgecolor=INK, linewidth=1.2, label=f'Centroid {nearest_idx}')

    axes[1, 0].set_xlabel('Dimension (within subspace)', fontsize=10)
    axes[1, 0].set_ylabel('Value', fontsize=10)
    axes[1, 0].set_title(f'(c) Quantize Subspace 1\n(store ID: {nearest_idx}, not values)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].legend(fontsize=8.8)
    style_axes(axes[1, 0])

    # Panel (d): Compression statistics
    # Original: d * 4 bytes (float32)
    # Compressed: m * 1 byte (uint8 for centroid ID)

    original_bytes = d * 4
    compressed_bytes = m * 1
    compression_ratio = original_bytes / compressed_bytes

    categories = ['Original', 'PQ Compressed']
    sizes = [original_bytes, compressed_bytes]
    colors_bar = [COLOR_PRIMARY, COLOR_TERTIARY]

    bars = axes[1, 1].bar(categories, sizes, color=colors_bar, alpha=0.75,
                         edgecolor=INK, linewidth=1.4)

    # Add value labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{size} bytes', ha='center', va='bottom', fontsize=10, weight='bold')

    axes[1, 1].set_ylabel('Storage (bytes)', fontsize=10)
    axes[1, 1].set_title(f'(d) Compression: {compression_ratio:.0f}× smaller', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, original_bytes * 1.2])
    style_axes(axes[1, 1])

    # Add annotation about tradeoff, placed above bars to avoid occlusion
    axes[1, 1].text(
        0.98,
        0.92,
        f'Tradeoff:\n{compression_ratio:.0f}× compression\nvs quantization error',
        transform=axes[1, 1].transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=callout_box(edge=COLOR_GRAY, face=HIGHLIGHT),
    )

    plt.tight_layout()
    save_figure(fig, 'fig_4_10_product_quantization')
    print("✓ Figure 4.10 saved")
    plt.close(fig)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    configure_matplotlib()
    print("=" * 70)
    print("Generating Chapter 4 figures (High-Dimensional Representations)")
    print("=" * 70)
    print()

    figure_4_1()
    figure_4_2()
    figure_4_3()
    figure_4_4()
    figure_4_5()
    figure_4_6()
    figure_4_7()
    figure_4_8()
    figure_4_9()
    figure_4_10()

    print()
    print("=" * 70)
    print("All figures generated successfully!")
    print(f"Output location: {OUTPUT_DIR}")
    print("Formats: PDF (vector) and PNG (high-res raster)")
    print("=" * 70)
    print()
    print("Figure Summary:")
    print("  4.1: Volume Concentration Near Surface")
    print("  4.2: Random Vector Orthogonality")
    print("  4.3: Distance Uniformity in High Dimensions")
    print("  4.4: Gaussian Thin Shell (norm concentration)")
    print("  4.5: Johnson-Lindenstrauss Lemma (distance preservation)")
    print("  4.6: Anisotropy in Embeddings")
    print("  4.7: Whitening Transformation Effect")
    print("  4.8: Spherical Geometry (Angular vs Euclidean)")
    print("  4.9: Manifold Structure in High Dimensions")
    print("  4.10: Product Quantization Concept")
    print()
    print("Key adjustments:")
    print("  • Standardized font sizes")
    print("  • Clearer dashed lines and markers")
    print("  • Tuned legend positioning and sizing")
    print("  • Reduced visual clutter")
    print("  • More prominent annotations and labels")
    print("  • Balanced transparency for overlaps")
    print("=" * 70)
