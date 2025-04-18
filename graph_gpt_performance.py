import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.lines import Line2D

# Load your data
df = pd.read_csv('gpt-performance-to-plot.csv')

# Define metrics and their line‐colors
metrics = [
    'Hamming Loss',
    'Jaccard Index',
    'Precision',
    'Recall',
    'F1-score',
    'Exact Match'
]
metric_colors = {
    'Hamming Loss': 'crimson',
    'Jaccard Index': 'plum',
    'Precision': 'hotpink',
    'Recall': 'fuchsia',
    'F1-score': 'royalblue',
    'Exact Match': 'darkviolet'
}

# Create figure & axes
fig, ax = plt.subplots(figsize=(12, 7))

# Switch to a 'coolwarm' colormap (softer extremes)
cmap = mpl.colormaps.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=0.0, vmax=2.0)

# Plot each metric
for metric in metrics:
    ax.plot(
        df['Top-p'],
        df[metric],
        color=metric_colors[metric],
        linewidth=2,
        zorder=1
    )
    ax.scatter(
        df['Top-p'],
        df[metric],
        c=df['Temperature'],
        cmap=cmap,
        norm=norm,
        edgecolors='none',      # remove marker borders
        marker='o',
        s=90,
        alpha=0.9,
        zorder=2
    )

# Force y-axis from 0.0 to 1.0
ax.set_ylim(0.0, 0.9)

# Make axis tick labels smaller
ax.tick_params(axis='both', labelsize=10)

# Add colorbar for temperature scale
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Temperature', fontsize=12)

# Legend for metric lines
legend_lines = [Line2D([0], [0], color=metric_colors[m], lw=2) for m in metrics]
ax.legend(legend_lines, metrics, title='Metrics',
          fontsize=10, title_fontsize=11, loc='upper right')

# Labels, title, grid
ax.set_xlabel('Top-p', fontsize=14)
ax.set_ylabel('Performance', fontsize=14)
ax.set_title('GPT-4o-mini Performance vs Top-p', fontsize=16)
ax.set_xticks(df['Top-p'])
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
