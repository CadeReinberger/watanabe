import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fit logistic regression to buzz accuracy vs. buzz position.")
parser.add_argument("--bin-width", type=int, default=10,
                    help="Number of words per bin (default: 10)")
parser.add_argument("--stats-dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "../detailed-stats"),
                    help="Root directory containing tournament subdirectories")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
stats_dir = os.path.abspath(args.stats_dir)
buzz_files = glob.glob(os.path.join(stats_dir, "**", "*buzzes.csv"), recursive=True)

if not buzz_files:
    raise FileNotFoundError(f"No buzzes.csv files found under {stats_dir}")

print(f"Found {len(buzz_files)} buzz file(s) across tournaments.")

dfs = []
for path in buzz_files:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    dfs.append(df[["buzz_position", "value"]])

data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=["buzz_position", "value"])

# correct = 1 for 10/15 power, 0 for neg/no-value
data["correct"] = data["value"].isin([10, 15]).astype(int)
# Drop any rows with unexpected values (dead buzzes, etc.)
data = data[data["value"].isin([10, 15, 0, -5])]

print(f"Total buzz observations: {len(data):,}")

# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
X = data["buzz_position"].values.reshape(-1, 1)
y = data["correct"].values

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

beta0 = model.intercept_[0]
beta1 = model.coef_[0][0]

print(f"Logistic regression: β₀ = {beta0:.4f}, β₁ = {beta1:.4f}")

# ---------------------------------------------------------------------------
# Binned proportions
# ---------------------------------------------------------------------------
bin_width = args.bin_width
pos_min = int(data["buzz_position"].min())
pos_max = int(data["buzz_position"].max())

bin_edges = np.arange(pos_min, pos_max + bin_width + 1, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

data["bin"] = pd.cut(data["buzz_position"], bins=bin_edges, labels=bin_centers, right=False)
data["bin"] = data["bin"].astype(float)

binned = (
    data.groupby("bin", observed=True)["correct"]
    .agg(proportion="mean", count="count")
    .reset_index()
)
# Drop bins with very few observations (< 5) to avoid noisy points
binned = binned[binned["count"] >= 5]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
x_smooth = np.linspace(pos_min, pos_max, 500)
y_smooth = expit(beta0 + beta1 * x_smooth)

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter: size proportional to number of buzzes in bin
sizes = np.sqrt(binned["count"]) * 3
sc = ax.scatter(
    binned["bin"], binned["proportion"],
    s=sizes, color="#2c7bb6", alpha=0.75,
    edgecolors="white", linewidths=0.6,
    zorder=3, label="Observed proportion (bin)"
)

# Logistic curve
ax.plot(x_smooth, y_smooth, color="#d7191c", linewidth=2.5,
        label="Logistic regression fit", zorder=4)

# Annotation with fit parameters
sign = "+" if beta1 >= 0 else "−"
param_text = (
    f"Logistic regression:\n"
    f"  $\\beta_0$ = {beta0:.3f}\n"
    f"  $\\beta_1$ = {beta1:.4f}\n"
    f"  $P(\\mathrm{{correct}}) = \\sigma({beta0:.3f} {sign} {abs(beta1):.4f} \\cdot x)$"
)
ax.text(0.97, 0.05, param_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

# Axes and labels
ax.set_xlabel("Buzz position (word index)", fontsize=12)
ax.set_ylabel("Proportion correct", fontsize=12)
ax.set_title("Buzz Accuracy vs. Buzz Position\n(logistic regression over all tournaments)",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(0, 1.05)
ax.set_xlim(pos_min - 5, pos_max + 5)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.grid(axis="x", linestyle=":", alpha=0.25)

# Legend (combine scatter + line)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#d7191c", linewidth=2.5, label="Logistic fit"),
    ax.scatter([], [], s=60, color="#2c7bb6", alpha=0.75, edgecolors="white",
               linewidths=0.6, label=f"Binned proportion (bin width = {bin_width} words)")
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "buzz_logistic.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()
