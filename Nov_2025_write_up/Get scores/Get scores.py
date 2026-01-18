import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

# Folder containing predictions_Fold_*.csv
DATA_DIR = pathlib.Path("/Users/elizabeth/Desktop/line_broadening.nosync/Scratch/myriad_other_broadeners_2025-11-27_the_final/")  # <--- UPDATE THIS

# ===============================
# LOAD AND CONCATENATE DATA
# ===============================

files = sorted(DATA_DIR.glob("predictions_Fold_*.csv"))
print(f"Found {len(files)} files")

# ------------------------------------------------------------
# 1. Load and concatenate all prediction CSVs
# ------------------------------------------------------------
# Adjust the pattern or path if needed

dfs = [pd.read_csv(f) for f in files]
all_df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(all_df)}")

# sanity check required columns
required_cols = {"pair", "y", "y_pred", "y_error"}
missing = required_cols - set(all_df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ------------------------------------------------------------
# 2. Compute RMSE% per pair + true vs predicted gamma means
# ------------------------------------------------------------

pair_stats = []
for pair, sub in all_df.groupby("pair"):
    y = sub["y"].values               # true literature gamma
    y_pred = sub["y_pred"].values     # predicted gamma

    rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    mean_true_gamma = np.mean(y)
    mean_pred_gamma = np.mean(y_pred)
    rmse_pct = 100.0 * rmse / mean_true_gamma  # normalized to TRUE literature mean

    pair_stats.append(
        {
            "pair": pair,
            "n_lines": len(sub),
            "rmse_pct": rmse_pct,
            "mean_true_gamma": mean_true_gamma,
            "mean_pred_gamma": mean_pred_gamma,
        }
    )

pair_stats_df = pd.DataFrame(pair_stats).sort_values("rmse_pct").reset_index(drop=True)


print("\n=== Summary over molecule–perturber pairs ===")
print(f"Number of pairs: {len(pair_stats_df)}")
print(f"Mean RMSE% over pairs:   {pair_stats_df['rmse_pct'].mean():.2f}%")
print(f"Median RMSE% over pairs: {pair_stats_df['rmse_pct'].median():.2f}%")
print(
    f"Range RMSE% over pairs:  "
    f"{pair_stats_df['rmse_pct'].min():.2f}% – {pair_stats_df['rmse_pct'].max():.2f}%"
)

# ------------------------------------------------------------
# 3. Histogram of RMSE% per pair
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(pair_stats_df["rmse_pct"], bins=40)
#plt.xscale("log")  # LOG SCALE ON X-AXIS
plt.xlabel("Pair-wise RMSE [% of mean γ]")
plt.ylabel("Count")
plt.title("Distribution of RMSE% per pair")

plt.axvline(pair_stats_df["rmse_pct"].median(), color='red', linestyle='--', label='Median')
plt.axvline(pair_stats_df["rmse_pct"].mean(), color='green', linestyle='--', label='Mean')
plt.legend()

plt.tight_layout()
plt.savefig('RMSE_hist.png')



# ------------------------------------------------------------
# 4. Top-10 best and worst pairs (with true and predicted means)
# ------------------------------------------------------------
print("\n=== 10 best pairs (lowest RMSE%) ===")
print(
    pair_stats_df[["pair", "n_lines", "rmse_pct", "mean_true_gamma", "mean_pred_gamma"]]
    .head(10)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3e}" if abs(x) < 1e-2 else f"{x:.3f}"
    )
)

print("\n=== 10 worst pairs (highest RMSE%) ===")
print(
    pair_stats_df[["pair", "n_lines", "rmse_pct", "mean_true_gamma", "mean_pred_gamma"]]
    .tail(10)
    .iloc[::-1]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3e}" if abs(x) < 1e-2 else f"{x:.3f}"
    )
)

# ------------------------------------------------------------
# 5. Fraction of pairs within quoted uncertainty (pair-averaged)
# ------------------------------------------------------------
# y_error is fractional (e.g. 0.05 = 5%)
df_unc = all_df.copy()

# Drop rows with missing or non-positive uncertainty
df_unc = df_unc.replace([np.inf, -np.inf], np.nan)
df_unc = df_unc.dropna(subset=["y", "y_pred", "y_error"])
df_unc = df_unc[df_unc["y_error"] > 0]

abs_err = (df_unc["y_pred"] - df_unc["y"]).abs()
allowed = df_unc["y_error"] * df_unc["y"].abs()

within = abs_err <= allowed  # boolean per line

# --- NEW: pair-level averaging (each pair equal weight) ---
per_pair_within = (
    pd.DataFrame({"within": within, "pair": df_unc["pair"]})
    .groupby("pair")["within"]
    .mean()               # fraction of lines within uncertainty for this pair
    .rename("within_fraction")
)

mean_pair_fraction = per_pair_within.mean()

print(
    f"\nMean fraction of lines within quoted uncertainty, "
    f"averaged equally over molecule–perturber pairs: "
    f"{mean_pair_fraction*100:.1f}% "
    f"({len(per_pair_within)} pairs)"
)

# Optional: merge back into pair_stats_df for inspection
per_pair_stats = pair_stats_df.merge(
    per_pair_within, on="pair", how="left"
).sort_values("rmse_pct")

print("\n=== Example of per-pair within-uncertainty fractions (first 10 by RMSE%) ===")
print(
    per_pair_stats[["pair", "n_lines", "rmse_pct", "within_fraction"]]
    .head(10)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x),
    )
)

