import pandas as pd
import numpy as np
import os
from glob import iglob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.lines as mlines
import pickle
from pathlib import Path
import re
from sklearn.metrics import mean_squared_error
from math import sqrt

# ==========================================
# 1. CONFIGURATION
# ==========================================
ORACLE_PATH_STR = '/Users/elizabeth/Desktop/line_broadening.nosync/line_broadening/model_search/raw_data'
HOME = Path.home()
BASE_PROJECT = HOME / "Desktop" / "line_broadening.nosync" / "line_broadening"
PICKLE_DIR = BASE_PROJECT / "plots for paper"
SCRATCH_DIR = HOME / "Desktop" / "line_broadening.nosync" / "Scratch"

if not SCRATCH_DIR.exists():
    SCRATCH_DIR = HOME / "Scratch"

OUTPUT_DIR = Path("thesis_scatter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T = 298  # Kelvin

# Colours
OLD_MODEL_COLOR = '#d62728'   # red
NEW_MODEL_COLOR = '#2ca02c'   # green
LIT_COLOR = '#a6cee3'         # pale blue crosses (literature)
RMSE_NEW_COLOR = '#2ca02c'    # green bar (new model)
RMSE_OLD_COLOR = '#d62728'    # blue bar (old model)
UNC_COLOR = '#a6cee3'         # red bar (lit. unc.)

ERROR_COLORS = {
    7: '#d62728',  # Red (<1%)
    6: '#ff7f0e',  # Orange (1-2%)
    5: '#2ca02c',  # Green (2-5%)
    4: '#1f77b4',  # Blue (5-10%)
    3: '#9467bd',  # Purple (10-20%)
    0: 'grey'      # Default
}

plt.rc('font', size=24)

# ==========================================
# 3. LOAD DATA
# ==========================================

# ==========================================
# 4. PLOTTING LOOP
# ==========================================
print("\nSTEP 3: Generating Plots...")

latest_run = sorted([p for p in SCRATCH_DIR.glob("*other_broadeners_*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime, reverse=True)[0]

pred_files = sorted(latest_run.glob("predictions_*.csv"))

for p_file in pred_files:
    try:
        df_new = pd.read_csv(p_file)
    except Exception:
        continue

    if 'pair' not in df_new.columns:
        continue

    for pair_name, group_new in df_new.groupby('pair'):
        str_pair = str(pair_name)

        active_species = re.split(r'[-_]', str_pair, flags=re.IGNORECASE)[0]


        try:
            # 2. New model & errors
            group_new = group_new.copy()

            rmse_new_val = sqrt(mean_squared_error(group_new['y'], group_new['y_pred']))
            group_new['abs_err'] = group_new['y'] * group_new['y_error']
            avg_hitran_error = group_new['abs_err'].mean()

            # ----------------------------------------
            # PLOTTING
            # ----------------------------------------
            fig, ax = plt.subplots(figsize=(18, 6), dpi=300)

            # A. Literature (HITRAN) – pale blue crosses
            ax.plot(group_new['M'], group_new['y'], 'x',
                    color=LIT_COLOR, markersize=6,
                    markeredgewidth=1.8, label='HITRAN data')

            # B. New model – green dots
            ax.plot(group_new['M'], group_new['y_pred'], 'o',
                    color=NEW_MODEL_COLOR, markersize=4,
                    label='Multi perturber prediction')

            # D. Error bars (RMSE new, RMSE old, mean literature uncertainty)
            x_max = group_new['M'].max()
            x_min = group_new['M'].min()
            x_span = x_max - x_min
            y_mean = group_new['y'].mean()

            rmse_new_x = x_min - 0.08 * x_span
            unc_x = x_max + 0.08 * x_span

            rmse_new_handle = ax.errorbar(rmse_new_x, y_mean, yerr=rmse_new_val,
                                          fmt='none', ecolor=RMSE_NEW_COLOR,
                                          elinewidth=3, capsize=8)
            unc_handle = ax.errorbar(unc_x, y_mean, yerr=avg_hitran_error,
                                     fmt='none', ecolor=UNC_COLOR,
                                     elinewidth=3, capsize=8)

            # Axis labels and limits
            ax.set_xlabel('m', fontsize=20)
            ax.set_ylabel(r'$\gamma$ / cm$^{-1}$ atm$^{-1}$', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.set_xlim(x_min - 0.2 * x_span, x_max + 0.2 * x_span)
            # y-axis down to 0
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(bottom=0, top=y_max)

            ax.grid(True, linestyle='-', alpha=0.3)

            # Legend outside the axes on the right
            lit_handle = mlines.Line2D([], [], color=LIT_COLOR, marker='x',
                                       linestyle='None', markersize=6,
                                       label='Literature Data')
            new_handle = mlines.Line2D([], [], color=NEW_MODEL_COLOR, marker='o',
                                       linestyle='None', markersize=6,
                                       label='Model Predictions')
            rmse_new_leg = mlines.Line2D([], [], color=RMSE_NEW_COLOR,
                                         linestyle='-',
                                         label=f'Model RMSE: {rmse_new_val:.1g}')

            unc_leg = mlines.Line2D([], [], color=UNC_COLOR,
                                    linestyle='-',
                                    label=f'Mean Lit. Unc.: {avg_hitran_error:.1g}')

            handles = [lit_handle, new_handle,
                       rmse_new_leg, unc_leg]
            ax.legend(handles=handles,
                      loc='center left',
                      bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0.,
                      fontsize=20,
                      frameon=True)

            clean_name = re.sub(r"[^A-Za-z0-9._]+", "_", str_pair)
            out_path = OUTPUT_DIR / f"Paper_Final_{clean_name}.png"
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"     [+] Saved: {out_path.name}")

        except Exception as e:
            print(f"     [!] Error plotting {pair_name}: {e}")

print("\n--- Finished ---")
