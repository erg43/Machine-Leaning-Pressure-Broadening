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

OUTPUT_DIR = Path("model_plot_for_paper_fixed")
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
# 2. HELPER FUNCTIONS
# ==========================================

def get_hitran_code(fractional_err):
    if fractional_err < 0.01: return 7
    if fractional_err < 0.02: return 6
    if fractional_err < 0.05: return 5
    if fractional_err < 0.10: return 4
    if fractional_err < 0.20: return 3
    return 0


def broadening(m, T, ma, mp, b0):
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = 1.7796e-5 * (m / (m - 2)) * (1 / np.sqrt(T)) * np.sqrt((ma + mp) / (ma * mp)) * b0 ** 2
    return gamma


def process_oracle_data(df, key):
    required_cols = ['gamma_air', 'J', 'Jpp', 'molecule_weight', 'gamma_air-err', 'm', 'findair',
                     'molecule_dipole', 'polar', 'B0a', 'B0b', 'B0c', 'air_weight',
                     'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox']

    if not set(required_cols).issubset(df.columns):
        df.columns = df.columns.str.strip()
        if not set(required_cols).issubset(df.columns):
            return None

    data = df[required_cols].copy().dropna()

    if key == 'C2H6':
        data = data.drop(data[(data['J'] > 20) & (data['gamma_air'] > 0.08)].index)

    branch = data['Jpp'] - data['J']
    data = data.drop(branch[abs(branch) > 2].index)
    branch = data['Jpp'] - data['J']

    data['P'] = np.where(branch == 1, -data['Jpp'], 0)
    data['Q'] = np.where(branch == 0, data['Jpp'], 0)
    data['R'] = np.where(branch == -1, data['Jpp'] + 1, 0)
    data['O'] = np.where(branch == 2, -data['Jpp'], 0)
    data['S'] = np.where(branch == -2, data['Jpp'] + 1, 0)
    data['M'] = data['P'] + data['Q'] + data['R'] + data['O'] + data['S']
    data = data.drop(columns=['P', 'Q', 'R', 'O', 'S'])

    try:
        ma_val = data['molecule_weight'].iloc[0]
        mp_val = data['air_weight'].iloc[0]
        b0_val = data['findair'].iloc[0]
        data['broadness_jeanna'] = broadening(data['m'], T, ma_val, mp_val, b0_val)
    except Exception:
        return None

    return data.drop(columns=['air_weight'])


# ==========================================
# 3. LOAD DATA
# ==========================================
print("STEP 1: Loading Oracle Data...")
molecules_oracle = {}
for f in iglob(os.path.join(ORACLE_PATH_STR, "*.csv")):
    try:
        raw_df = pd.read_csv(f, skipinitialspace=True)
        processed_df = process_oracle_data(raw_df, Path(f).stem)
        if processed_df is not None:
            molecules_oracle[Path(f).stem] = processed_df
    except Exception:
        pass

print(f"Loaded {len(molecules_oracle)} oracle datasets.")

print("STEP 2: Loading Old Models...")
named_voter_models = []
try:
    with open(PICKLE_DIR / 'voter_model_best_voter.pkl', 'rb') as f:
        voter_models = pickle.load(f)
    with open(PICKLE_DIR / 'voter_model_best_voter_data.pkl', 'rb') as f:
        list_data_compare = pickle.load(f)
    for i, model in enumerate(voter_models):
        labels = [l.strip() for l in list_data_compare[i][0].split(',')]
        named_voter_models.append([labels, model])
except Exception as e:
    print(f"Error loading models: {e}")

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
        if "air" not in str_pair.lower():
            continue

        active_species = re.split(r'[-_]air', str_pair, flags=re.IGNORECASE)[0]

        matched_key = next((k for k in molecules_oracle if k.lower() == active_species.lower()), None)
        old_model = next((m for l, m in named_voter_models
                          if any(active_species.lower() == x.lower() for x in l)),
                         None)

        if not matched_key or not old_model:
            continue

        try:
            # 1. Old model predictions
            data_old = molecules_oracle[matched_key]
            X_old = data_old.drop(columns=['gamma_air', 'gamma_air-err', 'errorbar'], errors='ignore')
            y_true_old = data_old['gamma_air']
            y_pred_old = old_model.predict(X_old)

            if len(data_old) > 1000:
                idx = np.random.choice(data_old.index, 1000, replace=False)
                plot_x_old = data_old.loc[idx, 'M']
                plot_y_old = y_pred_old[data_old.index.get_indexer(idx)]
            else:
                plot_x_old = data_old['M']
                plot_y_old = y_pred_old

            rmse_old_val = sqrt(mean_squared_error(y_true_old, y_pred_old))

            # 2. New model & errors
            group_new = group_new.copy()
            group_new['err_code'] = group_new['y_error'].apply(get_hitran_code)

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

            # C. Old model – red dots
            ax.plot(plot_x_old, plot_y_old, 'o',
                    color=OLD_MODEL_COLOR, markersize=4,
                    label='Air broadening prediction')

            # D. Error bars (RMSE new, RMSE old, mean literature uncertainty)
            x_max = group_new['M'].max()
            x_min = group_new['M'].min()
            x_span = x_max - x_min
            y_mean = group_new['y'].mean()

            rmse_new_x = x_min - 0.08 * x_span
            rmse_old_x = x_min - 0.13 * x_span
            unc_x = x_max + 0.08 * x_span

            rmse_new_handle = ax.errorbar(rmse_new_x, y_mean, yerr=rmse_new_val,
                                          fmt='none', ecolor=RMSE_NEW_COLOR,
                                          elinewidth=3, capsize=8)
            rmse_old_handle = ax.errorbar(rmse_old_x, y_mean, yerr=rmse_old_val,
                                          fmt='none', ecolor=RMSE_OLD_COLOR,
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
                                       linestyle='None', markersize=8,
                                       label='HITRAN data')
            new_handle = mlines.Line2D([], [], color=NEW_MODEL_COLOR, marker='o',
                                       linestyle='None', markersize=6,
                                       label='Multi perturber prediction')
            old_handle = mlines.Line2D([], [], color=OLD_MODEL_COLOR, marker='o',
                                       linestyle='None', markersize=6,
                                       label='Air broadening prediction')
            rmse_new_leg = mlines.Line2D([], [], color=RMSE_NEW_COLOR,
                                         linestyle='-',
                                         label=f'Multi perturber RMSE: {rmse_new_val:.1g}')
            rmse_old_leg = mlines.Line2D([], [], color=RMSE_OLD_COLOR,
                                         linestyle='-',
                                         label=f'Air broadening RMSE: {rmse_old_val:.1g}')
            unc_leg = mlines.Line2D([], [], color=UNC_COLOR,
                                    linestyle='-',
                                    label=f'Mean HITRAN Unc.: {avg_hitran_error:.1g}')

            handles = [lit_handle, new_handle, old_handle,
                       rmse_new_leg, rmse_old_leg, unc_leg]
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
            print(f"     [!] Error plotting {matched_key}: {e}")

print("\n--- Finished ---")
