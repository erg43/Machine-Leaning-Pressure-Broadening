import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path
import pandas as pd

BASE = Path.home() / "Desktop" / "line_broadening.nosync" / "Scratch" / "other_broadeners_2025-11-14_add_HDO"

model_filename_H2O_CO2 = "model_Test_CH3Br_air_N2_O2-OCS_CO2-CH3_air_N2_O2-NO_air_N2_O2-OCS_air_N2_O2-O2_air_N2_O2-H2O_H2O-N2_H2-CS_CS-CO_CO-HDO_H2O-HI_air_N2_O2-HCl_H2-HBr_air_N2_O2-CO2_CO2-H2O_CO2-HCl_CO-SO3_SO3-HOCl_HOCl-HBr_HBr.joblib"
model_filename_air = "model_Test_HI_HI-CH3F_SF6-HCl_HCl-O2_H2O-ClO_ClO-OH_H2-HF_air_N2_O2-O2_CO2-CO2_air_N2_O2-CO_air_N2_O2-H2O_air_N2_O2-SO3_air_N2_O2-HF_CO2-CO_H2-CH3F_NH3-CS_air_N2_O2-CH4_CH4-NH3_H2-HCN_air_N2_O2-OH_air_N2_O2.joblib"
model_filename_H2 = "model_Test_N2_air_N2_O2-CH4_CO2-CH3F_CH3I-H2O_H2-CH4_0.84 H2 + 0.joblib"

data_test = pd.read_csv(BASE / 'H2O_air')
save_name = 'H2O_air'

model = joblib.load(BASE / model_filename_air)

# 2. Define HDO Parameters
# ------------------------
hdo_quad_rms = np.sqrt((-2.29) ** 2 + (2.55) ** 2 + (-0.26) ** 2)

# 3. Prepare HDO Data Frame
# -------------------------
data_hdo = data_test.copy()

data_hdo["active_weight"] = 19.0216
data_hdo["active_dipole"] = 1.857
data_hdo["active_polar"] = 1.501
data_hdo["active_rms_quadrupole"] = hdo_quad_rms

if "active_m" in data_hdo.columns:
    data_hdo["active_m"] = 2.0

if "d_act_per" in data_hdo.columns:
    data_hdo["d_act_per"] = 285.5

# HDO B0 values
data_hdo["active_B0a"] = 23.38
data_hdo["active_B0b"] = 9.10
data_hdo["active_B0c"] = 6.41

# 4. Common drops and extract H2O literature y, M
# ----------------------------------------------
drop_cols = ['M', 'Ka_aprox', 'Kapp_aprox', 'gamma_uncertainty', 'T', 'profile',
             'active_d', 'perturber_d', 'Unnamed: 0']

# Keep M, y from the original H2O data for plotting
M = data_test['M'].to_numpy()
y = data_test['gamma'].to_numpy()

data_h2o = data_test.copy()
data_h2o = data_h2o.drop(columns=[c for c in drop_cols if c in data_h2o.columns])
data_hdo = data_hdo.drop(columns=[c for c in drop_cols if c in data_hdo.columns])

# 5. Build feature matrices in the same order
# -------------------------------------------
try:
    feat_cols = model.feature_names_in_
except AttributeError:
    feat_cols = [c for c in data_h2o.columns if c not in ["gamma", "pair", "fractional_error", "weight"]]

X_h2o = data_h2o[feat_cols].to_numpy()
X_hdo = data_hdo[feat_cols].to_numpy()

# Predict (model output is log(gamma))
y_pred_log_h2o = model.predict(X_h2o)
yhat_h2o = np.exp(y_pred_log_h2o)

y_pred_log_hdo = model.predict(X_hdo)
yhat_hdo = np.exp(y_pred_log_hdo)

# 6. Plotting
# -----------
if save_name.split("_")[0] == "H2O":
    fig = plt.figure(figsize=(9, 4.5), dpi=160)
    ax = plt.gca()

    # H2O literature
    ax.plot(M, y, "x", alpha=0.4, markeredgewidth=1.5,
            label="H2O literature")

    # H2O prediction
    ax.plot(M, yhat_h2o, ".", alpha=0.8, markersize=4,
            label="H2O predicted")

    # HDO prediction
    ax.plot(M, yhat_hdo, "o", alpha=0.8, markersize=4,
            label="HDO predicted")

    # RMSE for HDO vs H2O (for context)
    rmse_iso = np.sqrt(mean_squared_error(y, yhat_hdo))

    y_anchor = np.nanmean(y)
    x_min, x_max = np.nanmin(M), np.nanmax(M)
    x_span = x_max - x_min if x_max != x_min else 1.0
    pos_rmse = x_min - (0.1 * x_span)

    ax.errorbar(pos_rmse, y_anchor, yerr=rmse_iso, fmt='none',
                elinewidth=2.5, capsize=5,
                label=f"HDO shift vs H2O: {rmse_iso:.2g}")

    ax.set_xlabel('M"')
    ax.set_ylabel(r"$\gamma$ [cm$^{-1}$ atm$^{-1}$]")
    ax.set_title(f"H2O / HDO predictions\n{str(save_name)[:60]}")

    ymax = max(np.nanmax(y), np.nanmax(yhat_h2o), np.nanmax(yhat_hdo))
    ax.set_ylim(0, ymax * 1.2 if np.isfinite(ymax) else 1)

    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    fig.tight_layout()

    save_name_hdo = f"scatter_HDO_pred_{save_name}"
    fig.savefig(f"{save_name_hdo}.png")
    plt.close(fig)
