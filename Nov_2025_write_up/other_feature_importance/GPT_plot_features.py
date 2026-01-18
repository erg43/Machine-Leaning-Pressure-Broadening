from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.ensemble import VotingRegressor
# Added necessary imports for the custom classes
from sklearn.base import BaseEstimator, RegressorMixin, clone

# ---- CONFIG ----
RUN_DIR: str | None = None
LOCAL = True
BASE = (Path.home() / "Desktop" / "line_broadening.nosync" / "Scratch") if LOCAL else (Path.home() / "Scratch")
MYRIAD = True

# ---- CUSTOM CLASSES (Must be defined before loading) ----

class SampleWeightIgnoringRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps an sklearn regressor that does NOT support sample_weight,
    so that VotingRegressor can call fit(..., sample_weight=...) safely.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        # Ignore sample_weight, just fit the underlying estimator
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class WeightRouter(BaseEstimator, RegressorMixin):
    """
    Wraps a Pipeline to route 'sample_weight' to a specific step's fit_params.
    VotingRegressor passes 'sample_weight', but Pipeline expects 'step_name__sample_weight'.
    """

    def __init__(self, estimator, step_name):
        self.estimator = estimator
        self.step_name = step_name

    def fit(self, X, y, sample_weight=None):
        # Create a fresh copy of the pipeline
        self.estimator_ = clone(self.estimator)

        # Prepare the kwargs: map generic 'sample_weight' to 'step_name__sample_weight'
        fit_params = {}
        if sample_weight is not None:
            fit_params[f"{self.step_name}__sample_weight"] = sample_weight

        # Fit with the routed parameters
        self.estimator_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


# ---- Helpers ----
def pick_latest_run(base: Path) -> Path:
    if MYRIAD:
        cands = sorted((p for p in base.glob("myriad_other_broadeners_*") if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        cands = sorted((p for p in base.glob("other_broadeners_*") if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No run dirs found under {base}")
    return cands[0]


def find_voter(model):
    """Return (voter, features_len) for Pipeline(VotingRegressor) or bare VotingRegressor."""
    # sklearn Pipeline has .steps = [(name, obj), ...]
    if hasattr(model, "steps"):
        # find the VotingRegressor step
        for name, step in model.steps:
            if isinstance(step, VotingRegressor):
                return step
    # bare voter
    return model


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:180]


# ---- Locate run ----
out_dir = Path(RUN_DIR).expanduser() if RUN_DIR else pick_latest_run(BASE)
plots_dir = out_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
print(f"Using run: {out_dir}")

# ---- Load feature names ----
feat_path = out_dir / "features.csv"
if not feat_path.exists():
    raise FileNotFoundError(f"features.csv not found at {feat_path}")
feature_names = pd.read_csv(feat_path)["feature"].tolist()

# ---- Load models ----
model_files = sorted(out_dir.glob("model_*.joblib"))
if not model_files:
    # fallback to .pkl if joblib not used
    model_files = sorted(out_dir.glob("model_*.pkl"))
if not model_files:
    raise FileNotFoundError(f"No model_*.joblib or model_*.pkl in {out_dir}")

print(f"Found {len(model_files)} models")

# ---- Compute fold-wise feature importances from the voting ensemble ----
fold_importances = []  # list of 1D arrays (n_features,)
fold_labels = []

for mf in model_files:
    model = load(mf)
    voter = find_voter(model)

    # Collect base estimators that expose feature_importances_
    names = getattr(voter, "named_estimators_", {}) or {}
    weights = voter.weights if voter.weights is not None else [1.0] * len(names)

    parts = []
    used_weights = []

    # Iterate over the sub-estimators
    for (name, est), w in zip(names.items(), weights):
        # Try to get feature importances
        fi = getattr(est, "feature_importances_", None)

        # If the estimator is wrapped (e.g., WeightRouter or SampleWeightIgnoringRegressor),
        # we might need to look inside self.estimator_
        if fi is None and hasattr(est, "estimator_"):
            fi = getattr(est.estimator_, "feature_importances_", None)

        # NOTE: 
        # 1. HistGradientBoostingRegressor: No intrinsic feature_importances_.
        # 2. MLPRegressor: No feature_importances_.
        # 3. SVR (Nystroem): Has coef_, but they map to 500 kernel components, not original features.
        # These will be skipped, so the plot reflects RF/AdaBoost only.
        if fi is None:
            continue

        fi = np.asarray(fi, dtype=float)
        if fi.size != len(feature_names):
            # This might happen if using SVR coefs (size 500) vs features (e.g. size 50)
            print(f"Skipping {name}: feature vector size {fi.size} != feature names {len(feature_names)}")
            continue

        parts.append(w * fi)
        used_weights.append(w)

    if not parts:
        print(f"Warning: No base estimator with valid feature_importances_ found in {mf.name}. Skipping this fold.")
        continue

    combined = np.sum(parts, axis=0) / (np.sum(used_weights) if np.sum(used_weights) > 0 else 1.0)
    # normalise to sum=1 for comparability
    s = combined.sum()
    if s > 0:
        combined = combined / s

    fold_importances.append(combined)
    fold_labels.append(mf.stem.replace("model_", ""))

if not fold_importances:
    print("No feature importances could be extracted from any model. Exiting.")
    exit()

arr = np.vstack(fold_importances)  # shape (n_folds, n_features)
mean_imp = arr.mean(axis=0)
std_imp = arr.std(axis=0)

# ---- Save numeric table ----
tbl = pd.DataFrame({
    "feature": feature_names,
    "mean_importance": mean_imp,
    "std_importance": std_imp,
})
tbl.sort_values("mean_importance", ascending=False).to_csv(out_dir / "feature_importances_mean_std.csv", index=False)

# ---- Plot: top-K mean ± std ----
top_k = min(30, mean_imp.size)
order = np.argsort(mean_imp)[-top_k:][::-1]

plt.figure(figsize=(9, 7))
ypos = np.arange(top_k)
plt.barh(ypos, mean_imp[order], xerr=std_imp[order], align="center", capsize=3)
plt.yticks(ypos, np.array(feature_names)[order])
plt.xlabel("Mean importance (normalised)")
plt.title("VotingRegressor feature importance (mean ± std across folds)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(plots_dir / "feature_importances_mean_std.png", dpi=180)
plt.savefig(plots_dir / "feature_importances_mean_std.pdf", bbox_inches="tight")
plt.close()

# ---- Optional: per-fold bars for the top-K features ----
fig = plt.figure(figsize=(max(8, 0.5 * len(fold_labels) + 4), 0.5 * top_k + 3))
ax = plt.gca()
for i, (lab, vec) in enumerate(zip(fold_labels, arr)):
    ax.plot(vec[order], ypos, marker="o", linestyle="-", label=lab)
ax.set_yticks(ypos)
ax.set_yticklabels(np.array(feature_names)[order])
ax.invert_yaxis()
ax.set_xlabel("Importance (normalised)")
ax.set_title("Per-fold feature importances (top-K)")
ax.legend(loc="best", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(plots_dir / "feature_importances_per_fold.png", dpi=180)
plt.close()

print(f"Saved tables and plots to: {plots_dir}")