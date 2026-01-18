from __future__ import annotations

from pathlib import Path
import json
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone  # needed if wrappers are inside the saved model
import re
from typing import Literal


OUT_DIR: str | None = None

BASE = Path.home() / "Desktop" / "line_broadening.nosync" / "Scratch"
MYRIAD = True

ACTIVE_PATH = Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "Exomol_molecules.csv"
BROAD_DIR = Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "creation_of_air_diet_files" / "diet_files" / "saved"

ACTIVE_MOLECULES = pd.read_csv(ACTIVE_PATH, header=None)[0].values.tolist()
PERTURBER_MOLECULES = ['Ar', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'He', 'N2', 'NH3', 'NO', 'self', 'O2']

# Temperature for broadness_jeanna
T_K = 296.0

# ---- Helpers ----
def pick_latest_run(base: Path) -> Path:
    if MYRIAD:
        candidates = sorted(
            [p for p in base.glob("myriad_other_broadeners_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    else:
        candidates = sorted(
            [p for p in base.glob("other_broadeners_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"No run dirs found under {base}")
    return candidates[0]


def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _safe_name(s: str) -> str:
    return s.replace(",", "__").replace("/", "_").replace("\\", "_")


# ---- Locate run ----
RUN_DIR = Path(OUT_DIR) if OUT_DIR else pick_latest_run(BASE)
print(f"Using run: {RUN_DIR}")

# With the new training code, production bundle lives in RUN_DIR / "production"
MODEL_DIR = RUN_DIR / "production"
if not MODEL_DIR.exists():
    # fallback: allow pointing directly at the production dir
    if (RUN_DIR / "production_model.joblib").exists() or (RUN_DIR / "production_model.pkl").exists():
        MODEL_DIR = RUN_DIR
    else:
        raise FileNotFoundError(
            f"Could not find production bundle. Looked in:\n"
            f"  {RUN_DIR / 'production'}\n"
            f"  {RUN_DIR}\n"
            f"Expected files: production_model.joblib (or .pkl) and production_features.csv"
        )

print(f"Using production model dir: {MODEL_DIR}")


# ----------------------------------------------------------------------
# Molecular constants
# ----------------------------------------------------------------------

PARAMS_PATH = Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "molecule_parameters.csv"
PARAMS = pd.read_csv(PARAMS_PATH, index_col=0)


def get_params(name: str) -> dict:
    if name not in PARAMS.index:
        raise KeyError(f"{name} not found in molecule_parameters.csv")
    row = PARAMS.loc[name]
    return {
        "MASS": float(row["weight"]),
        "DIPOLE": float(row["dipole"]),
        "POLAR": float(row["polar"]),
        "B0A": float(row["B0a"]),
        "B0B": float(row["B0b"]),
        "B0C": float(row["B0c"]),
        "QXX": float(row["quadrupole_xx"]),
        "QYY": float(row["quadrupole_yy"]),
        "QZZ": float(row["quadrupole_zz"]),
        "d": float(row["d"]),
        "m": float(row["m"]),
    }

# ----------------------------------------------------------------------
# Wrappers used in training model
# ----------------------------------------------------------------------

class SampleWeightIgnoringRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps an sklearn regressor that does NOT support sample_weight.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.estimator_ = clone(self.estimator)
        # ignore sample_weight
        self.estimator_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class WeightRouter(BaseEstimator, RegressorMixin):
    """
    Routes sample_weight to a named step inside a Pipeline.
    """
    def __init__(self, estimator, step_name):
        self.estimator = estimator
        self.step_name = step_name

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.estimator_ = clone(self.estimator)
        if sample_weight is not None:
            fit_params = dict(fit_params)
            fit_params[f"{self.step_name}__sample_weight"] = sample_weight
        self.estimator_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class ExpPredictWrapper(BaseEstimator, RegressorMixin):
    """
    Wraps a regressor trained on log(gamma) so predict() returns gamma.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        y_log = self.estimator_.predict(X)
        return np.exp(y_log)

# ----------------------------------------------------------------------
# Feature helpers
# ----------------------------------------------------------------------
def broadening_jeanna(m: float, T: float, ma: float, mp: float, b0: float) -> float:
    """
    Jeanna's semiclassical broadening estimate:

        gamma = 1.7796e-5 * (m/(m-2)) * (1/sqrt(T)) * sqrt((ma+mp)/(ma*mp)) * b0^2
    """
    if m == 2:
        raise ValueError("m cannot be 2 in broadening_jeanna.")
    return float(
        1.7796e-5
        * (m / (m - 2.0))
        * (1.0 / np.sqrt(T))
        * np.sqrt((ma + mp) / (ma * mp))
        * (b0**2)
    )


def rms_quadrupole(qxx: float, qyy: float, qzz: float) -> float:
    return float(np.sqrt((qxx**2 + qyy**2 + qzz**2) / 3.0))


def load_production_bundle(model_dir: Path):
    """
    Loads:
      - production_model.joblib (or .pkl)
      - production_features.csv
      - optional production_metadata.json
    Returns (model, feat_cols, meta_dict|None)
    """
    feat_path = model_dir / "production_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}")

    feat_cols = pd.read_csv(feat_path)["feature"].tolist()

    joblib_path = model_dir / "production_model.joblib"
    pkl_path = model_dir / "production_model.pkl"

    if joblib_path.exists():
        model = joblib.load(joblib_path)
    elif pkl_path.exists():
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Missing production_model.joblib or production_model.pkl in {model_dir}")

    meta_path = model_dir / "production_metadata.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return model, feat_cols, meta


# ----------------------------------------------------------------------
# Feature builder
# IMPORTANT: we will reorder/select columns using production_features.csv
# ----------------------------------------------------------------------
def build_features(active, pert, j_values: np.ndarray, temperature: float = T_K) -> pd.DataFrame:
    """
    Build a *superset* of possible features.
    We'll later select exactly the production feature list.
    """
    n = len(j_values)

    J = j_values['J']
    Jpp = j_values['Jpp']
    Kc_aprox = j_values['Kc_aprox']
    Kcpp_aprox = j_values['Kcpp_aprox']
    Ka_aprox = j_values['Ka_aprox']
    Kapp_aprox = j_values['Kapp_aprox']

    # Active
    active_weight = np.full(n, active["MASS"], dtype=float)
    active_dipole = np.full(n, active["DIPOLE"], dtype=float)
    active_polar = np.full(n, active["POLAR"], dtype=float)
    active_m = np.full(n, active["m"], dtype=float)
    active_d = np.full(n, active["d"], dtype=float)
    active_B0a = np.full(n, active["B0A"], dtype=float)
    active_B0b = np.full(n, active["B0B"], dtype=float)
    active_B0c = np.full(n, active["B0C"], dtype=float)
    active_rms_q = np.full(
        n, rms_quadrupole(active["QXX"], active["QYY"], active["QZZ"]), dtype=float
    )
    active_qxx = np.full(n, active["QXX"], dtype=float)
    active_qyy = np.full(n, active["QYY"], dtype=float)
    active_qzz = np.full(n, active["QZZ"], dtype=float)

    # Perturber
    pert_weight = np.full(n, pert["MASS"], dtype=float)
    pert_dipole = np.full(n, pert["DIPOLE"], dtype=float)
    pert_polar = np.full(n, pert["POLAR"], dtype=float)
    pert_m = np.full(n, pert["m"], dtype=float)
    pert_d = np.full(n, pert["d"], dtype=float)
    pert_B0a = np.full(n, pert["B0A"], dtype=float)
    pert_B0b = np.full(n, pert["B0B"], dtype=float)
    pert_B0c = np.full(n, pert["B0C"], dtype=float)
    pert_rms_q = np.full(
        n, rms_quadrupole(pert["QXX"], pert["QYY"], pert["QZZ"]), dtype=float
    )
    pert_qxx = np.full(n, pert["QXX"], dtype=float)
    pert_qyy = np.full(n, pert["QYY"], dtype=float)
    pert_qzz = np.full(n, pert["QZZ"], dtype=float)

    # Combined features
    m_feature = active_m + pert_m
    d_act_per = np.full(n, 0.5 * (active["d"] + pert["d"]), dtype=float)

    # Jeanna broadening estimate (constant over this grid for a fixed pair & T)
    jeanna_val = broadening_jeanna(
        float(m_feature[0]),
        float(temperature),
        float(active["MASS"]),
        float(pert["MASS"]),
        float(d_act_per[0]),
    )
    broadness_jeanna = np.full(n, jeanna_val, dtype=float)

    # Self flag
    is_self = np.full(n, float(active == pert), dtype=float)

    # If your training included T, include it. If not, harmless to keep as a superset.
    T = np.full(n, float(temperature), dtype=float)

    return pd.DataFrame(
        {
            "J": J,
            "Jpp": Jpp,
            "Kc_aprox": Kc_aprox,
            "Kcpp_aprox": Kcpp_aprox,
            "Ka_aprox": Ka_aprox,
            "Kapp_aprox": Kapp_aprox,
            "T": T,
            "active_B0a": active_B0a,
            "active_B0b": active_B0b,
            "active_B0c": active_B0c,
            "active_dipole": active_dipole,
            "active_m": active_m,
            "active_polar": active_polar,
            "active_quadrupole_xx": active_qxx,
            "active_quadrupole_yy": active_qyy,
            "active_quadrupole_zz": active_qzz,
            "active_rms_quadrupole": active_rms_q,
            "active_weight": active_weight,
            "active_d": active_d,
            "broadness_jeanna": broadness_jeanna,
            "d_act_per": d_act_per,
            "is_self": is_self,
            "m": m_feature,
            "perturber_B0a": pert_B0a,
            "perturber_B0b": pert_B0b,
            "perturber_B0c": pert_B0c,
            "perturber_dipole": pert_dipole,
            "perturber_m": pert_m,
            "perturber_polar": pert_polar,
            "perturber_quadrupole_xx": pert_qxx,
            "perturber_quadrupole_yy": pert_qyy,
            "perturber_quadrupole_zz": pert_qzz,
            "perturber_rms_quadrupole": pert_rms_q,
            "perturber_weight": pert_weight,
            "perturber_d": pert_d,
        }
    )


def align_features(X: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Ensure X has exactly feat_cols, in order, with no extras.
    """
    missing = [c for c in feat_cols if c not in X.columns]
    if missing:
        raise ValueError(
            f"Missing required features for production model: {missing}\n"
            f"Available columns: {sorted(X.columns.tolist())}"
        )
    return X.loc[:, feat_cols].copy()

# ----------------------------------------------------------------------
# Convert ExoMol isotope strings -> plain formula, e.g. "1H-14N-16O3" -> "HNO3"
# ----------------------------------------------------------------------

def strip_isotopes_to_formula(spec: str) -> str:
    """
    Convert e.g.
      '1H-14N-16O3' -> 'HNO3'
      '31P2-1H2'    -> 'P2H2'
    """
    s = spec.strip()

    # retain leading cis-/trans- if present
    prefix = ""
    m_pref = re.match(r"^(cis|trans)-(.*)$", s, flags=re.IGNORECASE)
    if m_pref:
        prefix = m_pref.group(1).lower() + "-"
        s = m_pref.group(2)

    parts = s.split("-")
    out = []
    for part in parts:
        m = re.match(r"^\d*([A-Z][a-z]?)(\d*)$", part.strip())
        if not m:
            continue
        elem, count = m.groups()
        out.append(elem + count)

    return prefix + "".join(out)

def active_from_broad_filename(p: Path) -> str:
    # "trans-31P2-1H2__air.broad" -> "trans-P2H2"
    stem = p.stem
    active_raw = stem.split("__", 1)[0]
    return strip_isotopes_to_formula(active_raw)

# ----------------------------------------------------------------------
# Extract J values from .broad files
# ----------------------------------------------------------------------

RotorType = Literal["prolate", "oblate", "spherical", "linear", "asymmetric"]

def read_qns_from_broad(path: Path, rotor_type: RotorType) -> pd.DataFrame:
    """
    Returns: J, Jpp, Ka_aprox, Kc_aprox, Kapp_aprox, Kcpp_aprox

    Diet codes:
      a0: J''
      a1: J'', K''
      m0: m
      m1: m, K''
      a5: J', Ka', Kc', J'', Ka'', Kc''

    m convention:
      m > 0 : R branch, J'' = m - 1, J = J'' + 1
      m < 0 : P branch, J'' = -m,   J = J'' - 1
      m == 0: ambiguous -> skipped
    """
    rows: list[dict] = []

    def _f(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            code = parts[0]
            qns = parts[3:]  # after recipe, gamma, n

            J = np.nan
            Jpp = np.nan
            Ka_aprox = np.nan
            Kc_aprox = np.nan
            Kapp_aprox = np.nan
            Kcpp_aprox = np.nan

            if code.startswith("a5"):
                # explicit asymmetric-top QNs
                if len(qns) >= 1: J   = _f(qns[0])
                if len(qns) >= 2: Ka_aprox = _f(qns[1])
                if len(qns) >= 3: Kc_aprox = _f(qns[2])
                if len(qns) >= 4: Jpp = _f(qns[3])
                if len(qns) >= 5: Kapp_aprox = _f(qns[4])
                if len(qns) >= 6: Kcpp_aprox = _f(qns[5])

                # if rotor_type is 'asymmetric', fine; otherwise still OK (explicit wins)
                # but require J/Jpp present
                if not (np.isfinite(J) and np.isfinite(Jpp)):
                    continue

            elif code.startswith("a0"):
                if len(qns) >= 1:
                    Jpp = _f(qns[0])
                    J = Jpp + 1
                if not np.isfinite(Jpp):
                    continue

                # need rotor_type mapping: no K provided, so set K=0
                if rotor_type == "asymmetric":
                    raise ValueError(f"{path.name}: found a0 but rotor_type='asymmetric' needs a5 with explicit Ka/Kc.")
                elif rotor_type == "prolate":
                    raise ValueError(f"{path.name}: found a0 but rotor_type='prolate' needs a5 with explicit Ka/Kc.")
                elif rotor_type == "oblate":
                    raise ValueError(f"{path.name}: found a0 but rotor_type='oblate' needs a5 with explicit Ka/Kc.")

                Ka_aprox, Kc_aprox = _map_K_to_KaKc(J, 0, rotor_type)
                Kapp_aprox, Kcpp_aprox = _map_K_to_KaKc(Jpp, 0, rotor_type)

            elif code.startswith("a1"):
                if len(qns) >= 1:
                    Jpp = _f(qns[0])
                Kpp = _f(qns[1]) if len(qns) >= 2 else 0.0

                if not np.isfinite(Jpp):
                    continue

                if rotor_type == "asymmetric":
                    raise ValueError(f"{path.name}: found a1 but rotor_type='asymmetric' needs a5 with explicit Ka/Kc.")

                J = Jpp + 1.0  # default R-branch
                # assume ΔK=1
                K = Kpp + 1

                Ka_aprox, Kc_aprox = _map_K_to_KaKc(J, K, rotor_type)
                Kapp_aprox, Kcpp_aprox = _map_K_to_KaKc(Jpp, Kpp, rotor_type)

            elif code.startswith(("m0", "m1")):
                if len(qns) < 1:
                    continue
                mval = _f(qns[0])
                if path.name.split('__')[0] not in ['1H2', '14N2']:
                    if mval == 0.0:
                        Jpp = J = Ka_aprox = Kc_aprox = Kapp_aprox = Kcpp_aprox = 0
                    if mval > 0:      # R branch
                        Jpp = mval - 1.0
                        J   = Jpp + 1.0
                        Kpp = _f(qns[1]) if (code.startswith("m1") and len(qns) >= 2) else 0.0
                        K = Kpp + 1  # assume ΔK=ΔJ
                    else:             # P branch
                        Jpp = -mval
                        J   = Jpp - 1.0
                        Kpp = _f(qns[1]) if (code.startswith("m1") and len(qns) >= 2) else 0.0
                        K = Kpp - 1  # assume ΔK=ΔJ
                else:     # Quadrupole transition! Bodged to remove negative qns
                    if mval >= 0:      # R branch
                        Jpp = mval
                        J   = Jpp + 2.0
                        Kpp = _f(qns[1]) if (code.startswith("m1") and len(qns) >= 2) else 0.0
                        K = Kpp + 2  # assume ΔK=ΔJ
                    else:             # P branch
                        Jpp = -mval + 1
                        J   = Jpp - 2.0
                        Kpp = _f(qns[1]) if (code.startswith("m1") and len(qns) >= 2) else 0.0
                        K = Kpp - 2  # assume ΔK=ΔJ


                if rotor_type == "asymmetric":
                    Ka_aprox = 0
                    Kc_aprox = K
                    Kapp_aprox = 0
                    Kcpp_aprox = Kpp
                else:
                    Ka_aprox, Kc_aprox = _map_K_to_KaKc(J, K, rotor_type)
                    Kapp_aprox, Kcpp_aprox = _map_K_to_KaKc(Jpp, Kpp, rotor_type)

            else:
                continue

            rows.append({
                "J": J,
                "Jpp": Jpp,
                "Ka_aprox": Ka_aprox,
                "Kc_aprox": Kc_aprox,
                "Kapp_aprox": Kapp_aprox,
                "Kcpp_aprox": Kcpp_aprox,
            })

    df = pd.DataFrame(rows)
    return df

def _map_K_to_KaKc(J: float, K: float, rotor_type: RotorType) -> tuple[float, float]:
    if not np.isfinite(J) or not np.isfinite(K):
        return np.nan, np.nan

    if rotor_type == "prolate":
        return K, J - K
    if rotor_type == "oblate":
        return J - K, K
    if rotor_type == "spherical":
        return J / 2.0, J / 2.0
    if rotor_type == "linear":
        return 0.0, J

    # asymmetric handled only via a5 (explicit Ka/Kc)
    raise ValueError("rotor_type='asymmetric' only valid for a5 (explicit Ka/Kc in file).")

def get_rotor_type(molecule):
    if molecule in ['MgH', 'NaH', 'NiH', 'AlH', 'CrH', 'CaH', 'BeH', 'TiH', 'FeH', 'LiH', 'ScH', 'NH', 'CH', 'OH',
     'HCl', 'SiH', 'SH', 'HF', 'PH', 'HBr', 'VO', 'AlO', 'YO', 'MgO', 'TiO', 'SiO', 'CaO', 'NaO', 'LaO', 'ZrO', 'ScO',
     'CO', 'NO', 'SO', 'PO', 'O2', 'HCN', 'C3', 'OCS', 'PN', 'KCl', 'NaCl', 'LiCl', 'CN', 'C2', 'H2', 'CS', 'CP', 'PS',
     'NS', 'SiS', 'NaF', 'AlCl', 'AlF', 'KF', 'LiF', 'CaF', 'MgF', 'N2', 'SiN', 'CaCl', 'CO2', 'N2O', 'SiO2', 'HBO', 'C2H2']:
        rotor_type = 'linear'
    elif molecule in ['CH3F', 'CH3Cl']:
        rotor_type = 'prolate'
    elif molecule in ['NH3', 'CH3', 'AsH3', 'PF3', 'PH3', 'SO3']:
        rotor_type = 'oblate'
    elif molecule in ['SiH4', 'CH4',]:
        rotor_type = 'spherical'
    elif molecule in ['H2O', 'SO2', 'H2S', 'CaOH', 'KOH', 'NaOH', 'SiH2', 'LiOH', 'HNO3', 'H2O2', 'H2CO', 'C2H4',
     'cis-P2H2', 'trans-P2H2', 'H2CS']:
        rotor_type = 'asymmetric'
    else:
        print(molecule)
        print('NO TYPE')
        rotor_type = 'help'

    return rotor_type

# ----------------------------------------------------------------------
# Extract J values from .broad files
# ----------------------------------------------------------------------

def build_active_broad_index(broad_dir: Path) -> dict[str, Path]:
    """
    Map ACTIVE -> .broad file.
    Assumes exactly one file per active (after isotope stripping).
    """
    index: dict[str, Path] = {}

    for p in broad_dir.glob("*.broad"):
        act = active_from_broad_filename(p)

        if act in index:
            raise ValueError(
                f"Duplicate .broad files for active={act}: "
                f"{index[act].name} and {p.name}"
            )

        index[act] = p

    return index

ACTIVE_TO_BROAD = build_active_broad_index(BROAD_DIR)

# ----------------------------------------------------------------------
# Write new broadening file from old one with new gamma
# ----------------------------------------------------------------------

def write_broad_with_new_gamma(
    broad_in: Path,
    broad_out: Path,
    gammas
) -> None:
    """
    Copy broad_in to broad_out, replacing gamma field when a key match exists.
    Preserves comments/whitespace minimally; rewrites numeric fields in fixed widths.
    """
    g_iter = iter(gammas)
    with broad_in.open("r") as fin, broad_out.open("w") as fout:
        for line in fin:
            try:
                g = next(g_iter)
            except StopIteration:
                raise ValueError(
                    f"Not enough gamma values: ran out while writing {broad_out}"
                )

            assert line[13] != ' '
            assert line[14] == ' '

            # Format gamma to 4 dp, fixed width 6 (F6.4), then right-align within 12 chars
            gamma6 = f"{float(g):6.4f}"      # e.g. '0.1234' with leading spaces if needed
            gamma12 = gamma6.rjust(12)       # fill slice [2:14] exactly (12 chars)

            new_line = line[:2] + gamma12 + line[14:]

            fout.write(new_line)

    # Optional: check for leftover gammas (can be an error if you expect exact match)
    try:
        next(g_iter)
        raise ValueError(f"Too many gamma values provided for {broad_in.name}")
    except StopIteration:
        pass

with open(Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "creation_of_air_diet_files" / 'molecules.pkl', 'rb') as f:
    molecules_made = pickle.load(f)
MOLECULES_MADE = {}
for key, data in molecules_made.items():
    key = strip_isotopes_to_formula(key)
    MOLECULES_MADE[key] = data

def make_new_broad_file(active, perturber, data, gammas):
    filepath = Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "Nov_2025_write_up" / "Make_new_predictions" / "broad_files"
    active = active.replace('C3', '12C3').replace('ScO', '45Sc-16O').replace('CaCl', '40Ca-35Cl')
    filename = filepath / f"{active}__{perturber}.broad"
    data['predicted_gamma'] = gammas
    first = 'm0'
    third = '     0.500'

    with open(filename, 'w') as f:
        for line in data.index:
            gamma = str(round(data.loc[line]['predicted_gamma'], 4))

            Jpp = data.loc[line]['Jpp']
            J = data.loc[line]['J']
            del_j = J - Jpp

            if int(del_j) == 1:
                m = Jpp + 1
            elif int(del_j) == -1:
                m = - Jpp
            elif int(del_j) == 0:
                if Jpp == 0:
                    m = Jpp
                else:
                    continue     # No space for Q branch in m0 format
            else:
                print('Delta J problem - not P Q or R')

            if len(gamma) != 6:
                if len(gamma) == 5:
                    gamma+='0'
                if len(gamma) == 4:
                    gamma+='00'
                if len(gamma) == 3:
                    gamma+='000'
            gamma = '     '+gamma
            if Jpp.is_integer():
                fourth = str(int(m))
                if len(fourth) == 1:
                    fourth = '      '+str(fourth)
                elif len(fourth) == 2:
                    fourth = '     '+str(fourth)
                elif len(fourth) == 3:
                    fourth = '    '+str(fourth)
                elif len(fourth) == 4:
                    fourth = '   ' + str(fourth)
                elif len(fourth) == 5:
                    fourth = '  ' + str(fourth)
                elif len(fourth) == 6:
                    fourth = ' '+str(fourth)
                elif len(fourth) == 7:
                    fourth = ''+str(fourth)
                f.write(first+' '+gamma+' '+third+' '+fourth+'\n')

            else:
                fourth = str(m)
                if len(fourth) == 3:
                    fourth = '    '+str(fourth)
                elif len(fourth) == 4:
                    fourth = '   '+str(fourth)
                elif len(fourth) == 5:
                    fourth = '  '+str(fourth)
                elif len(fourth) == 6:
                    fourth = ' '+str(fourth)
                elif len(fourth) == 7:
                    fourth = ''+str(fourth)
                f.write(first+' '+gamma+' '+third+' '+fourth+'\n')

# ----------------------------------------------------------------------
# Main prediction
# ----------------------------------------------------------------------

def main() -> None:
    model, feat_cols, meta = load_production_bundle(MODEL_DIR)
    print(f"Loaded production model. n_features={len(feat_cols)}")
    if meta is not None:
        print(f"Metadata: n_rows={meta.get('n_rows')}")

    for ACTIVE_NAME in ACTIVE_MOLECULES:
        try:
            ACTIVE = get_params(ACTIVE_NAME)
        except:
            print(f"{ACTIVE_NAME} not found in molecule_parameters.csv")
            continue

        rotor_type = get_rotor_type(ACTIVE_NAME)

        # Jpp grid from matching .broad file for this ACTIVE (fallback if missing)
        broad_path = ACTIVE_TO_BROAD.get(ACTIVE_NAME)
        if broad_path is None:
            print(ACTIVE_NAME)
            print("FILE_MISSING!")
            print("Making new broad file")
            j_vals = MOLECULES_MADE[ACTIVE_NAME][['Jpp', 'J', 'Ka_aprox', 'Kc_aprox', 'Kapp_aprox', 'Kcpp_aprox']]
        else:
            j_vals = read_qns_from_broad(broad_path, rotor_type)
            print(len(j_vals))
            if len(j_vals) == 0:
                continue
            if j_vals.size == 0:
                print(ACTIVE_NAME)
                print("No J values found")

        for PERTURBER_NAME in PERTURBER_MOLECULES:
            if PERTURBER_NAME == 'self':
                PERT = get_params(ACTIVE_NAME)
            else:
                PERT = get_params(PERTURBER_NAME)
            X_raw = build_features(ACTIVE, PERT, j_vals, temperature=T_K)

            print(ACTIVE_NAME)
            print(PERTURBER_NAME)

            # Select/reorder to match training contract
            X = align_features(X_raw, feat_cols)

            # Predict: with the new production wrapper, predict() already returns gamma
            gamma = model.predict(X)

            if broad_path is not None:
                out_name = broad_path.name.replace('air', PERTURBER_NAME)
                out_path = Path.home() / "Desktop" / "line_broadening.nosync" / "line_broadening" / "Nov_2025_write_up" / "Make_new_predictions" / "broad_files" / out_name
                write_broad_with_new_gamma(broad_path, out_path, gamma)
            else:
                make_new_broad_file(ACTIVE_NAME, PERTURBER_NAME, X, gamma)


if __name__ == "__main__":
    main()
