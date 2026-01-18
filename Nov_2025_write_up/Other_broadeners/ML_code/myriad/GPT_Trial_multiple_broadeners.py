# === Helpers and constants ===
# === Imports ===
from __future__ import annotations

import re
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Iterable, Literal, List
import itertools
import collections

import numpy as np
import pandas as pd

# Machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
import time

start = time.time()
# === Environment setup ===

# Toggle this manually or via an environment variable
local = True  # set False when running on cluster

VOIGT_ONLY = True  # set False to keep all profiles
USE_T = False
USE_ELECTRONIC = False
N_CROSS_VAL = 20
MAX_FOLDS = None

# --- Uncertainty normalisation options ---

# 1) Apply a minimum fractional uncertainty for LLM-scraped data
APPLY_LLM_MIN_FRAC_UNCERT = True
LLM_MIN_FRAC_UNCERT = 0.10     # e.g. at least 10% on γ for LLM data

# 2) Apply a minimum fractional uncertainty for all NON-HITRAN data
#    (this can be the same or different from LLM)
APPLY_NONHITRAN_MIN_FRAC_UNCERT = False
NON_HITRAN_MIN_FRAC_UNCERT = 0.50   # e.g. at least 10% on γ for any non-HITRAN data

# 3) Use a flat fractional uncertainty for ALL data (overrides 1 and 2)
APPLY_FLAT_FRACTIONAL_UNCERT = False
FLAT_FRACTIONAL_UNCERT = 0.50       # e.g. 50% relative uncertainty everywhere

# Drop self-broadening data (active species == perturber)
DROP_SELF_BROADENING = False   # set to True to remove self data
COMPARE_SELF_BROADENING = False

CHECK_PATHS = False

# Base data path
if local:
    HOME = Path.home() / "Desktop" / "line_broadening.nosync"
else:
    HOME = Path.home()

# set a single tag so training and plotting share the same folder
date_tag = datetime.now().strftime("%Y-%m-%d")
out = HOME / "Scratch" / f"other_broadeners_{date_tag}"
out.mkdir(parents=True, exist_ok=True)

# Small utility for conditional loading
def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV unless it is too large for local use.
    On local runs, skip any file >10 MB.
    """
    max_size_mb = 10

    # Ensure path is a Path object (just in case string was passed)
    if isinstance(path, str):
        path = Path(path)

    # Check 'local' variable (assuming it is defined globally in your script)
    if local and path.exists() and path.stat().st_size > max_size_mb * 1024 * 1024:
        print(f"Skipping large file on local mode: {path.name}")
        skip_kwargs = kwargs.copy()
        skip_kwargs['nrows'] = 0
        skip_kwargs.pop('chunksize', None)  # Ensure we don't get an iterator back

        return pd.read_csv(path, **skip_kwargs)
    return pd.read_csv(path, **kwargs)

def broadening(m: int, T: float, ma: float, mp: float, b0_pm: float) -> float:
    """
    Collisional HWHM γ for (vib)rotational lines from the simple semiclassical model.

    Formula (Eq. 6):
        γ = 1.7796e-5 * [ m / (m - 2) ] * T^(-1/2) * sqrt( (ma + mp) / (ma * mp) ) * b0^2

    Parameters
    ----------
    m : int
        Exponent of the leading anisotropic interaction’s S2(b) ∝ b^(-m).
        Typical values: 4 (dipole–dipole), 6 (dipole–quadrupole), 8 (quadrupole–quadrupole),
        10 (dipole–induced dipole).
    T : float
        Temperature in Kelvin.
    ma : float
        Molecular mass (Da) of the active species.
    mp : float
        Molecular mass (Da) of the perturber.
    b0_pm : float
        Cutoff impact parameter b0 in picometres. Often taken ≈ kinetic diameter d.

    Returns
    -------
    float
        γ in cm⁻¹ atm⁻¹.

    Notes
    -----
    - Uses number density at 1 atm and mean relative speed from kinetic theory as in the paper.
    - T-scaling is T^(-1/2).
    - See: Buldyreva, Yurchenko & Tennyson, RASTI 1, 43 (2022). Eq. (6). :contentReference[oaicite:1]{index=1}
    """
    if m == 2:
        raise ValueError("m cannot be equal to 2.")
    return 1.7796e-5 * (m / (m - 2.0)) * (1.0 / np.sqrt(T)) * np.sqrt((ma + mp) / (ma * mp)) * (b0_pm ** 2)

_HITRAN_ERR_TO_FRAC = {
    0: 1.00,  # unreported/unavailable
    1: 0.80,  # default/constant
    2: 0.60,  # average/estimate
    3: 0.50,  # >=20%
    4: 0.20,  # 10–20%
    5: 0.10,  # 5–10%
    6: 0.05,  # 2–5%
    7: 0.02,  # 1–2%
    8: 0.01,  # <=1%
}

def convert_hitran_error_to_uncertainty(err_col: pd.Series,
                                        gamma_col: pd.Series) -> pd.Series:
    """
    Map HITRAN error codes to *absolute* uncertainties on gamma.

    err_col  : HITRAN error codes (ints/strings)
    gamma_col: gamma values (cm^-1 atm^-1)

    Returns absolute uncertainty (same units as gamma).
    """
    codes = pd.to_numeric(err_col, errors="coerce").astype("Int64")
    frac = codes.map(_HITRAN_ERR_TO_FRAC).astype(float)        # fractional
    gamma = pd.to_numeric(gamma_col, errors="coerce").astype(float)
    abs_unc = frac * gamma
    # sanitize
    abs_unc = abs_unc.replace([np.inf, -np.inf], np.nan)
    abs_unc[gamma <= 0] = np.nan
    return abs_unc

def unify_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise vibrational/electronic state columns from HITRAN into 'state_up' and 'state_low'.

    Handles various schema patterns:
      - ElecStateLabel, v, ElecStateLabelpp, vpp
      - ElecStateLabel, v1..v12, ElecStateLabelpp, v1pp..v12pp
      - ElecStateLabel, Ka, Kc, ElecStateLabelpp, Kapp, Kcpp
    """
    df = df.copy()

    # --- upper state ---
    up_parts = []
    if "ElecStateLabel" in df:
        up_parts.append(df["ElecStateLabel"].astype(str))
    vib_cols = [c for c in df.columns if re.fullmatch(r"v\d+", c)]
    if "v" in df:
        up_parts.append(df["v"].astype(str))
    elif vib_cols:
        up_parts.append(df[vib_cols].astype(str).agg("-".join, axis=1))

    # --- lower state ---
    low_parts = []
    if "ElecStateLabelpp" in df:
        low_parts.append(df["ElecStateLabelpp"].astype(str))
    vibpp_cols = [c for c in df.columns if re.fullmatch(r"v\d+pp", c)]
    if "vpp" in df:
        low_parts.append(df["vpp"].astype(str))
    elif vibpp_cols:
        low_parts.append(df[vibpp_cols].astype(str).agg("-".join, axis=1))

    # combine parts into canonical identifiers
    if up_parts:
        df["state_up"] = up_parts[0] if len(up_parts) == 1 else up_parts[0] + "_" + up_parts[1]
    if low_parts:
        df["state_low"] = low_parts[0] if len(low_parts) == 1 else low_parts[0] + "_" + low_parts[1]

    return df

def _keep_ground_state(df: pd.DataFrame,
                       up_col: str = "ElecStateLabel",
                       low_col: str = "ElecStateLabelpp") -> tuple[pd.DataFrame, dict]:
    """
    Keep rows where electronic labels (if present) start with 'X' (case-insensitive).
    If a label column is missing or NaN, it is not filtered on.
    Returns (filtered_df, stats).
    """
    df = df.copy()
    def ok(col):
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        s = df[col].astype(str).str.strip()
        m = s.str.len().gt(0)  # non-empty
        # allow 'X', 'X1Sigma+', 'X^2Π', etc. → startswith 'X' or 'x'
        keep = s.str[0].str.upper().eq('X')
        return (~m) | keep  # keep rows with missing/empty OR starting with X

    m_up  = ok(up_col)
    m_low = ok(low_col)
    mask  = m_up & m_low
    stats = {
        "n_before": int(len(df)),
        "n_after":  int(mask.sum()),
        "dropped":  int((~mask).sum()),
    }
    return df[mask].reset_index(drop=True), stats

def parse_active_perturber_from_path(path: Path) -> Tuple[str, str] | None:
    """
    Infer active molecule and broadener from a filename.

    Accepts common patterns like:
      H2O_CO2_*.csv,  H2O__CO2.csv,  H2O-by-CO2.csv

    Returns None if no match.
    """
    stem = path.stem
    # try several patterns
    patterns = [
        r"^([A-Za-z0-9]+)[\-_]{1,2}([A-Za-z0-9]+)$",
        r"^([A-Za-z0-9]+)[\-_]{1,2}([A-Za-z0-9]+)[\-_].*$",
        r"^([A-Za-z0-9]+)[-_ ]by[-_ ]([A-Za-z0-9]+).*$",
    ]
    for pat in patterns:
        m = re.match(pat, stem)
        if m:
            return m.group(1), m.group(2)
    return None

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

# === File discovery and primary loaders ===

# 1) New data files
files_new = list((HOME / "line_broadening" / "Other_broadeners" / "Files_with_new_data").glob("*.csv"))

db: Dict[str, pd.DataFrame] = {}

for f in files_new:
    key_pair = parse_active_perturber_from_path(f)
    if not key_pair:
        # skip filenames like "Broadening_data_from_smaller_sources.csv"
        continue
    i, j = key_pair
    if i == "Broadening" and j == "data":  # guard against the summary file
        continue

    cols = ['J', 'Ka_aprox', 'Kc_aprox', 'Jpp', 'Kapp_aprox', 'Kcpp_aprox', 'T', 'profile', 'gamma', 'gamma_uncertainty']
    try:
        file = read_csv_safe(f, low_memory=False, dtype={'J': np.float64, 'Jpp': np.float64}, usecols=cols)
    except Exception:
        # if columns differ slightly, read permissively then subset
        file = read_csv_safe(f, low_memory=False)
        missing = [c for c in cols if c not in file.columns]
        if missing:
            # cannot use this file reliably
            continue
        file = file[cols]

    # Get rid of zeros!
    file = file[file["gamma"] > 0]

    if VOIGT_ONLY and "profile" in file:
        file = file[file["profile"].astype(str).str.lower().eq("voigt")]

    if not USE_T and "T" in file.columns:
        file = file[np.abs(file["T"] - 296) <= 10]

    if file.empty:
        continue

    # Tag source: literature CSV
    file["source"] = "CSV_NEW"

    key = f"{i}_{j}"
    db[key] = pd.concat([db[key], file], ignore_index=True) if key in db else file.reset_index(drop=True)

def read_exomol_broad_file(path: Path) -> pd.DataFrame:
    """
    Parse an ExoMol diet .broad file into a DataFrame with columns:
    ['J', 'Ka_aprox', 'Kc_aprox', 'Jpp', 'Kapp_aprox', 'Kcpp_aprox',
     'T', 'profile', 'gamma', 'gamma_uncertainty', 'recipe', 'n']
    """
    print(f"Reading ExoMol .broad file: {path}")
    try:
        raw = pd.read_csv(
            path,
            delim_whitespace=True,
            comment="#",
            header=None,
            dtype=str,   # read as str first; we’ll cast as needed
        )
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return raw

    # Require at least recipe, gamma, n
    raw = raw.dropna(how="all")
    if raw.shape[1] < 3:
        return pd.DataFrame()

    rows = []
    for _, row in raw.iterrows():
        # ignore blank / malformed lines
        if pd.isna(row.iloc[0]):
            continue

        code = str(row.iloc[0]).strip()
        try:
            gamma = float(row.iloc[1])
        except Exception:
            continue
        try:
            n_val = float(row.iloc[2])
        except Exception:
            n_val = np.nan

        qns = [r for r in row.iloc[3:].tolist() if not pd.isna(r)]

        rec = {
            "recipe": code,
            "gamma": gamma,
            "n": n_val,
            "T": 296.0,
            "profile": "Voigt",
            "gamma_uncertainty": gamma / 2,
            "J": np.nan,
            "Ka_aprox": np.nan,
            "Kc_aprox": np.nan,
            "Jpp": np.nan,
            "Kapp_aprox": np.nan,
            "Kcpp_aprox": np.nan,
        }

        # Map diet codes to quantum numbers
        # You can refine this mapping if your files have extra info.
        if code.startswith("a0"):
            # a0: J''
            if len(qns) >= 1:
                rec["Jpp"] = float(qns[0])
                rec["J"] = rec["Jpp"] + 1
        elif code.startswith("a1"):
            # a1: J'', K''
            if len(qns) >= 1:
                rec["Jpp"] = float(qns[0])
                rec["J"] = rec["Jpp"] + 1
            if len(qns) >= 2:
                rec["Kcpp_aprox"] = float(qns[1])
                rec["Kc_aprox"] = rec["Kcpp_aprox"] + 1
        elif code.startswith("m0"):
            # m0: |m|
            if len(qns) >= 1:
                # reuse M in J column if you later compute M-specific features
                rec["Jpp"] = float(qns[0])
                rec["J"] = rec["Jpp"] + 1
        elif code.startswith("m1"):
            # m1: |m|, K''
            if len(qns) >= 1:
                rec["Jpp"] = float(qns[0])
                rec["J"] = rec["Jpp"] + 1
            if len(qns) >= 2:
                rec["Kcpp_aprox"] = float(qns[1])
                rec["Kc_aprox"] = rec["Kcpp_aprox"] + 1
        elif code.startswith("a5"):
            # a5: J', K'a, K'c, J'', K''a, K''c
            if len(qns) >= 1:
                rec["J"] = float(qns[0])
            if len(qns) >= 2:
                rec["Ka_aprox"] = float(qns[1])
            if len(qns) >= 3:
                rec["Kc_aprox"] = float(qns[2])
            if len(qns) >= 4:
                rec["Jpp"] = float(qns[3])
            if len(qns) >= 5:
                rec["Kapp_aprox"] = float(qns[4])
            if len(qns) >= 6:
                rec["Kcpp_aprox"] = float(qns[5])

        rows.append(rec)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop(columns=["recipe"])

    # Cast numeric columns
    for col in ["J", "Ka_aprox", "Kc_aprox", "Jpp", "Kapp_aprox", "Kcpp_aprox",
                "T", "gamma", "gamma_uncertainty", "n"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def strip_isotopes(spec: str) -> str:
    """
    Convert something like '12C2-1H2' -> 'C2H2'
    by removing leading isotope numbers from each atomic block.
    """
    parts = spec.split('-')  # e.g. ['12C2', '1H2']
    cleaned = []

    for part in parts:
        # Match: optional digits, then element symbol, then optional digits
        m = re.match(r'(\d*)([A-Z][a-z]?)(\d*)', part)
        if not m:
            continue
        _, elem, count = m.groups()
        cleaned.append(elem + count)

    return ''.join(cleaned)

def parse_broad_filename(filename: str):
    """
    '12C2-1H2__CO2.broad' -> ('C2H2', 'CO2')
    """
    stem = Path(filename).stem          # '12C2-1H2__CO2'
    active_raw, perturber_raw = stem.split('__', 1)
    active = strip_isotopes(active_raw) # 'C2H2'
    perturber = perturber_raw  # 'CO2' (unchanged)
    return (active, perturber)

exomol_files = list((HOME / "line_broadening" / "ExoMol_broadening_data").glob("*/*.broad"))

for f in exomol_files:
    key_pair = parse_broad_filename(f)
    if not key_pair:
        continue
    i, j = key_pair

    if j == "self":
        j = i

    file = read_exomol_broad_file(f)

    # Get rid of zeros / non-positive values
    file = file[file["gamma"] > 0]

    if file.empty:
        continue

    # VOIGT_ONLY is naturally satisfied (we set profile="Voigt"),
    # but keep the guard if you want.
    if VOIGT_ONLY and "profile" in file:
        file = file[file["profile"].astype(str).str.lower().eq("voigt")]

    if not USE_T and "T" in file.columns:
        file = file[np.abs(file["T"] - 296) <= 10]

    if file.empty:
        continue

    if i in ['AlH', 'C2H2']:
        file['Kc_aprox'] = file['J']
        file['Kcpp_aprox'] = file['Jpp']
        file['Ka_aprox'] = 0
        file['Kapp_aprox'] = 0
    elif i == 'H2O':
        file['Ka_aprox'] = file['J'] - file['Kc_aprox']
        file['Kapp_aprox'] = file['Jpp'] - file['Kcpp_aprox']
    elif i == 'CH4':
        file['Ka_aprox'] = file['J'] / 2
        file['Kc_aprox'] = file['J'] / 2
        file['Kapp_aprox'] = file['Jpp'] / 2
        file['Kcpp_aprox'] = file['Jpp'] / 2

    key = f"{i}_{j}"
    db[key] = pd.concat([db[key], file], ignore_index=True) if key in db else file.reset_index(drop=True)

t1 = time.time()

# 2) HITRAN raw data

files_hit = [
    p for p in (HOME / "line_broadening" / "hitran_data").rglob("*")
    if p.is_file()
    and p.name == "1_iso.csv"          # exact match, not CH3F_1_iso.csv
    and "readme" not in p.name.lower()
]

hit_db: Dict[str, pd.DataFrame] = {}

for f in files_hit:
    i = f.parent.name
    i = i.replace("_par", "")

    # Define critical columns to read explicitly
    # We include 'ElecStateLabel'/'statep' to ensure _keep_ground_state works
    needed_qns = {'J', 'Jpp', 'Ka_aprox', 'Kc_aprox', 'Kapp_aprox', 'Kcpp_aprox'}

    data_i = read_csv_safe(
        f,
        low_memory=False,
        dtype={'J': np.float64, 'Jpp': np.float64},
        # Lambda function: Keep if it's a QN/State col OR if it starts with "gamma_" (captures values and errors)
        usecols=lambda c: c in needed_qns or c.startswith("gamma_")
    )

    if not USE_ELECTRONIC:
        data_i, st = _keep_ground_state(data_i)
        if st["dropped"] > 0:
            print(f"[electronic] {i}: kept ground state only → {st['n_before']} → {st['n_after']} "
                  f"(dropped {st['dropped']})")

    # gamma_* columns excluding *err and *ref
    gamma_cols = [c for c in data_i.columns if c.startswith("gamma_") and "err" not in c and "ref" not in c]

    for gamma_col in gamma_cols:
        err_col = f"{gamma_col}-err"
        cols = ['J', 'Ka_aprox', 'Kc_aprox', 'Jpp', 'Kapp_aprox', 'Kcpp_aprox']

        # We must check if columns exist in the filtered dataframe before selecting
        available_cols = [c for c in cols if c in data_i.columns]

        take = data_i[available_cols + [gamma_col] + ([err_col] if err_col in data_i.columns else [])].copy()

        # coerce numeric and drop non-numeric rows
        take[gamma_col] = pd.to_numeric(take[gamma_col], errors="coerce")
        if err_col in take.columns:
            take[err_col] = convert_hitran_error_to_uncertainty(take[err_col], take[gamma_col])
        else:
            take[err_col] = np.nan

        take = take.dropna(subset=[gamma_col])
        if take.empty:
            continue

        j = gamma_col.replace("gamma_", "")
        if j == "self":
            j = i

        take['molecule'] = i
        take['broadener'] = j
        take['T'] = 296
        take['profile'] = 'Voigt'
        take = take.rename(columns={gamma_col: 'gamma', err_col: 'gamma_uncertainty'})
        take['gamma'] = take['gamma'].astype(float)

        # Tag source: HITRAN
        take['source'] = 'HITRAN'

        key = f"{i}_{j}"
        db[key] = pd.concat([db[key], take], ignore_index=True) if key in db else take.reset_index(drop=True)

t2 = time.time()

# === Part 3: Small sources + LLM merge ===

def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_small_sources(
    base_dir: Path = HOME / "line_broadening" / "Other_broadeners" / "Files_with_new_data",
    llm_path: Path = HOME / "line_broadening" / "Other_broadeners" / "Files_with_new_data" / "LLM_data" / "LLM_scraped_data.csv",
    guest_llm_path: Path = HOME / "line_broadening" / "Other_broadeners" / "Files_with_new_data" / "LLM_data" / "Guest_LLM_scraped_data.csv",
    voigt_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load curated 'small sources' and LLM-scraped datasets and return a dict keyed by 'active_broadener'.

    Required columns:
      molecule, broadener, J, Jpp, Ka_aprox, Kapp_aprox, Kc_aprox, Kcpp_aprox, gamma, gamma_uncertainty, n, T, profile, paper
    """
    cols = ['molecule', 'broadener', 'J', 'Jpp', 'Ka_aprox', 'Kapp_aprox', 'Kc_aprox', 'Kcpp_aprox',
            'gamma', 'gamma_uncertainty', 'n', 'T', 'profile', 'paper']

    # 1) curated small sources
    small_path = base_dir / "Broadening_data_from_smaller_sources.csv"
    small = read_csv_safe(
        small_path,
        dtype={'J': np.float64, 'Jpp': np.float64, 'gamma': np.float64, 'gamma_uncertainty': np.float64},
        usecols=cols
    )
    # Tag source
    if not small.empty:
        small["source"] = "SMALL_LIT"

    # 2) LLM-scraped
    if llm_path.exists():
        masters = read_csv_safe(llm_path, index_col=0)
        masters = masters[cols]

        if not masters.empty:
            masters["source"] = "LLM"

            if APPLY_LLM_MIN_FRAC_UNCERT:
                frac = masters["gamma_uncertainty"] / masters["gamma"]
                bad = frac < LLM_MIN_FRAC_UNCERT
                masters.loc[bad, "gamma_uncertainty"] = (
                        LLM_MIN_FRAC_UNCERT * masters.loc[bad, "gamma"]
                )

        data = pd.concat([small, masters], ignore_index=True)
    else:
        data = small

    # 3) LLM-scraped 2
    if guest_llm_path.exists():
        masters = read_csv_safe(guest_llm_path, index_col=0)
        masters = masters[cols]

        if not masters.empty:
            masters["source"] = "LLM_GUEST"

            if APPLY_LLM_MIN_FRAC_UNCERT:
                frac = masters["gamma_uncertainty"] / masters["gamma"]
                bad = frac < LLM_MIN_FRAC_UNCERT
                masters.loc[bad, "gamma_uncertainty"] = (
                        LLM_MIN_FRAC_UNCERT * masters.loc[bad, "gamma"]
                )

        data = pd.concat([data, masters], ignore_index=True)

    # Basic cleaning
    num_cols = ['J','Jpp','Ka_aprox','Kapp_aprox','Kc_aprox','Kcpp_aprox','gamma','gamma_uncertainty','n','T']
    data = _coerce_numeric(data, num_cols)
    data = data.dropna(subset=['molecule','broadener','J','Jpp','gamma','gamma_uncertainty'])
    if voigt_only:
        data = data[data['profile'] == 'Voigt']
    if not USE_T and "T" in data.columns:
        data = data[np.abs(data["T"] - 296) <= 10]

    # Build dict: active_broadener -> DataFrame
    out: Dict[str, pd.DataFrame] = {}
    for (act, pert), df in data.groupby(['molecule','broadener'], sort=False):
        key = f"{act}_{pert}"
        out[key] = df.reset_index(drop=True)

    return out


# Load and merge into existing db
small_dict = load_small_sources()
for key, df in small_dict.items():
    if df.empty:
        continue
    # unify schema with main db expectation
    df = df[['J', 'Ka_aprox', 'Kc_aprox', 'Jpp', 'Kapp_aprox', 'Kcpp_aprox', 'T', 'profile',
             'gamma', 'gamma_uncertainty', 'molecule', 'broadener', 'n', 'paper', 'source']]
    db[key] = pd.concat([db[key], df], ignore_index=True) if key in db else df.reset_index(drop=True)

t3 = time.time()

# === Part 3.5: Gamache data ===

gamache_data = HOME / "line_broadening" / "Gamache stuff" / "Processed_files_for_ML" / "combined_broadening_data.csv"

if gamache_data.exists():
    cols = ['J', 'Ka_aprox', 'Kc_aprox', 'Jpp', 'Kapp_aprox', 'Kcpp_aprox', 'T', 'profile', 'gamma', 'gamma_uncertainty', 'molecule', 'broadener']
    file = read_csv_safe(gamache_data, low_memory=False, dtype={'J': np.float64, 'Jpp': np.float64}, usecols=cols)

    # Get rid of zeros!
    file = file[file["gamma"] > 0]

    if VOIGT_ONLY and "profile" in file:
        file = file[file["profile"].astype(str).str.lower().eq("voigt")]

    if not USE_T and "T" in file.columns:
        file = file[np.abs(file["T"] - 296) <= 10]

    # Tag source: literature CSV
    file["source"] = "Gamache_data"

    for (molecule, perturber), group_df in file.groupby(['molecule', 'broadener']):
        key = f"{molecule}_{perturber}"

        # Save to db
        if key in db:
            db[key] = pd.concat([db[key], group_df], ignore_index=True)
        else:
            db[key] = group_df.reset_index(drop=True)

if CHECK_PATHS:
    output_path = HOME / "line_broadening" / "Nov_2025_write_up" / "Find_files" / "sources.txt"
    with open(output_path, "w") as file:
        for p in files_new:
            file.write(f"{p}\n")

        for p in exomol_files:
            file.write(f"{p}\n")

        for p in files_hit:
            file.write(f"{p}\n")

        file.write(f'{HOME / "line_broadening" / "Other_broadeners" / "Files_with_new_data" / "Broadening_data_from_smaller_sources.csv"}\n')
        file.write(f'{HOME / "line_broadening" / "Other_broadeners" / "Files_with_new_data" / "LLM_data" / "LLM_scraped_data.csv"}\n')
        file.write(f'{HOME / "line_broadening" / "Other_broadeners" / "Files_with_new_data" / "LLM_data" / "Guest_LLM_scraped_data.csv"}\n')
        file.write(f'{gamache_data}\n')


# === Part 4: Feature engineering and QC ===

def attach_molecule_params(df: pd.DataFrame, active: str, pert: str, params: pd.DataFrame) -> pd.DataFrame:
    """
    Add active_* and perturber_* descriptors from molecule_parameters.
    Expects params indexed by molecule name with columns like weight, dipole, m, d, polar, B0a, B0b, B0c.
    """
    if active not in params.index or pert not in params.index:
        return pd.DataFrame()  # skip if metadata missing

    act = params.loc[active]
    per = params.loc[pert]

    for idx, val in act.items():
        df[f"active_{idx}"] = val
    for idx, val in per.items():
        df[f"perturber_{idx}"] = val

    # derived combos
    df["m"] = df["active_m"] + df["perturber_m"]
    df["d_act_per"] = 0.5 * df["active_d"] + 0.5 * df["perturber_d"]
    return df

def compare_M_symmetry(df):
    if "M" not in df or "gamma" not in df:
        return None

    # separate positive and negative
    pos = df[df["M"] > 0].copy()
    neg = df[df["M"] < 0].copy()

    # pair by absolute M
    pos["absM"] = pos["M"].abs()
    neg["absM"] = neg["M"].abs()

    g_pos = pos.groupby("absM")["gamma"].mean()
    g_neg = neg.groupby("absM")["gamma"].mean()

    # intersect and compare
    common = g_pos.index.intersection(g_neg.index)
    ratios = g_pos.loc[common] / g_neg.loc[common]

    return ratios

def compute_branch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rotational-branch indicators and M index.
    Uses Jpp - J to detect P/Q/R/O/S as in your original code.
    """
    branch = df["Jpp"] - df["J"]
    # initialize zeros
    P = np.zeros(len(df))
    Q = np.zeros(len(df))
    R = np.zeros(len(df))
    O = np.zeros(len(df))
    S = np.zeros(len(df))

    # fill per rule
    P_idx = branch.eq(1)
    Q_idx = branch.eq(0)
    R_idx = branch.eq(-1)
    O_idx = branch.eq(2)
    S_idx = branch.eq(-2)

    P[P_idx] = -df.loc[P_idx, "Jpp"]
    Q[Q_idx] = df.loc[Q_idx, "Jpp"]
    R[R_idx] = df.loc[R_idx, "Jpp"] + 1.0
    O[O_idx] = -df.loc[O_idx, "Jpp"]
    S[S_idx] = df.loc[S_idx, "Jpp"] + 1.0

    df = df.copy()
    df["P"], df["Q"], df["R"], df["O"], df["S"] = P, Q, R, O, S
    df["M"] = df[["P", "Q", "R", "O", "S"]].sum(axis=1)
    compare_M_symmetry(df)
    return df.drop(columns=["P", "Q", "R", "O", "S"])

def _agg(g):
    # ensure numeric
    g = g.copy()
    g['gamma'] = g['gamma'].astype(float)
    g['gamma_uncertainty'] = g['gamma_uncertainty'].astype(float)

    # collapse exact-duplicate gamma values:
    # for each unique gamma keep the row with the smallest uncertainty
    g_unique = (
        g.sort_values('gamma_uncertainty')
         .drop_duplicates(subset='gamma', keep='first')
    )

    sig = g_unique['gamma_uncertainty'].clip(lower=1e-9)
    w = 1.0 / (sig**2)

    mu = np.average(g_unique['gamma'], weights=w)
    stat_err = np.sqrt(1.0 / w.sum())  # standard error on weighted mean

    if len(g_unique) > 1:
        scatter_err = np.std(g_unique['gamma'], ddof=1)
    else:
        scatter_err = 0.0
    min_input_uncertainty = g_unique['gamma_uncertainty'].min()
    final_uncertainty = max(stat_err, scatter_err)
    # Optional: Floor it so it never drops below 50% of the best single measurement
    final_uncertainty = max(final_uncertainty, 0.5 * min_input_uncertainty)

    return pd.Series({
        'gamma': mu,
        'gamma_uncertainty': final_uncertainty,
        # 'n_meas': len(g),              # total raw rows in group
        # 'n_unique_gamma': len(g_unique),
    })

def clean_and_engineer(
    db: Dict[str, pd.DataFrame],
    molecule_parameters: pd.DataFrame,
    skip_actives: set[str] = {},
    skip_perturbers: set[str] = {},
) -> Dict[str, pd.DataFrame]:
    """
    Convert raw db[key] tables into engineered feature tables per active_broadener key.
    Steps:
      - drop unwanted species
      - filter to iso=1 if local_iso_id present
      - attach molecule parameters
      - select feature/target columns
      - compute fractional_error and filters
      - remove implausible branches |Jpp-J|>2
      - compute branch features and M
      - add 'broadness_jeanna' constant per key
    """
    out: Dict[str, pd.DataFrame] = {}

    pre_len_sum = 0
    post_len_sum = 0

    for key, data in db.items():
        active, pert = key.split("_", 1)

        # flag self-broadening
        is_self = float(active == pert)

        if active in skip_actives or pert in skip_perturbers:
            continue

        df = data.copy()

        if active == 'HF' and pert == 'CO2':
            # remove unphysically high gamma values above ~0.2 cm^-1/atm
            df = df[df['gamma'] < 0.2]
        if active == 'HCN' and pert == 'HCN':
            # remove unphysically large gamma values at high J
            df = df[(df['gamma'] < 0.3) | (df['J'] <= 50)]

        # optional isotopologue filter
        if "local_iso_id" in df.columns:
            df = df[df["local_iso_id"] == 1]

        # skip if parameters missing
        if active not in molecule_parameters.index or pert not in molecule_parameters.index:
            print(f"Missing params for {active} or {pert}; skipping {key}")
            continue

        # attach parameters and keep schema
        df = attach_molecule_params(df, active, pert, molecule_parameters)
        if df.empty:
            continue

        # --- Uncertainty normalisation & caps ---

        if APPLY_FLAT_FRACTIONAL_UNCERT:
            # 3) Global flat scheme: ignore all input uncertainties
            df["gamma_uncertainty"] = FLAT_FRACTIONAL_UNCERT * df["gamma"]

        else:
            if "source" in df.columns:
                src = df["source"].astype(str)

                # 1) LLM minimum fractional uncertainty
                if APPLY_LLM_MIN_FRAC_UNCERT:
                    llm_mask = src.str.contains("LLM", case=False, na=False)
                    if llm_mask.any():
                        frac = df.loc[llm_mask, "gamma_uncertainty"] / df.loc[llm_mask, "gamma"]
                        too_low = frac < LLM_MIN_FRAC_UNCERT
                        if too_low.any():
                            df.loc[llm_mask & too_low, "gamma_uncertainty"] = (
                                    LLM_MIN_FRAC_UNCERT * df.loc[llm_mask & too_low, "gamma"]
                            )

                # 2) Non-HITRAN minimum fractional uncertainty (no max cap)
                if APPLY_NONHITRAN_MIN_FRAC_UNCERT:
                    hitran_mask = src.str.upper().eq("HITRAN")
                    non_hitran = ~hitran_mask
                    if non_hitran.any():
                        frac = df.loc[non_hitran, "gamma_uncertainty"] / df.loc[non_hitran, "gamma"]
                        too_low = frac < NON_HITRAN_MIN_FRAC_UNCERT
                        if too_low.any():
                            df.loc[non_hitran & too_low, "gamma_uncertainty"] = (
                                    NON_HITRAN_MIN_FRAC_UNCERT * df.loc[non_hitran & too_low, "gamma"]
                            )

        cols = [
            "J", "Ka_aprox", "Kc_aprox", "Jpp", "Kapp_aprox", "Kcpp_aprox", "T", "profile",
            "gamma", "gamma_uncertainty", "active_weight", "active_dipole", "active_m",
            "active_d", "active_polar", "active_B0a", "active_B0b", "active_B0c",
            "perturber_weight", "perturber_dipole", "perturber_m", "perturber_d",
            "perturber_polar", "perturber_B0a", "perturber_B0b", "perturber_B0c", "m",
            "d_act_per", "active_quadrupole_xx", "active_quadrupole_yy", "active_quadrupole_zz",
            "perturber_quadrupole_xx", "perturber_quadrupole_yy", "perturber_quadrupole_zz"
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].copy()

        df["is_self"] = is_self
        # Optional removal of self-broadening rows
        # if DROP_SELF_BROADENING:
        #     df = df[df["is_self"] == 0]
        #     if df.empty:
        #         continue

                # errors and basic filters
        df["fractional_error"] = df["gamma_uncertainty"] / df["gamma"]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["gamma", "gamma_uncertainty", "fractional_error"])
        df = df[(df["gamma"] != 0) & (df["gamma_uncertainty"] != 0)]

        # remove lines with |Jpp - J| > 2
        branch = df["Jpp"] - df["J"]
        df = df[branch.abs() <= 2].copy()

        # branch-derived features
        df = compute_branch_features(df).reset_index(drop=True)

        # constant per key: Jeanna broadening estimate at first row values
        try:
            broadness_jeanna = broadening(
                df["m"].iloc[0], df["T"].iloc[0],
                df["active_weight"].iloc[0], df["perturber_weight"].iloc[0],
                df["d_act_per"].iloc[0]
            )
            df["broadness_jeanna"] = broadness_jeanna
        except Exception:
            df["broadness_jeanna"] = np.nan

        # --- Aggregation to reduce duplicates ---
        pre_len = len(df)

        # measure spread before aggregation
        if pre_len > 1:
            gamma_spread = (
                df.groupby(['J', 'Jpp', 'Kc_aprox', 'Kcpp_aprox'])['gamma']
                .agg(['count', 'std', 'min', 'max'])
                .dropna()
            )

        group_cols = ['J', 'Jpp', 'Kc_aprox', 'Kcpp_aprox', 'M']
        target_cols = ['gamma', 'gamma_uncertainty']

        pre_len = len(df)
        aggregated = (df
              .groupby(group_cols, as_index=False, sort=True)
              .apply(_agg)
              .reset_index(drop=True))
        meta_cols = [c for c in df.columns if c not in set(target_cols)]
        meta = (df.groupby(group_cols, as_index=False)[meta_cols].agg(lambda s: s.iloc[0]))
        df = aggregated.merge(meta, on=group_cols, how='left')
        post_len = len(df)
        df['fractional_error'] = df['gamma_uncertainty'] / df['gamma']

        pre_len_sum += pre_len
        post_len_sum += post_len

        # --- NaN diagnostics ---
        nan_counts = df.isna().sum()
        total_nans = int(nan_counts.sum())
        if total_nans > 0:
            print(f"[warn] {key}: {total_nans} NaN values remain after cleaning")
            print(nan_counts[nan_counts > 0].sort_values(ascending=False).to_string())
            df = df.dropna().reset_index(drop=True)
            print(f"[info] {key}: dropped {total_nans} NaN cells → {len(df)} rows remain")

        df['active_rms_quadrupole'] = np.sqrt(
                (
                    df["active_quadrupole_xx"]**2 +
                    df["active_quadrupole_yy"]**2 +
                    df["active_quadrupole_zz"]**2
                ) / 3
            )
        df['perturber_rms_quadrupole'] = np.sqrt(
                (
                    df["perturber_quadrupole_xx"]**2 +
                    df["perturber_quadrupole_yy"]**2 +
                    df["perturber_quadrupole_zz"]**2
                ) / 3
            )
        # df = df.drop(columns=['active_quadrupole_xx', 'active_quadrupole_yy',
        #                      'active_quadrupole_zz', 'perturber_quadrupole_xx',
        #                      'perturber_quadrupole_yy', 'perturber_quadrupole_zz'])

        out[key] = df

    print(f"SUM! Pre_len: {pre_len_sum:,} Post_len {post_len_sum:,} rows")
    return out

# === Load molecule parameters ===
molecule_parameters_path = HOME / "line_broadening" / "molecule_parameters.csv"
molecule_parameters = read_csv_safe(molecule_parameters_path, index_col=0)

# required descriptors used downstream
REQUIRED_COLS = {
    "weight", "dipole", "m", "d", "polar", "B0a", "B0b", "B0c", "quadrupole_xx", "quadrupole_yy", "quadrupole_zz"
}

def validate_params(params: pd.DataFrame) -> None:
    missing_cols = REQUIRED_COLS - set(params.columns)
    if missing_cols:
        raise ValueError(f"molecule_parameters missing columns: {sorted(missing_cols)}")

validate_params(molecule_parameters)

print("\n=== Check for unusually large γ values (γ > 1 cm⁻¹ atm⁻¹) ===")
for key, df in db.items():
    if "gamma" not in df.columns or df.empty:
        continue
    high = df[df["gamma"] > 1.0]
    if not high.empty:
        gmax = high["gamma"].max()
        print(f"{key:20s}  max γ = {gmax:.3f}  ({len(high)} rows > 1)")
        print(f"\n--- {key} ---")
        print(high.head(5)[["gamma","T","J","Jpp","paper"]] if "paper" in high.columns else high.head(5))


# Build engineered tables
molecules = clean_and_engineer(db, molecule_parameters)

# === Temperature diagnostics ===
if USE_T:
    print("\n=== Temperature range diagnostics ===")
    temp_stats = []
    for key, df in molecules.items():
        if "T" not in df.columns or df.empty:
            continue
        Tmin, Tmax = df["T"].min(), df["T"].max()
        n = len(df)
        temp_stats.append((key, Tmin, Tmax, n))

    if temp_stats:
        tdf = pd.DataFrame(temp_stats, columns=["pair", "T_min", "T_max", "n_rows"])
        tdf = tdf.sort_values("T_min")
        print(tdf.to_string(index=False))
        global_min = tdf["T_min"].min()
        global_max = tdf["T_max"].max()
        print(f"\nOverall temperature range: {global_min:.1f}–{global_max:.1f} K")
        if global_max - global_min < 50:
            print("Warning: very narrow temperature span—model may not learn T-dependence.")
        if global_max > 2000:
            print("Note: temperatures above 2000 K detected, check for unit errors.")

# === QC: value checks, outliers, duplicates, units ===

QC = {
    "gamma_min": 1e-4,     # cm^-1 atm^-1
    "gamma_max": 2.0,      # anything above is suspicious
    "T_min": 50.0,         # K
    "T_max": 5000.0,       # K
    "J_max": 500,          # sanity
    "mad_k": 100,          # robust outlier threshold (per pair)
}

def qc_filter_and_log(molecules: dict[str, pd.DataFrame], out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Quick QC pass: print summary of suspicious values and return cleaned molecules."""
    fails = []
    summ = []

    def _mad(x):
        med = np.nanmedian(x)
        mad = 1.4826 * np.nanmedian(np.abs(x - med))
        return med, mad

    cleaned = {}

    for key, df in molecules.items():
        if df.empty:
            continue

        g = df["gamma"].astype(float)
        gu = df.get("gamma_uncertainty", np.nan)
        T = df.get("T", np.nan)
        J = df.get("J", np.nan)
        Jpp = df.get("Jpp", np.nan)

        # physical ranges
        mask_phys = (
            (g >= QC["gamma_min"]) & (g <= QC["gamma_max"]) &
            (~np.isfinite(T) | ((T >= QC["T_min"]) & (T <= QC["T_max"]))) &
            (~np.isfinite(J) | ((J >= 0) & (J <= QC["J_max"]))) &
            (~np.isfinite(Jpp) | ((Jpp >= 0) & (Jpp <= QC["J_max"])))
        ).to_numpy()

        mask_unc = (~np.isfinite(gu)) | ((gu >= 0) & (gu <= 5 * g))
        mask_unc = np.asarray(mask_unc, bool)

        med, mad = _mad(g)
        mask_out = np.ones(len(df), bool)
        if np.isfinite(mad) and mad > 0:
            z = np.abs(g - med) / mad
            mask_out = z <= QC["mad_k"]

        ok = mask_phys & mask_unc & mask_out
        bad = ~ok
        kept = df.loc[ok].copy()
        cleaned[key] = kept

        n_bad = bad.sum()
        if n_bad:
            bad_rows = df.loc[bad, ["gamma","T","J","Jpp"]].head(5)
            print(f"\n--- QC warning: {key} ---")
            print(f"Removed {n_bad} / {len(df)} rows ({100*n_bad/len(df):.1f}%)")
            print(bad_rows)
            gmin, gmax = g.min(), g.max()
            print(f"gamma range {gmin:.3g}–{gmax:.3g} cm⁻¹ atm⁻¹, median={med:.3g}, MAD={mad:.3g}")

        summ.append({
            "pair": key,
            "n_total": len(df),
            "n_removed": n_bad,
            "frac_removed": float(n_bad/len(df)),
            "gamma_min": float(np.nanmin(g)),
            "gamma_max": float(np.nanmax(g)),
            "T_min": float(np.nanmin(T)) if np.any(np.isfinite(T)) else np.nan,
            "T_max": float(np.nanmax(T)) if np.any(np.isfinite(T)) else np.nan,
        })

    print("\n=== QC summary ===")
    summary = pd.DataFrame(summ).sort_values("frac_removed", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nQC done.\n")

    return cleaned

date_tag = datetime.now().strftime("%Y-%m-%d")
qc_dir = HOME / "Scratch" / f"other_broadeners_{date_tag}" / "qc"
# molecules = qc_filter_and_log(molecules, qc_dir)   -----> BAD!  Removes data from HCl/HBr due to unusual spread./

t4 = time.time()

# === Part 5: Weighting with normalized influence ===

def compute_weights_sequential(
        molecules: Dict[str, pd.DataFrame],
        stages: List[Literal['pair', 'active', 'perturber']] = ['pair', 'active', 'perturber'],
        eps: float = 0.02,
        compression_factor: float = 0.5,
        wmin: float = 1e-3,
        wmax: float = 1e3,
) -> Dict[str, pd.DataFrame]:
    """
    Sequential Weighting: Applies normalization stages in the order specified.

    Logic:
      1. Initialize weights = Precision (1/error^2).
      2. For each stage in `stages`:
         - Group data (by Pair, Active Species, or Perturber).
         - Calculate the Group's accuracy (mean of contained data).
         - Calculate the Group's current total weight.
         - Scale weights so Total_Weight ~ Accuracy^compression_factor.

    Note: Later stages effectively 'perturb' the balance of earlier stages.
    E.g., Balancing 'perturber' (He vs N2) last might slightly shift the
    balance between 'active' species (H2O vs CO2) if their compositions differ.
    """

    # 1. Initialization: Assign base precision as weight
    #    This ensures relative row weights within a pair are always correct (cleaner rows > noisier rows)
    pair_stats = {}

    for k, df in molecules.items():
        if df.empty: continue

        sigma = df["fractional_error"].astype(float).values
        precision = 1.0 / (sigma ** 2 + eps ** 2)

        # We store this temporarily in the dataframe to track cumulative scaling
        df = df.copy()
        df["weight"] = precision
        molecules[k] = df

        # Cache stats needed for accuracy calculations to avoid re-computing
        pair_stats[k] = {
            "mean_acc": np.mean(precision)
        }

    # Helper to identify groups
    def get_group_key(pair_key: str, mode: str) -> str:
        parts = pair_key.split('_')
        if mode == 'active': return parts[0]  # "H2O" from "H2O_He"
        if mode == 'perturber': return parts[-1]  # "He" from "H2O_He"
        return pair_key  # "H2O_He" (Own group)

    # 2. Sequential Renormalization Stages
    for stage in stages:

        # A. Build Groups for this stage
        #    Map GroupName -> List of Molecules keys
        groups = {}
        for k in molecules.keys():
            if molecules[k].empty: continue
            g_key = get_group_key(k, stage)
            if g_key not in groups: groups[g_key] = []
            groups[g_key].append(k)

        # B. Calculate Group Stats
        group_data = {}
        for g_name, keys in groups.items():
            # Accuracy: Mean accuracy of the constituent pairs
            # (We use pair-level averages to avoid one huge pair dominating the accuracy score)
            avg_acc = np.mean([pair_stats[k]["mean_acc"] for k in keys])

            # Current Mass: Sum of weights currently assigned to this group
            current_mass = np.sum([molecules[k]["weight"].sum() for k in keys])

            group_data[g_name] = {
                "acc": avg_acc,
                "current_mass": current_mass
            }

        # C. Calculate Target Mass Scaling
        #    Target ~ (GroupAcc / MedianAcc) ^ Factor
        all_accs = np.array([v["acc"] for v in group_data.values()])
        median_acc = np.median(all_accs)

        for g_name, keys in groups.items():
            stats = group_data[g_name]

            # How heavy should this entire group be relative to the median group?
            if median_acc > 0:
                rel_acc = stats["acc"] / median_acc
            else:
                rel_acc = 1.0

            target_mass = rel_acc ** compression_factor

            # SCALER: Transform current mass to target mass
            if stats["current_mass"] > 0:
                scaler = target_mass / stats["current_mass"]
            else:
                scaler = 0.0

            # Apply to all constituent dataframes
            for k in keys:
                molecules[k]["weight"] *= scaler

    # 3. Final safeguards & cleanup
    all_weights = np.concatenate([df["weight"].values for df in molecules.values() if not df.empty])

    # Clip to avoid exploding gradients from tiny datasets
    all_weights = np.clip(all_weights, wmin, wmax)

    # Global Mean Normalization (Mean = 1.0)
    global_scale = 1.0 / np.mean(all_weights)

    for k, df in molecules.items():
        if df.empty: continue
        # Apply clip and global scale
        df["weight"] = np.clip(df["weight"], wmin, wmax) * global_scale
        molecules[k] = df

    return molecules

# --- Apply and Inspect ---

molecules = compute_weights_sequential(molecules)

# --- Diagnostic Print ---
print(f"{'Pair':<15} | {'N':<5} | {'Raw Acc':<10} | {'Total W':<10} | {'W/row (avg)':<10}")
print("-" * 65)

for k, df in molecules.items():
    if df.empty: continue

    raw_acc = np.mean(1.0 / (df["fractional_error"] ** 2 + 0.02 ** 2))
    total_w = df["weight"].sum()
    avg_w = total_w / len(df)

    print(f"{k:<15} | {len(df):<5} | {raw_acc:<10.1f} | {total_w:<10.2f} | {avg_w:<10.4f}")

t5 = time.time()

# === Part 6: grouped train/test split ===

def make_grouped_splits(
    molecules: Dict[str, pd.DataFrame],
    test_group_size: int = N_CROSS_VAL,
    seed: int = 41,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create cross-validation splits.
    1. Keeps active_air in the same block as active_N2 and active_O2.
    2. Groups isotopologues of Cl and Br together (e.g., 35Cl, -35Cl, 37Cl treated as same species).

    Parameters
    ----------
    molecules : dict
        Keyed by 'active_perturber' with engineered DataFrames.
    test_group_size : int
        Number of dependency groups to put in the test set per fold.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict[str, (DataFrame, DataFrame)]
        Each entry contains training and test concatenations for one fold.
    """
    rng = np.random.default_rng(seed)

    # 1. Define Dependency Groups
    dependency_groups = collections.defaultdict(list)
    air_system_perturbers = {'air', 'N2', 'O2'}

    for key in molecules.keys():
        # key format expected: "Active_Perturber"
        parts = key.rsplit('_', 1)

        if len(parts) != 2:
            dependency_groups[key].append(key)
            continue

        active, perturber = parts[0], parts[1]

        # --- STEP A: Normalize Isotope Names (Cl and Br only) ---
        # Regex explanation:
        # -?        : Matches 0 or 1 hyphen (handling "Molecule-35Cl")
        # \d+       : Matches 1 or more digits (the mass number, e.g., "35")
        # (Cl|Br)   : Matches specifically Chlorine or Bromine
        # Replacement: r'\1' keeps just the element (the group in parenthesis)

        base_active = re.sub(r'-?\d+(Cl|Br)', r'\1', active)

        # --- STEP B: Determine Group ID ---
        if perturber in air_system_perturbers:
            # Bundle Air/N2/O2 AND isotopologues together
            group_id = f"{base_active}_air_system"
        else:
            # Keep H2, He, etc. separate, but bundle isotopologues
            group_id = f"{base_active}_{perturber}"

        dependency_groups[group_id].append(key)

    # 2. Shuffle the Groups
    unique_groups = list(dependency_groups.keys())
    rng.shuffle(unique_groups)

    splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    # 3. Create Folds
    for i in range(0, len(unique_groups), test_group_size):
        test_groups = unique_groups[i: i + test_group_size]
        train_groups = [g for g in unique_groups if g not in test_groups]

        test_keys = [k for g in test_groups for k in dependency_groups[g]]
        train_keys = [k for g in train_groups for k in dependency_groups[g]]

        data_test_list = [molecules[k] for k in test_keys if not molecules[k].empty]
        data_train_list = [molecules[k] for k in train_keys if not molecules[k].empty]

        if not data_test_list or not data_train_list:
            continue

        data_test = pd.concat(data_test_list, ignore_index=True)
        data_train = pd.concat(data_train_list, ignore_index=True)

        fold_name = f"Fold_{i // test_group_size}"
        splits[fold_name] = (data_train, data_test)

    return splits

# Ensure each df has a 'pair' label for later diagnostics
for k in molecules.keys():
    molecules[k] = molecules[k].copy()
    molecules[k]["pair"] = k

# Build grouped splits
molecule_splits = make_grouped_splits(molecules, test_group_size=N_CROSS_VAL, seed=41)

# Optional: inspect
for k, (train, test) in list(molecule_splits.items()):
    print(f"{k}: train {len(train)} rows, test {len(test)} rows")

t6 = time.time()

# === Part 7: ML pipeline, training, and evaluation ===

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Infer feature columns by dropping targets/meta."""
    drop = {
        "gamma", "gamma_uncertainty", "fractional_error", "profile",
        "paper", "weight", "pair"
    }
    if not USE_T and "T" in df.columns:
        drop.add("T")
    return [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]

def build_model() -> VotingRegressor:
    """
    Weighted ensemble. All base learners respect sample_weight.
    Trees don’t need scaling, but we keep scaler for possible future features.
    """
    rbf_approx_svr = make_pipeline(
        Nystroem(
            kernel='rbf',
            n_components=500,  # tune: more components -> better, slower
            gamma=0.1,  # analogous to RBF gamma
            random_state=42
        ),
        SGDRegressor(
            loss='epsilon_insensitive',  # SVR-like loss
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3
        )
    )

    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), VotingRegressor(
        estimators=[('hist', HistGradientBoostingRegressor()),
                    ('ada', AdaBoostRegressor()),
                    ('svr', WeightRouter(rbf_approx_svr, step_name='sgdregressor')),
                    ('forest', RandomForestRegressor(n_estimators=10, min_weight_fraction_leaf=0.001, n_jobs=-1)),
                    ('mlp',
                     SampleWeightIgnoringRegressor(MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, learning_rate='adaptive', random_state=42,
                                n_iter_no_change=1, tol=0.00001)))
                    ]
        , n_jobs=-1, verbose=True
    ))
    return pipe

def evaluate_fold(model, X_test, y_test, w_test) -> dict:
    """Return weighted metrics."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred, sample_weight=w_test)
    mae = mean_absolute_error(y_test, y_pred, sample_weight=w_test)

    scores =  {
        "r2": model.score(X_test, y_test, sample_weight=w_test),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
    }, y_pred

    return scores

def _safe_name(s: str) -> str:
    return s.replace(",", "__").replace("/", "_").replace("\\", "_")

fold_results: list[dict] = []
models: list[Tuple[str, object]] = []  # (fold_name, fitted_pipeline)

for fold_name, (data_train, data_test) in itertools.islice(molecule_splits.items(), MAX_FOLDS):
    # features
    feat_cols = get_feature_columns(data_train)
    feat_cols.remove('M')
    feat_cols.remove('Ka_aprox')
    feat_cols.remove('Kapp_aprox')
    feat_cols.remove('active_d')
    feat_cols.remove('perturber_d')

    # Optional removal of self-broadening rows
    if DROP_SELF_BROADENING:
        data_train = data_train[data_train["is_self"] == 0]
        feat_cols.remove('is_self')

    # 1. Prepare Data
    X_train = data_train[feat_cols].to_numpy()
    y_train = data_train["gamma"].to_numpy()
    w_train = data_train["weight"].to_numpy()
    error_train = data_train['fractional_error'].to_numpy()

    X_test = data_test[feat_cols].to_numpy()
    y_test = data_test["gamma"].to_numpy()
    w_test = data_test["weight"].to_numpy()
    error_test = data_test['fractional_error'].to_numpy()

    # 2. ### CHANGE: Log Transform Training Target
    # Using np.log (natural log). If gamma can be 0, use np.log1p instead.
    y_train_log = np.log(y_train)

    model = build_model()

    # 3. ### CHANGE: Fit on Log-Target
    # Note: We pass y_train_log here
    model.fit(X_train, y_train_log, votingregressor__sample_weight=w_train)

    # 4. ### CHANGE: Manual Prediction & Inverse Transform
    # We cannot rely on evaluate_fold's internal metrics because it will see log-preds.
    # We predict manually to get log-scale predictions first.
    y_pred_log = model.predict(X_test)

    # Inverse transform (Un-log) to get back to physical units
    y_pred = np.exp(y_pred_log)

    # 5. ### CHANGE: Recalculate Global Metrics Manually

    # Evaluation mask: drop self-broadening from *evaluation* if requested
    if COMPARE_SELF_BROADENING:
        eval_mask = (data_test["is_self"].to_numpy() == 0)
    else:
        eval_mask = np.ones_like(y_test, dtype=bool)

    # If there is no non-self data in this fold's test set, skip this fold in evaluation
    if not np.any(eval_mask):
        print(f"{fold_name}: no non-self test data available for evaluation; skipping fold.")
        continue

    y_eval = y_test[eval_mask]
    y_pred_eval = y_pred[eval_mask]
    w_eval = w_test[eval_mask]

    # We must calculate metrics using the un-logged y_pred and original y_test
    metrics = {
        "mae": mean_absolute_error(y_eval, y_pred_eval, sample_weight=w_eval),
        "mse": mean_squared_error(y_eval, y_pred_eval, sample_weight=w_eval),
        "rmse": np.sqrt(mean_squared_error(y_eval, y_pred_eval, sample_weight=w_eval)),
        "r2": r2_score(y_eval, y_pred_eval, sample_weight=w_eval),
    }

    # --- Everything below stays mostly the same, using the un-logged y_pred ---

    # pointwise predictions for plotting
    preds = pd.DataFrame({
        "pair": data_test["pair"].to_numpy(),
        "y": y_test,  # Original physical scale
        "y_pred": y_pred,  # Un-logged physical scale
        "y_log": np.log(y_test),
        "y_pred_log": y_pred_log,
        "w": w_test,
        "y_error": error_test,
    })

    xcol = "M" if "M" in data_test.columns else None
    if xcol:
        preds[xcol] = data_test[xcol].to_numpy()

    preds.to_csv(out / f"predictions_{_safe_name(fold_name)}.csv", index=False)

    # per-pair diagnostics on test (use same eval_mask as metrics)
    test_pairs = data_test["pair"].to_numpy()[eval_mask]
    per_pair = (
        pd.DataFrame({
            "pair": test_pairs,
            "y": y_eval,
            "y_pred": y_pred_eval,
            "w": w_eval,
        })
        .groupby("pair")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "wMAE": mean_absolute_error(g["y"], g["y_pred"], sample_weight=g["w"]),
            "wMSE": mean_squared_error(g["y"], g["y_pred"], sample_weight=g["w"]),
            "bias": float(np.average(g["y_pred"] - g["y"], weights=g["w"])),
        }))
        .reset_index()
    )

    fold_results.append({
        "fold": fold_name,
        "features": feat_cols,
        **metrics,  # These are now the corrected, un-logged metrics
        "per_pair": per_pair
    })
    models.append((fold_name, model))

# Summary
overall_r2  = np.mean([fr["r2"] for fr in fold_results]) if fold_results else float("nan")
overall_mse = np.mean([fr["mse"] for fr in fold_results]) if fold_results else float("nan")
overall_mae = np.mean([fr["mae"] for fr in fold_results]) if fold_results else float("nan")
print(f"CV mean R2={overall_r2:.3f}  MSE={overall_mse:.4g}  MAE={overall_mae:.4g}")

t7 = time.time()

# === Part 8: persistence ===

try:
    from joblib import dump  # preferred for sklearn pipelines
except Exception:
    dump = None  # will fallback to pickle

def _safe_name(s: str) -> str:
    return s.replace(",", "__").replace("/", "_").replace("\\", "_")

def persist_results(
    models: list[tuple[str, object]],
    fold_results: list[dict],
    base_dir: Path = HOME,
    tag: str | None = None,
    out: Path | None = None,
) -> Path:
    """
    Persist trained models and evaluation artifacts.

    Saves:
      - model_<fold>.joblib (or .pkl if joblib unavailable)
      - cv_summary.csv
      - per_pair_<fold>.csv for each fold
      - fold_results.pkl (full Python object for reuse)
      - features.csv (union of features seen across folds)

    Returns
    -------
    Path to the output directory.
    """
    date_tag = tag or datetime.now().strftime("%Y-%m-%d")
    if out is None:
        out = base_dir / "Scratch" / f"other_broadeners_{date_tag}"
    out.mkdir(parents=True, exist_ok=True)

    # 1) save models
    for fold_name, model in models:
        fname = out / f"model_{_safe_name(fold_name)}"
        if dump is not None:
            dump(model, fname.with_suffix(".joblib"))
        else:
            with open(fname.with_suffix(".pkl"), "wb") as f:
                pickle.dump(model, f)

    # 2) summary metrics
    rows = []
    for fr in fold_results:
        per_pair = fr["per_pair"]
        rows.append({
            "fold": fr["fold"],
            "r2": fr["r2"],
            "mse": fr["mse"],
            "mae": fr["mae"],
            "n_test": int(per_pair["n"].sum()) if "n" in per_pair.columns else np.nan,
        })
        per_pair.to_csv(out / f"per_pair_{_safe_name(fr['fold'])}.csv", index=False)

    pd.DataFrame(rows).to_csv(out / "cv_summary.csv", index=False)

    # 3) union of features used across folds
    feature_union = sorted({f for fr in fold_results for f in fr.get("features", [])})
    pd.Series(feature_union, name="feature").to_csv(out / "features.csv", index=False)

    # 4) full object for later analysis
    with open(out / "fold_results.pkl", "wb") as f:
        pickle.dump(fold_results, f)

    print(f"Saved results to: {out}")
    return out


# Example call after Section 7 finishes:
out_dir = persist_results(models, fold_results, base_dir=HOME, tag=None, out=out)

t8 = time.time()

print(f"Part 1: {t1 - start:.2f}s, Part 2: {t2 - t1:.2f}s, "
      f"Part 3: {t3 - t2:.2f}s, Part 4: {t4 - t3:.2f}s, "
      f"Part 5: {t5 - t4:.2f}s, Part 6: {t6 - t5:.2f}s, "
      f"Part 7: {t7 - t6:.2f}s, Part 8: {t8 - t7:.2f}s, Total: {t8 - start:.2f}s")