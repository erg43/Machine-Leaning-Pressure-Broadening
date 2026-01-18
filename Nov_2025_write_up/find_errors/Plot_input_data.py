# === Helpers and constants ===
# === Imports ===
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Iterable

import numpy as np
import pandas as pd

# === Environment setup ===

# Toggle this manually or via an environment variable
local = True  # set False when running on cluster

VOIGT_ONLY = True  # set False to keep all profiles
USE_T = False
USE_ELECTRONIC = True

# Base data path
if local:
    HOME = Path.home() / "Desktop" / "line_broadening.nosync"
else:
    HOME = Path.home()

# Small utility for conditional loading
def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a CSV unless it is too large for local use.
    On local runs, skip any file >50 MB (configurable).
    """
    max_size_mb = 10
    expected_cols = list(kwargs.get("usecols") or [])
    if local and path.exists() and path.stat().st_size > max_size_mb * 1024 * 1024:
        print(f"Skipping large file on local mode: {path.name}")
        return pd.DataFrame(columns=expected_cols)
    return pd.read_csv(path, **kwargs)

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

    file['paper'] = f.stem
    key = f"{i}_{j}"
    db[key] = pd.concat([db[key], file], ignore_index=True) if key in db else file.reset_index(drop=True)

# 2) HITRAN raw data
files_hit = list((HOME / "line_broadening" / "model_search" / "raw_data").glob("*.csv"))
hit_db: Dict[str, pd.DataFrame] = {}

for f in files_hit:
    i = f.stem
    data_i = read_csv_safe(f, low_memory=False, dtype={'J': np.float64, 'Jpp': np.float64})

    # build state labels EARLY so we can keep them during column pruning
    # keep or drop excited electronic states depending on your flag
    if not USE_ELECTRONIC:
        data_i, st = _keep_ground_state(data_i)
    # create canonical labels from whatever state columns exist
    data_i = unify_states(data_i)   # adds 'state_up' and 'state_low' when possible

    # choose gamma columns
    gamma_cols = [c for c in data_i.columns if c.startswith("gamma_") and "err" not in c and "ref" not in c]

    for gamma_col in gamma_cols:
        err_col = f"{gamma_col}-err"

        base_cols = ['J','Ka_aprox','Kc_aprox','Jpp','Kapp_aprox','Kcpp_aprox']
        state_cols = [c for c in ('state_up','state_low') if c in data_i.columns]
        cols = base_cols + state_cols + [gamma_col] + ([err_col] if err_col in data_i.columns else [])

        take = data_i[cols].copy()

        # numeric coercion
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
        take['paper'] = "HITRAN"
        take = take.rename(columns={gamma_col: 'gamma', err_col: 'gamma_uncertainty'})
        take['gamma'] = take['gamma'].astype(float)

        key = f"{i}_{j}"
        db[key] = pd.concat([db[key], take], ignore_index=True) if key in db else take.reset_index(drop=True)

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
    cols = ['molecule','broadener','J','Jpp','Ka_aprox','Kapp_aprox','Kc_aprox','Kcpp_aprox',
            'gamma','gamma_uncertainty','n','T','profile','paper']

    # 1) curated small sources
    small_path = base_dir / "Broadening_data_from_smaller_sources.csv"
    small = read_csv_safe(
        small_path,
        dtype={'J': np.float64, 'Jpp': np.float64, 'gamma': np.float64, 'gamma_uncertainty': np.float64},
        usecols=cols
    )

    # 2) LLM-scraped
    if llm_path.exists():
        masters = read_csv_safe(llm_path, index_col=0)
        masters = masters[cols]
        # Coerce paper to string, removing .0 if numeric
        masters["paper"] = (
            pd.to_numeric(masters["paper"], errors="coerce")
            .dropna()
            .astype("Int64")
            .astype(str)
        ).reindex(masters.index).fillna("unknown")
        data = pd.concat([small, masters], ignore_index=True)
    else:
        data = small

    # 2) LLM-scraped
    if guest_llm_path.exists():
        masters = read_csv_safe(guest_llm_path, index_col=0)
        masters = masters[cols]
        # Coerce paper to string, removing .0 if numeric
        masters["paper"] = (
            pd.to_numeric(masters["paper"], errors="coerce")
            .dropna()
            .astype("Int64")
            .astype(str)
        ).reindex(masters.index).fillna("unknown")
        data = pd.concat([data, masters], ignore_index=True)
    else:
        data = data

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
    df = df[['J','Ka_aprox','Kc_aprox','Jpp','Kapp_aprox','Kcpp_aprox','T','profile','gamma','gamma_uncertainty',
             'molecule','broadener','n','paper']]
    df = df.copy()
    df["paper"] = df["paper"].fillna("small_source")
    db[key] = pd.concat([db[key], df], ignore_index=True) if key in db else df.reset_index(drop=True)


import matplotlib.pyplot as plt

def _build_state_label(df: pd.DataFrame) -> pd.Series:
    """Return 'state_up→state_low' if available, else NaN."""
    df2 = unify_states(df)  # uses your helper
    up  = df2.get("state_up")
    low = df2.get("state_low")
    if up is None and low is None:
        return pd.Series(index=df.index, dtype="object")

    up  = up.fillna("")
    low = low.fillna("")
    lab = up + "→" + low
    lab = lab.str.strip("→")
    lab = lab.replace({"": np.nan})
    return lab

def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("J", "gamma"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _infer_source_col(df: pd.DataFrame) -> pd.Series:
    """
    Prefer a non-empty 'paper' or 'source' value per row.
    Fallback to 'HITRAN' when missing/NaN/placeholder.
    """
    s = None
    if "paper" in df.columns:
        s = df["paper"]
    else:
        return pd.Series("concern", index=df.index, dtype="object")

    s = s.astype("string")  # keeps <NA>
    # normalize empties and placeholders
    bad = s.isna() | s.str.strip().isin({"", "nan", "NaN", "None", "NULL"})
    return s.mask(bad, "HITRAN")

def plot_gamma_vs_J_by_source(db: Dict[str, pd.DataFrame],
                              keys: Iterable[str] | None = None,
                              max_sources_per_key: int | None = None) -> None:
    """
    For each key in db, make one scatter per source showing gamma vs J.
    Color encodes state_up/state_low when present.
    """
    keys_to_use = list(keys) if keys is not None else list(db.keys())
    for key in keys_to_use:
        df = db.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        df = _coerce_numeric_cols(df)
        df = df.dropna(subset=["J", "gamma"])
        if df.empty:
            continue

        # source id
        df["_source"] = _infer_source_col(df)

        # state label
        df["_state"] = _build_state_label(df).fillna("state: unknown")

        # order sources by size, largest first
        sources = df["_source"].value_counts().index.tolist()
        if max_sources_per_key is not None:
            sources = sources[:max_sources_per_key]

        for s in sources:
            # print(s)
            # if s == '122':
            #     continue
            # else:
            #     continue
            g = df[df["_source"] == s]
            if g.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for st in g["_state"].unique():
                sel = g["_state"] == st
                ax.scatter(g.loc[sel, "J"], g.loc[sel, "gamma"], s=18, label=st)

            ax.set_title(f"{key} | source: {s}")
            ax.set_ylim(0)
            ax.set_xlim(-1)
            ax.set_xlabel("J")
            ax.set_ylabel(r"$\gamma$ (cm$^{-1}$ atm$^{-1}$)")
            ax.legend(title="state_up→state_low")
            fig.tight_layout()
            plt.savefig(f"figures/{key}-{s}.png")
            plt.close(fig)  # add this


# one-time cleanup after building db
db = {k: v for k, v in db.items() if isinstance(v, pd.DataFrame)}

# plot_gamma_vs_J_by_source(db, keys=["H2O_CO2", "CO2_H2"], max_sources_per_key=None)
plot_gamma_vs_J_by_source(db, keys=db.keys(), max_sources_per_key=None)
