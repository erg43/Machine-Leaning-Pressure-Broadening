# diagnostics_plot_other_broadeners.py
from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ----
# 1) Set OUT_DIR explicitly, or leave None to auto-pick latest "other_broadeners_*" under ~/Desktop/line_broadening.nosync/Scratch
OUT_DIR: str | None = None
MYRIAD = True

LOCAL = True
if LOCAL:
    BASE = Path.home() / "Desktop" / "line_broadening.nosync" / "Scratch"
else:
    BASE = Path.home() / "Scratch"

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

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---- Locate run ----
out_dir = Path(OUT_DIR) if OUT_DIR else pick_latest_run(BASE)
plots_dir = out_dir / "plots"
ensure_dir(plots_dir)
# pred_files = sorted(out_dir.glob("predictions_*.csv"))



df = pd.read_csv(out_dir / "predictions_Fold_8.csv")
df["y_error"] = pd.to_numeric(df["y_error"], errors="coerce")
df["abs_unc"] = (df["y_error"] * df["y"].abs())
print(df["abs_unc"].describe())

pair_stats = df.groupby("pair")["abs_unc"].mean().sort_values(ascending=False)
print(pair_stats.head(10))
