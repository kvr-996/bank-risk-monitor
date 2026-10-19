"""Input sanity checks for investor-entered financials.

validate_company_input(df, strict=True) -> (errors, warnings)
    errors   : the scores cannot be trusted / computed -> stop
    warnings : scores are computed but something looks off -> show to the user
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .schema import INVESTOR_REQUIRED, MIN_YEARS_FOR_INVESTOR, NUMERIC_COLUMNS, REQUIRED_MINIMUM
from .scoring import fy_sort_key


def validate_company_input(df: pd.DataFrame, strict: bool = True) -> Tuple[List[str], List[str]]:
    """`df` must already use snake_case column names. strict=True is the investor workflow."""
    errors: List[str] = []
    warns: List[str] = []
    need = INVESTOR_REQUIRED if strict else REQUIRED_MINIMUM

    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        errors.append("Missing required fields: " + ", ".join(missing_cols))
        return errors, warns
    if df.empty:
        return ["No rows supplied."], warns

    d = df.copy()
    for c in NUMERIC_COLUMNS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_fy"] = fy_sort_key(d["fy"])

    if d["_fy"].isna().any():
        errors.append("Could not read the fiscal year on some rows (use FY2024, 2024 or 2023-24).")
        return errors, warns

    for bank, g in d.groupby("bank"):
        g = g.sort_values("_fy")
        if strict and len(g) < MIN_YEARS_FOR_INVESTOR:
            errors.append(f"{bank}: enter at least {MIN_YEARS_FOR_INVESTOR} consecutive years "
                          "(Beneish M compares each year with the previous one).")
        if g["_fy"].duplicated().any():
            errors.append(f"{bank}: the same fiscal year appears more than once.")
        if len(g) > 1 and (g["_fy"].diff().dropna() != 1).any():
            warns.append(f"{bank}: years are not consecutive - Beneish indices compare against the "
                         "previous *row*, so a gap makes M unreliable.")
        if strict:
            empty = [c for c in need if c not in ("bank", "fy") and g[c].isna().any()]
            if empty:
                errors.append(f"{bank}: empty required fields -> " + ", ".join(empty))

    for c in ["total_assets", "total_liabilities", "equity", "cash", "investments", "deposits",
              "total_income", "market_cap", "fixed_assets", "depreciation"]:
        if c in d.columns and (d[c] < 0).any():
            errors.append(f"{c} cannot be negative.")
    if "market_cap" in d.columns and (d["market_cap"] <= 0).any():
        errors.append("market_cap must be > 0 (needed for Altman X4).")
    if (d["total_assets"] <= 0).any():
        errors.append("total_assets must be > 0.")

    # balance-sheet identity: assets ~ liabilities + equity
    bal = (d["total_assets"] - d["total_liabilities"] - d["equity"]).abs() / d["total_assets"]
    if (bal > 0.01).any():
        warns.append("total_assets differs from total_liabilities + equity by more than 1% on some rows - "
                     "check units / typos.")

    # ratios typed as percentages instead of fractions
    for c in ["gnpa", "nnpa", "crar", "cet1", "pcr"]:
        if c in d.columns and (d[c] > 1.5).any():
            warns.append(f"{c} looks like a percentage (value > 1). Enter fractions: 0.05 = 5%.")
    if {"gnpa", "nnpa"} <= set(d.columns) and (d["nnpa"] > d["gnpa"] + 1e-9).any():
        warns.append("nnpa is higher than gnpa on some rows - net NPA cannot exceed gross NPA.")

    # unit consistency between consecutive years: a >10x jump in assets usually means mixed units
    for bank, g in d.sort_values("_fy").groupby("bank"):
        ta = g["total_assets"].to_numpy(dtype=float)
        if len(ta) > 1 and np.any((ta[1:] / ta[:-1] > 10) | (ta[1:] / ta[:-1] < 0.1)):
            warns.append(f"{bank}: total assets changed >10x between years - are both years in the same units?")
    return errors, warns