"""Deterministic scoring engine: Altman Z'' (bank-adapted), Beneish M (bank-mapped), Boyd-Graham Z.

This module reproduces, formula for formula, the `Ratios` and `Parameters` tabs of
Literature_Review_Research_Gap_with_Bank_Model.xlsx, so results can be checked against the
workbook (see tests/test_parity.py).

Why this is NOT a trained model: Z'' and M are closed-form formulas with published coefficients.
A learned model would only approximate them, so they are computed exactly. The learned part of
the project lives in bank_risk/ml.py and sits on top of these scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd

from .schema import NUMERIC_COLUMNS, TEXT_COLUMNS


# --------------------------------------------------------------------------- parameters
@dataclass
class Params:
    # Altman Z'' (non-manufacturer / emerging-market) -- Paper_Audit A1: 6.56, not the 6.65 typo in Paper A
    w_x1: float = 6.56
    w_x2: float = 3.26
    w_x3: float = 6.72
    w_x4: float = 1.05
    em_constant: float = 3.25          # Paper_Audit A3: omitted in Paper A
    z_safe: float = 5.85               # Z''-EM > 5.85  -> Safe      (= 2.60 without the constant)
    z_distress: float = 4.35           # Z''-EM < 4.35  -> Distress  (= 1.10 without the constant)

    # Beneish (1999) 8-variable model
    m_intercept: float = -4.84
    m_dsri: float = 0.920
    m_gmi: float = 0.528
    m_aqi: float = 0.404
    m_sgi: float = 0.892
    m_depi: float = 0.115
    m_sgai: float = -0.172
    m_tata: float = 4.679
    m_lvgi: float = -0.327
    m_cutoff_a: float = -1.78          # Beneish's commonly used cut-off
    m_cutoff_b: float = -2.22          # cut-off quoted in Parikh (2024)
    joint_cutoff: int = 1              # 1 -> A, 2 -> B (used only for the joint Z+M signal)

    # bank-adaptation switches (stress-test assumptions)
    x3_definition: int = 1             # 1 = PBT/TA ; 2 = (PBT + interest expense)/TA (replicates Paper A)
    receivables_proxy: int = 2         # 1 = interest receivable ; 2 = net (else gross) advances

    # fallbacks used ONLY when a field is missing (website users won't have every column)
    assumed_tax_rate: float = 0.25     # PBT ~= net profit / (1 - t) when PBT is absent
    neutral_fill: bool = False         # True -> missing Beneish indices become 1.0 (neutral) instead of NaN

    def m_cutoff(self) -> float:
        return self.m_cutoff_a if self.joint_cutoff == 1 else self.m_cutoff_b

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- helpers
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def fy_sort_key(fy: pd.Series) -> pd.Series:
    """'FY2019' -> 2019 (also accepts '2019', '2018-19')."""
    s = fy.astype(str).str.extract(r"(\d{4})(?:\D+(\d{2}))?")
    year = pd.to_numeric(s[0], errors="coerce")
    # '2018-19' means FY ending 2019
    end = pd.to_numeric(s[1], errors="coerce")
    year = np.where(end.notna(), (year // 100) * 100 + end, year)
    return pd.Series(year, index=fy.index)


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column set / dtypes, sort by bank and fiscal year."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in TEXT_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in NUMERIC_COLUMNS:
        df[c] = _num(df, c)
    df["fy_num"] = fy_sort_key(df["fy"])
    df = df.sort_values(["bank", "fy_num"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- fallbacks
def fill_derived_fields(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Derive any missing helper field from the ones a website user is likely to have.

    Each fallback sets a note so the UI can show *what was assumed*. Anything the user
    supplies directly is never overwritten.
    """
    df = df.copy()
    notes: List[List[str]] = [[] for _ in range(len(df))]

    def note(mask: pd.Series, text: str):
        for i in np.flatnonzero(mask.to_numpy()):
            notes[i].append(text)

    # total liabilities
    m = df["total_liabilities"].isna() & df["total_assets"].notna() & df["equity"].notna()
    df.loc[m, "total_liabilities"] = df.loc[m, "total_assets"] - df.loc[m, "equity"]
    note(m, "total_liabilities = total_assets - equity")

    # total income
    m = df["total_income"].isna() & df["interest_income"].notna() & df["other_income"].notna()
    df.loc[m, "total_income"] = df.loc[m, "interest_income"] + df.loc[m, "other_income"]
    note(m, "total_income = interest_income + other_income")

    # interest expense (NII = interest income - interest expense)
    m = df["interest_expense"].isna() & df["interest_income"].notna() & df["net_interest_income"].notna()
    df.loc[m, "interest_expense"] = df.loc[m, "interest_income"] - df.loc[m, "net_interest_income"]
    note(m, "interest_expense = interest_income - net_interest_income")

    # borrowings + other liabilities: X1 only needs their SUM, which is total liabilities - deposits
    both = df["borrowings"].notna() & df["other_liabilities"].notna()
    df["borrow_plus_other"] = np.where(both, df["borrowings"] + df["other_liabilities"],
                                       df["total_liabilities"] - df["deposits"])
    note(~both & df["borrow_plus_other"].notna(), "borrowings+other_liabilities = total_liabilities - deposits")

    # reserves (X2 proxy for retained earnings)
    m = df["reserves"].isna() & df["equity"].notna() & df["equity_capital"].notna()
    df.loc[m, "reserves"] = df.loc[m, "equity"] - df.loc[m, "equity_capital"]
    note(m, "reserves = equity - equity_capital")
    m = df["reserves"].isna() & df["equity"].notna()
    df.loc[m, "reserves"] = df.loc[m, "equity"]
    note(m, "reserves ~ total equity (no equity_capital given; slightly overstates X2)")

    # PBT
    m = df["pbt"].isna() & df["net_profit"].notna()
    df.loc[m, "pbt"] = df.loc[m, "net_profit"] / (1 - p.assumed_tax_rate)
    note(m, f"pbt = net_profit / (1 - {p.assumed_tax_rate:.0%} assumed tax)")

    # advances used as receivables proxy
    df["advances_used"] = df["net_advances"].where(df["net_advances"].notna(), df["gross_advances"])
    # earning assets proxy used by the workbook: cash + investments + net advances
    m = df["earning_assets"].isna() & df[["cash", "investments", "advances_used"]].notna().all(axis=1)
    df.loc[m, "earning_assets"] = df.loc[m, "cash"] + df.loc[m, "investments"] + df.loc[m, "advances_used"]
    note(m, "earning_assets = cash + investments + advances (proxy)")

    df["input_notes"] = ["; ".join(n) for n in notes]
    return df


# --------------------------------------------------------------------------- Altman Z''
CLASSIC_Z_WEIGHTS = (1.2, 1.4, 3.3, 0.6, 1.0)      # Altman (1968)
CLASSIC_Z_SAFE, CLASSIC_Z_DISTRESS = 2.99, 1.81


def altman_z(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    ta, tl = df["total_assets"], df["total_liabilities"]
    out = pd.DataFrame(index=df.index)
    out["x1_liquidity"] = (df["cash"] + df["investments"] - df["borrow_plus_other"]) / ta
    out["x2_reserves_ta"] = df["reserves"] / ta
    if p.x3_definition == 1:
        out["x3_profit_ta"] = df["pbt"] / ta
    else:
        out["x3_profit_ta"] = (df["pbt"] + df["interest_expense"]) / ta
    out["x4_mcap_liab"] = df["market_cap"] / tl
    # X5 (sales / total assets) -> total income / total assets. NOT used by Z'' (Altman drops it for non-manufacturers),
    # but it is the fifth ratio of the classic Altman Z, computed below for reference.
    out["x5_income_ta"] = df["total_income"] / ta
    comps = out[["x1_liquidity", "x2_reserves_ta", "x3_profit_ta", "x4_mcap_liab"]]
    raw = (p.w_x1 * comps["x1_liquidity"] + p.w_x2 * comps["x2_reserves_ta"]
           + p.w_x3 * comps["x3_profit_ta"] + p.w_x4 * comps["x4_mcap_liab"])
    raw = raw.where(comps.notna().all(axis=1))
    out["z_raw"] = raw
    out["z_em"] = raw + p.em_constant
    zone = np.where(out["z_em"] > p.z_safe, "Safe", np.where(out["z_em"] < p.z_distress, "Distress", "Grey"))
    out["z_zone"] = np.where(out["z_em"].isna(), "n/a", zone)

    # Classic Altman Z (1968), five ratios: 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5 ; Safe > 2.99, Distress < 1.81.
    # Built for US manufacturers - on banks X1 is a liquidity proxy and X5 is income/assets, so read it as a reference only.
    five = out[["x1_liquidity", "x2_reserves_ta", "x3_profit_ta", "x4_mcap_liab", "x5_income_ta"]]
    classic = (CLASSIC_Z_WEIGHTS[0] * five["x1_liquidity"] + CLASSIC_Z_WEIGHTS[1] * five["x2_reserves_ta"]
               + CLASSIC_Z_WEIGHTS[2] * five["x3_profit_ta"] + CLASSIC_Z_WEIGHTS[3] * five["x4_mcap_liab"]
               + CLASSIC_Z_WEIGHTS[4] * five["x5_income_ta"])
    out["z_classic"] = classic.where(five.notna().all(axis=1))
    czone = np.where(out["z_classic"] > CLASSIC_Z_SAFE, "Safe", np.where(out["z_classic"] < CLASSIC_Z_DISTRESS, "Distress", "Grey"))
    out["z_classic_zone"] = np.where(out["z_classic"].isna(), "n/a", czone)
    return out


# --------------------------------------------------------------------------- Beneish M (bank-mapped)
def beneish_m(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Mapping of Beneish variables to a bank balance sheet (Parameters!A33:A40):

    Sales  -> total income          COGS -> interest expense   (gross margin = (income - int. exp.)/income)
    SGA    -> operating expenses / total income
    Receivables -> net (else gross) advances, or interest receivable if switch = 1
    AQI    -> 1 - earning assets / total assets
    DEPI   -> depreciation / (depreciation + fixed assets)
    LVGI   -> total liabilities / total assets
    TATA   -> (net profit - operating cash flow) / total assets
    """
    out = pd.DataFrame(index=df.index)
    ta = df["total_assets"]
    inc = df["total_income"]
    recv = df["interest_receivable"] if p.receivables_proxy == 1 else df["advances_used"]

    out["receivables_used"] = recv
    out["gross_margin"] = (inc - df["interest_expense"]) / inc
    out["non_core_share"] = 1 - df["earning_assets"] / ta
    out["sga_income"] = df["operating_expenses"] / inc
    out["dep_rate"] = df["depreciation"] / (df["depreciation"] + df["fixed_assets"])
    out["leverage"] = df["total_liabilities"] / ta
    out["tata"] = (df["net_profit"] - df["operating_cash_flow"]) / ta

    g = df["bank"]

    def lag(s: pd.Series) -> pd.Series:
        return s.groupby(g).shift(1)

    def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        return (num / den).where(den != 0)

    out["dsri"] = ratio(recv / inc, lag(recv) / lag(inc))
    out["gmi"] = ratio(lag(out["gross_margin"]), out["gross_margin"])
    out["aqi"] = ratio(out["non_core_share"], lag(out["non_core_share"]))
    out["sgi"] = ratio(inc, lag(inc))
    out["depi"] = ratio(lag(out["dep_rate"]), out["dep_rate"])
    out["sgai"] = ratio(out["sga_income"], lag(out["sga_income"]))
    out["lvgi"] = ratio(out["leverage"], lag(out["leverage"]))

    idx = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi"]
    if p.neutral_fill:
        for c in idx:
            # Only neutralise when the *inputs* were absent; real computed values are kept.
            out[c] = out[c].where(out[c].notna(), 1.0)
    full = out[idx + ["tata"]].notna().all(axis=1)
    m = (p.m_intercept + p.m_dsri * out["dsri"] + p.m_gmi * out["gmi"] + p.m_aqi * out["aqi"]
         + p.m_sgi * out["sgi"] + p.m_depi * out["depi"] + p.m_sgai * out["sgai"]
         + p.m_tata * out["tata"] + p.m_lvgi * out["lvgi"])
    out["m_score"] = m.where(full)
    out["m_flag_a"] = np.where(out["m_score"].isna(), "n/a", np.where(out["m_score"] > p.m_cutoff_a, "Flag", "OK"))
    out["m_flag_b"] = np.where(out["m_score"].isna(), "n/a", np.where(out["m_score"] > p.m_cutoff_b, "Flag", "OK"))
    return out


# --------------------------------------------------------------------------- Boyd-Graham bank Z + extras
def boyd_graham(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["roa"] = df["net_profit"] / df["total_assets"]
    out["equity_ta"] = df["equity"] / df["total_assets"]
    # sample st. dev. of ROA over a rolling 3-year window within each bank (first value in 3rd year)
    sigma = out["roa"].groupby(df["bank"]).transform(lambda s: s.rolling(3, min_periods=3).std(ddof=1))
    out["roa_sigma_3y"] = sigma
    bg = (out["roa"] + out["equity_ta"]) / sigma
    out["boyd_graham_z"] = bg.where(sigma > 0)
    out["ln_boyd_graham_z"] = np.where(out["boyd_graham_z"] > 0, np.log(out["boyd_graham_z"].where(out["boyd_graham_z"] > 0)), np.nan)
    return out


def bank_extras(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["nim"] = df["net_interest_income"] / df["earning_assets"]
    out["gnpa_nnpa_gap_pp"] = (df["gnpa"] - df["nnpa"]) * 100
    out["d_gnpa_pp"] = df["gnpa"].groupby(df["bank"]).diff() * 100
    out["fee_income_share"] = df["other_income"] / df["total_income"]
    return out


# --------------------------------------------------------------------------- public API
def score_panel(raw: pd.DataFrame, params: Params | None = None) -> pd.DataFrame:
    """Compute Z''-EM, zone, Beneish M, flags, joint Z+M signal for every bank-year.

    `raw` needs at least the columns in schema.REQUIRED_MINIMUM. Missing optional columns are
    derived (see fill_derived_fields) and listed in `input_notes`.
    """
    p = params or Params()
    df = fill_derived_fields(clean_input(raw), p)
    z = altman_z(df, p)
    m = beneish_m(df, p)
    bg = boyd_graham(df)
    ex = bank_extras(df)

    keep = [c for c in ["bank", "segment", "fy", "fy_num", "n_estimated_fields", "input_notes"] if c in df.columns]
    res = pd.concat([df[keep], z, m, bg, ex], axis=1)
    # carry through raw asset-quality / capital fields the ML layer uses
    for c in ["gnpa", "nnpa", "crar", "cet1", "pcr"]:
        res[c] = df[c]

    cut = p.m_cutoff()
    mflag = res["m_score"] > cut
    both = res["z_zone"].eq("Distress") & mflag
    res["joint_signal"] = np.select(
        [res["m_score"].isna() | res["z_em"].isna(), both, res["z_zone"].eq("Distress"), mflag],
        ["n/a", "Z + M both flag", "Z only", "M only"],
        default="Neither",
    )
    return res
