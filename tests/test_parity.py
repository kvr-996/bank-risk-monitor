"""Check that bank_risk.scoring reproduces the workbook's own Ratios tab.

Run:  python -m pytest tests -q      (or: python tests/test_parity.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bank_risk.scoring import Params, score_panel  # noqa: E402

# workbook column -> engine column
PAIRS = {
    "X1 liquidity": "x1_liquidity",
    "X2 reserves/TA": "x2_reserves_ta",
    "X3 profit/TA": "x3_profit_ta",
    "X4 mcap/liab": "x4_mcap_liab",
    "Z'' raw": "z_raw",
    "Z''-EM (+3.25)": "z_em",
    "DSRI": "dsri", "GMI": "gmi", "AQI": "aqi", "SGI": "sgi",
    "DEPI": "depi", "SGAI": "sgai", "LVGI": "lvgi", "TATA": "tata",
    "Beneish M-score": "m_score",
    "ROA": "roa",
    "NIM (NII/earning assets)": "nim",
}
TEXT_PAIRS = {"Z'' zone": "z_zone", "M flag @ cut-off A": "m_flag_a",
              "M flag @ cut-off B": "m_flag_b", "Joint Z+M signal": "joint_signal"}


def _load():
    raw = pd.read_csv(ROOT / "data" / "bank_panel.csv")
    wb = pd.read_csv(ROOT / "data" / "workbook_ratios.csv", keep_default_na=False, na_values=[""])
    res = score_panel(raw, Params())
    wb = wb.rename(columns={"Bank": "bank", "FY": "fy"})
    return res.merge(wb, on=["bank", "fy"], how="inner", suffixes=("", "_wb"))


def test_numeric_parity():
    m = _load()
    assert len(m) == 54
    for wb_col, my_col in PAIRS.items():
        a = pd.to_numeric(m[my_col], errors="coerce")
        b = pd.to_numeric(m[wb_col], errors="coerce")
        assert (a.isna() == b.isna()).all(), f"NaN pattern differs for {wb_col}"
        ok = a.notna()
        assert np.allclose(a[ok], b[ok], rtol=1e-6, atol=1e-8), f"{wb_col}: max diff {(a[ok]-b[ok]).abs().max()}"


def test_text_parity():
    m = _load()
    for wb_col, my_col in TEXT_PAIRS.items():
        mism = (m[my_col].astype(str) != m[wb_col].astype(str))
        assert not mism.any(), f"{wb_col}: {m.loc[mism, ['bank','fy',my_col,wb_col]].head()}"


if __name__ == "__main__":
    test_numeric_parity()
    test_text_parity()
    print("parity OK: engine matches workbook Ratios tab on all 54 bank-years")
