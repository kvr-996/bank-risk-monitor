"""Tests for the investor-facing risk score and input validation.   python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bank_risk.ml import investment_risk_score  # noqa: E402
from bank_risk.pipeline import DATA, assess_company  # noqa: E402
from bank_risk.validate import validate_company_input  # noqa: E402


def _frame(z, m, anom=50.0):
    n = max(np.size(z), np.size(m), np.size(anom))
    return pd.DataFrame({"z_em": np.broadcast_to(z, n), "m_score": np.broadcast_to(m, n),
                         "anomaly_percentile": np.broadcast_to(anom, n), "early_warning_prob": np.nan,
                         "structural_break": False, "feature_coverage": 1.0})


def test_risk_rises_as_z_falls():
    s = investment_risk_score(_frame([6.5, 5.0, 4.0, 2.0], -2.5))["risk_score"].tolist()
    assert s == sorted(s)


def test_risk_rises_as_m_rises():
    s = investment_risk_score(_frame(5.0, [-3.0, -2.2, -2.0, -1.5]))["risk_score"].tolist()
    assert s == sorted(s)


def test_bands_and_caps():
    f = _frame([6.5, 1.5], [-3.0, -1.0], anom=[5.0, 99.0])
    out = investment_risk_score(f)
    assert out["risk_band"].tolist() == ["Low", "High"]
    f["structural_break"] = True
    assert investment_risk_score(f)["risk_band"].tolist() == ["Low", "Medium"]   # merger year capped
    f["feature_coverage"] = 0.4
    assert set(investment_risk_score(f)["risk_band"]) == {"Insufficient data"}


def test_ew_weight_zero_until_validated():
    f = _frame(5.0, -2.0)
    f["early_warning_prob"] = 1.0
    off = investment_risk_score(f, ew_validated=False)["risk_score"].iloc[0]
    on = investment_risk_score(f, ew_validated=True)["risk_score"].iloc[0]
    assert on > off


def _yes_bank(years=("FY2019", "FY2020")):
    raw = pd.read_csv(DATA / "bank_panel.csv")
    return raw[(raw["bank"] == "Yes Bank") & raw["fy"].isin(years)].drop(columns=["n_estimated_fields"])


def test_assess_yes_bank_is_high():
    r = assess_company(_yes_bank())
    assert r["risk_band"] == "High" and r["risk_score"] >= 66
    assert r["z_zone"] == "Distress" and r["reasons"]


def test_assess_needs_two_years():
    with pytest.raises(ValueError) as e:
        assess_company(_yes_bank(("FY2020",)))
    assert any("at least 2" in m for m in e.value.args[0])


def test_validation_catches_bad_input():
    d = _yes_bank().copy()
    d["market_cap"] = [0.0, 100.0]
    d["gnpa"] = [18.0, 0.2]          # a percent typed instead of a fraction
    errs, warns = validate_company_input(d, strict=True)
    assert any("market_cap" in e for e in errs)
    assert any("gnpa" in w for w in warns)


def test_blank_form_reports_missing_fields():
    d = _yes_bank().copy()
    d.loc[:, "total_assets"] = np.nan
    with pytest.raises(ValueError) as e:
        assess_company(d)
    assert any("empty required fields" in m for m in e.value.args[0])


def test_no_spurious_prior_year_warnings():
    r = assess_company(_yes_bank())
    assert not any("Beneish M not computable" in w for w in r["warnings"])