"""End-to-end helpers used by train.py, the notebook and the API."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import math
import numpy as np

from .ml import RiskModels, event_validation, load_events, make_labels
from .schema import INVESTOR_REQUIRED, RAW_COLUMN_MAP, REQUIRED_MINIMUM
from .scoring import Params, score_panel
from .validate import validate_company_input

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = MODELS / "risk_models.joblib"


def read_table(path_or_buffer, filename: str | None = None) -> pd.DataFrame:
    """Read CSV or XLSX (first sheet, or a sheet called 'Inputs')."""
    name = (filename or str(path_or_buffer)).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(path_or_buffer)
        sheet = "Inputs" if "Inputs" in xl.sheet_names else xl.sheet_names[0]
        return xl.parse(sheet)
    return pd.read_csv(path_or_buffer)


def train_and_save(panel_csv=DATA / "bank_panel.csv", events_csv=DATA / "events.csv",
                   horizon: int = 1, params: Params | None = None, verbose: bool = True):
    raw = pd.read_csv(panel_csv)
    events = load_events(events_csv) if Path(events_csv).exists() else None
    scored = score_panel(raw, params)
    models = RiskModels.fit(scored, events, horizon=horizon)
    pred = models.predict(scored, events)
    labels = make_labels(scored, events, horizon)
    val = event_validation(pred, labels)

    MODELS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
    models.save(MODEL_PATH)
    pred.assign(event_label=labels).to_csv(OUTPUTS / "scores_and_predictions.csv", index=False)
    val.to_csv(OUTPUTS / "event_validation.csv", index=False)
    report = dict(models.report)
    report["validation_auc"] = {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in val.attrs["auc"].items()}
    report["params"] = (params or Params()).to_dict()
    (OUTPUTS / "training_report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print(f"scored {len(pred)} bank-years; model saved to {MODEL_PATH}")
    return models, pred, labels, val


# --------------------------------------------------------------------------- shared inference path
# api.py and app.py both call these, so the website and the Streamlit app can never disagree.
_STATE: dict = {}

# Beneish (1999) mean index values of manipulators; an index above these is a red flag on its own.
BENEISH_REF = {"dsri": 1.465, "gmi": 1.193, "aqi": 1.254, "sgi": 1.607, "depi": 1.077, "sgai": 1.041, "lvgi": 1.111, "tata": 0.031}
BENEISH_NAMES = {"dsri": "Receivables growth (DSRI)", "gmi": "Margin deterioration (GMI)", "aqi": "Asset-quality drift (AQI)",
                 "sgi": "Very fast growth (SGI)", "depi": "Slowing depreciation (DEPI)", "sgai": "SG&A ratio (SGAI)",
                 "lvgi": "Rising leverage (LVGI)", "tata": "Accruals vs cash flow (TATA)"}


def load_models() -> RiskModels:
    """Load the saved models once; retrain automatically if the file is missing or unreadable
    (e.g. built with a different scikit-learn version)."""
    if "models" not in _STATE:
        if not MODEL_PATH.exists():
            train_and_save(verbose=False)
        try:
            _STATE["models"] = RiskModels.load(MODEL_PATH)
        except Exception:  # noqa: BLE001
            train_and_save(verbose=False)
            _STATE["models"] = RiskModels.load(MODEL_PATH)
        ev = DATA / "events.csv"
        _STATE["events"] = load_events(ev) if ev.exists() else None
    return _STATE["models"]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept snake_case keys or the original workbook headers ('Total assets', ...)."""
    return df.rename(columns=lambda c: RAW_COLUMN_MAP.get(str(c).strip(), str(c).strip()))


def json_safe(df: pd.DataFrame) -> list:
    out = df.astype(object).where(df.notna(), None)
    recs = out.to_dict(orient="records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, np.integer):
                r[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                r[k] = None if (v is None or math.isnan(v) or math.isinf(v)) else float(v)
            elif isinstance(v, np.bool_):
                r[k] = bool(v)
    return recs


def score_frame(df: pd.DataFrame, params: Params | None = None, strict: bool = False, latest_only: bool = False):
    """Validate + score + ML + risk score for a table of bank-years.

    strict=True  -> investor workflow: INVESTOR_REQUIRED fields and >= 2 years per company are mandatory.
    strict=False -> batch/API workflow: only REQUIRED_MINIMUM is mandatory (lenient, with warnings).
    latest_only=True -> data-quality warnings are raised only for each company's latest year (the one the investor
    is asking about); earlier years legitimately lack M-scores because they have no prior year.
    Returns (pred_dataframe, warnings). Raises ValueError(list_of_errors) when the input is unusable.
    """
    df = normalise_columns(df)
    if not strict and {"bank", "fy", "total_assets"} <= set(df.columns):
        n0 = len(df)
        df = df[df["bank"].notna() & df["fy"].notna() & df["total_assets"].notna()]
        dropped = n0 - len(df)
    else:
        dropped = 0
    errors, warns = validate_company_input(df, strict=strict)
    if errors:
        raise ValueError(errors)
    if dropped:
        warns.append(f"{dropped} row(s) without bank / fy / total_assets were ignored.")
    models = load_models()
    scored = score_panel(df, params)
    pred = models.predict(scored, _STATE.get("events"))
    chk = pred.sort_values("fy_num").groupby("bank").tail(1) if latest_only else pred
    if chk["z_em"].isna().any():
        warns.append("Altman Z'' not computable - usually market_cap (or cash/investments) is missing.")
    if chk["m_score"].isna().any():
        warns.append("Beneish M not computable - it needs the previous fiscal year for the same company, plus total "
                     "income, interest expense, depreciation and fixed assets.")
    if not latest_only and pred["roa_sigma_3y"].isna().all():
        warns.append("Boyd-Graham Z needs at least 3 consecutive years per bank.")
    if (chk["feature_coverage"] < 0.6).any():
        warns.append("Fewer than 60% of ML features are available - the risk band is 'Insufficient data'.")
    if not models.report["early_warning"].get("validated", False):
        warns.append("The early-warning model is not validated, so it is excluded from the risk score "
                     "(risk score = Z'' + Beneish M + anomaly model only).")
    return pred, warns


def explain_row(r: pd.Series) -> list:
    """Plain-English reasons behind one result row."""
    why = []
    if pd.notna(r.get("z_em")):
        why.append(f"Altman Z''-EM = {r['z_em']:.2f} ({r['z_zone']} zone; Safe > 5.85, Distress < 4.35).")
    if pd.notna(r.get("m_score")):
        flag = "above" if r["m_score"] > -1.78 else "below"
        why.append(f"Beneish M = {r['m_score']:.2f}, {flag} the -1.78 manipulation cut-off.")
    hot = [BENEISH_NAMES[k] for k, ref in BENEISH_REF.items() if pd.notna(r.get(k)) and r[k] > ref]
    if hot:
        why.append("Beneish indices above the typical-manipulator level: " + ", ".join(hot) + ".")
    if pd.notna(r.get("anomaly_percentile")):
        why.append(f"Financial pattern is more unusual than {r['anomaly_percentile']:.0f}% of the training bank-years.")
    if bool(r.get("structural_break", False)):
        why.append("Merger year: Beneish indices are not comparable, so the band is capped at Medium.")
    return why


def assess_company(df: pd.DataFrame, params: Params | None = None) -> dict:
    """Investor entry point: financials of ONE company (>= 2 consecutive years) -> latest-year assessment.

    Raises ValueError(list_of_errors) on unusable input.
    """
    pred, warns = score_frame(df, params, strict=True, latest_only=True)
    if pred["bank"].nunique() != 1:
        raise ValueError(["Enter one company at a time."])
    r = pred.sort_values("fy_num").iloc[-1]
    keep = ["x1_liquidity", "x2_reserves_ta", "x3_profit_ta", "x4_mcap_liab", "dsri", "gmi", "aqi", "sgi", "depi",
            "sgai", "lvgi", "tata"]
    return {
        "company": r["bank"], "fy": r["fy"],
        "risk_score": None if pd.isna(r["risk_score"]) else float(r["risk_score"]),
        "risk_band": r["risk_band"], "main_driver": r["main_driver"],
        "z_em": None if pd.isna(r["z_em"]) else float(r["z_em"]), "z_zone": r["z_zone"],
        "m_score": None if pd.isna(r["m_score"]) else float(r["m_score"]), "m_flag": r["m_flag_a"],
        "joint_signal": r["joint_signal"], "anomaly_percentile": None if pd.isna(r["anomaly_percentile"]) else float(r["anomaly_percentile"]),
        "reasons": explain_row(r), "warnings": warns,
        "components": {k: (None if pd.isna(r[k]) else float(r[k])) for k in keep},
        "score_parts": {k: (None if pd.isna(r[f"risk_part_{k}"]) else float(r[f"risk_part_{k}"])) for k in ["z", "m", "anom", "ew"]},
        "input_notes": r["input_notes"],
        "history": json_safe(pred.sort_values("fy_num")[["fy", "z_em", "m_score", "risk_score", "risk_band"]]),
    }