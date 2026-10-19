"""FastAPI backend for the website.

Run:   uvicorn api:app --reload --port 8000
Docs:  http://localhost:8000/docs   (interactive Swagger UI - the front-end team can try every endpoint)

Endpoints
    GET  /health            liveness + which model file is loaded
    GET  /schema            column names the front-end form / CSV needs (required vs optional)
    GET  /template.csv      downloadable CSV template (headers + 2 example rows)
    GET  /panel             the bundled 6-bank demo panel, already scored (for a dashboard)
    POST /score             JSON rows  -> scores, zones, flags, anomaly %, risk tier, risk score, warnings
    POST /assess            INVESTOR endpoint: one company, current + previous year -> risk score & reasons
    POST /score/upload      CSV / XLSX file -> same response
"""
from __future__ import annotations

import io
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from bank_risk.pipeline import (DATA, MODEL_PATH, OUTPUTS, assess_company, json_safe, load_models, read_table,
                                score_frame, train_and_save)
from bank_risk.schema import INVESTOR_OPTIONAL, INVESTOR_REQUIRED, NUMERIC_COLUMNS, REQUIRED_MINIMUM
from bank_risk.scoring import Params

app = FastAPI(title="Bank distress & manipulation screener", version="2.0.0",
              description="Altman Z'' (bank-adapted) + Beneish M (bank-mapped) + ML anomaly / early-warning layer, "
                          "combined into a 0-100 investment risk score.")

# Tighten allow_origins to your site's domain before deploying.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- request models
class Options(BaseModel):
    x3_definition: int = Field(1, description="1 = PBT/TA ; 2 = (PBT + interest expense)/TA")
    receivables_proxy: int = Field(2, description="1 = interest receivable ; 2 = net/gross advances")
    joint_cutoff: int = Field(1, description="1 = Beneish -1.78 ; 2 = -2.22")
    assumed_tax_rate: float = Field(0.25, ge=0, lt=1, description="used only when PBT is missing")
    neutral_fill: bool = Field(False, description="treat missing Beneish indices as 1.0 (neutral)")


class ScoreRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="One object per bank-year, snake_case keys (see /schema)")
    options: Options = Options()


class AssessRequest(BaseModel):
    company: str = Field(..., description="Company / bank name")
    current: Dict[str, Any] = Field(..., description="Latest fiscal year, snake_case keys incl. 'fy' (see /schema)")
    previous: Dict[str, Any] = Field(..., description="Immediately preceding fiscal year, same keys")
    earlier: Dict[str, Any] | None = Field(None, description="Optional 3rd year (enables Boyd-Graham Z)")
    options: Options = Options()


def _params(o: Options) -> Params:
    return Params(x3_definition=o.x3_definition, receivables_proxy=o.receivables_proxy, joint_cutoff=o.joint_cutoff,
                  assumed_tax_rate=o.assumed_tax_rate, neutral_fill=o.neutral_fill)


KEEP = ["bank", "fy", "risk_score", "risk_band", "main_driver", "z_em", "z_zone", "m_score", "m_flag_a", "m_flag_b",
        "joint_signal", "anomaly_percentile", "early_warning_prob", "risk_tier", "structural_break",
        "x1_liquidity", "x2_reserves_ta", "x3_profit_ta", "x4_mcap_liab",
        "dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata",
        "boyd_graham_z", "ln_boyd_graham_z", "roa", "nim", "feature_coverage", "input_notes"]


def _score(df: pd.DataFrame, opt: Options) -> dict:
    try:
        pred, warns = score_frame(df, _params(opt), strict=False)
    except ValueError as e:
        raise HTTPException(422, detail={"error": e.args[0], "hint": "GET /schema lists required and optional columns"})
    m = load_models()
    return {"n_rows": int(len(pred)), "results": json_safe(pred[[c for c in KEEP if c in pred.columns]]),
            "warnings": warns,
            "model_notes": {"early_warning": m.report["early_warning"]["note"],
                            "scores_are": "Z'' and M are exact closed-form formulas; anomaly_percentile and "
                                          "early_warning_prob are ML outputs; risk_score is a weighted composite."}}


# --------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    m = load_models()
    return {"status": "ok", "model_file": str(MODEL_PATH.name), "trained_at": m.report["trained_at"],
            "early_warning_validated": m.report["early_warning"].get("validated", False)}


@app.get("/schema")
def schema():
    optional = [c for c in NUMERIC_COLUMNS if c not in REQUIRED_MINIMUM] + ["segment", "basis"]
    return {"required": REQUIRED_MINIMUM, "optional": optional,
            "investor_required": INVESTOR_REQUIRED, "investor_optional": INVESTOR_OPTIONAL,
            "units": "Rs crore (any consistent unit works); ratios such as gnpa/nnpa/crar as fractions (0.05 = 5%)",
            "fy_format": "FY2019, 2019 or 2018-19 (all mean the year ended March 2019)",
            "workbook_headers_accepted": True}


@app.get("/template.csv", response_class=PlainTextResponse)
def template():
    raw = pd.read_csv(DATA / "bank_panel.csv")
    ex = raw[(raw["bank"] == "Yes Bank") & raw["fy"].isin(["FY2019", "FY2020"])].drop(columns=["n_estimated_fields"])
    return ex.to_csv(index=False)


@app.get("/panel")
def panel():
    path = OUTPUTS / "scores_and_predictions.csv"
    if not path.exists():
        train_and_save(verbose=False)
    df = pd.read_csv(path)
    keep = ["bank", "fy", "risk_score", "risk_band", "z_em", "z_zone", "m_score", "m_flag_a", "joint_signal",
            "anomaly_percentile", "early_warning_prob", "risk_tier", "structural_break", "n_estimated_fields", "event_label"]
    return {"n_rows": len(df), "rows": json_safe(df[[c for c in keep if c in df.columns]])}


@app.post("/score")
def score(req: ScoreRequest):
    return _score(pd.DataFrame(req.rows), req.options)


@app.post("/assess")
def assess(req: AssessRequest):
    """Investor endpoint: financials of ONE company for >= 2 consecutive years -> risk score for the latest year."""
    rows = [req.previous, req.current] if req.earlier is None else [req.earlier, req.previous, req.current]
    df = pd.DataFrame([{**r, "bank": req.company} for r in rows])
    try:
        return assess_company(df, _params(req.options))
    except ValueError as e:
        raise HTTPException(422, detail={"error": e.args[0], "hint": "GET /schema lists investor_required fields"})


@app.post("/score/upload")
async def score_upload(file: UploadFile = File(...), x3_definition: int = 1, receivables_proxy: int = 2,
                       joint_cutoff: int = 1, assumed_tax_rate: float = 0.25, neutral_fill: bool = False):
    content = await file.read()
    try:
        df = read_table(io.BytesIO(content), file.filename)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read file: {e}")
    opt = Options(x3_definition=x3_definition, receivables_proxy=receivables_proxy, joint_cutoff=joint_cutoff,
                  assumed_tax_rate=assumed_tax_rate, neutral_fill=neutral_fill)
    return _score(df, opt)