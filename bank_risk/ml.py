"""Machine-learning layer that sits on top of the deterministic Z'' / M scores.

Two models, deliberately small because the panel is small (6 banks x 9 years = 54 rows):

1. AnomalyModel (unsupervised)  - IsolationForest on the Z'' components, the eight Beneish indices and
   asset-quality / capital fields. It needs NO labels, so scoring Yes Bank against it is a fair,
   out-of-sample-style test of "does the pattern look unusual vs the rest of the panel".

2. EarlyWarningModel (supervised) - L2 logistic regression predicting "distress event within the next
   `horizon` fiscal years" from events.csv. With a single distressed bank in the panel it CANNOT be
   validated (leave-one-bank-out removes every positive). It is shipped so the website works end to
   end and so it can be retrained properly when more labelled events are added. Treat its output as
   illustrative until then; `RiskModels.report["early_warning"]["validated"]` says so explicitly.

`assign_risk_tier` combines everything with transparent, hand-written rules (not learned).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ANOMALY_FEATURES = [
    "z_em", "x1_liquidity", "x2_reserves_ta", "x3_profit_ta", "x4_mcap_liab",
    "m_score", "dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata",
    "gnpa", "nnpa", "crar", "roa", "equity_ta", "d_gnpa_pp",
]
EARLY_WARNING_FEATURES = ["z_em", "m_score", "gnpa", "nnpa", "crar", "roa", "d_gnpa_pp"]


# --------------------------------------------------------------------------- labels
def load_events(path) -> pd.DataFrame:
    ev = pd.read_csv(path)
    ev.columns = [c.strip().lower() for c in ev.columns]
    return ev


def _fy_num(s: pd.Series) -> pd.Series:
    from .scoring import fy_sort_key
    return fy_sort_key(s)


def make_labels(scored: pd.DataFrame, events: Optional[pd.DataFrame], horizon: int = 1) -> pd.Series:
    """1 if a *distress* event for that bank happens in [fy, fy + horizon]; else 0.

    Banks with no listed event are assumed healthy over the sample (an assumption: say so in any write-up).
    """
    y = pd.Series(0, index=scored.index, dtype=int)
    if events is None or events.empty:
        return y
    ev = events[events["event_type"].str.lower().eq("distress")].copy()
    ev["fy_num"] = _fy_num(ev["fy"])
    for _, e in ev.iterrows():
        m = (scored["bank"] == e["bank"]) & (scored["fy_num"] >= e["fy_num"] - horizon) & (scored["fy_num"] <= e["fy_num"])
        y[m] = 1
    return y


def flag_structural_breaks(scored: pd.DataFrame, events: Optional[pd.DataFrame]) -> pd.Series:
    out = pd.Series(False, index=scored.index)
    if events is None or events.empty:
        return out
    br = events[events["event_type"].str.lower().eq("merger")].copy()
    br["fy_num"] = _fy_num(br["fy"])
    for _, e in br.iterrows():
        out |= (scored["bank"] == e["bank"]) & (scored["fy_num"] == e["fy_num"])
    return out


# --------------------------------------------------------------------------- models
@dataclass
class RiskModels:
    anomaly: Pipeline
    train_anomaly_scores: np.ndarray
    early_warning: Optional[Pipeline]
    report: Dict
    anomaly_features: list
    ew_features: list

    # ---- fitting
    @classmethod
    def fit(cls, scored: pd.DataFrame, events: Optional[pd.DataFrame] = None,
            horizon: int = 1, random_state: int = 42) -> "RiskModels":
        X = _frame(scored, ANOMALY_FEATURES)
        anomaly = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler()),
            ("forest", IsolationForest(n_estimators=500, contamination="auto", random_state=random_state)),
        ])
        anomaly.fit(X)
        train_scores = -anomaly.score_samples(X)

        y = make_labels(scored, events, horizon)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        ew = None
        ew_info = {"trained": False, "validated": False, "n_positive_rows": n_pos, "n_negative_rows": n_neg,
                   "horizon_years": horizon,
                   "note": "Not trained: need at least 2 positive rows and 1 negative row."}
        if n_pos >= 2 and n_neg >= 1:
            ew = Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, random_state=random_state)),
            ])
            ew.fit(_frame(scored, EARLY_WARNING_FEATURES), y)
            coefs = dict(zip(EARLY_WARNING_FEATURES, ew.named_steps["clf"].coef_[0].round(4).tolist()))
            n_dist_banks = int(scored.loc[y == 1, "bank"].nunique())
            ew_info = {
                "trained": True,
                "validated": False,
                "n_positive_rows": n_pos, "n_negative_rows": n_neg,
                "n_distressed_banks": n_dist_banks,
                "horizon_years": horizon,
                "standardised_coefficients": coefs,
                "note": ("In-sample only. With %d distressed bank(s) a leave-one-bank-out test removes every "
                         "positive, so no out-of-sample performance can be reported. Add more labelled events "
                         "(data/events.csv) and retrain before relying on this output." % n_dist_banks),
            }
            if n_dist_banks >= 3:
                auc = _lobo_auc(scored, y)
                ew_info["lobo_auc"] = None if auc is None else round(auc, 4)
                if auc is not None:
                    ew_info["validated"] = bool(auc >= 0.70)
                    ew_info["note"] = ("Leave-one-bank-out AUC = %.2f over %d distressed banks. %s"
                                       % (auc, n_dist_banks,
                                          "Validated (AUC >= 0.70): used in the risk score." if auc >= 0.70 else
                                          "Below 0.70, so it is NOT used in the risk score."))
        report = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_rows": int(len(scored)),
            "n_banks": int(scored["bank"].nunique()),
            "anomaly": {"features": ANOMALY_FEATURES, "n_estimators": 500},
            "early_warning": ew_info,
        }
        return cls(anomaly, train_scores, ew, report, ANOMALY_FEATURES, EARLY_WARNING_FEATURES)

    # ---- inference
    def predict(self, scored: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        X = _frame(scored, self.anomaly_features)
        raw = -self.anomaly.score_samples(X)
        pct = np.searchsorted(np.sort(self.train_anomaly_scores), raw, side="right") / len(self.train_anomaly_scores) * 100

        out = scored.copy()
        out["anomaly_score"] = raw
        out["anomaly_percentile"] = np.round(pct, 1)
        out["feature_coverage"] = X.notna().mean(axis=1).round(2)
        if self.early_warning is not None:
            out["early_warning_prob"] = self.early_warning.predict_proba(_frame(scored, self.ew_features))[:, 1].round(4)
        else:
            out["early_warning_prob"] = np.nan
        out["structural_break"] = flag_structural_breaks(out, events)

        ew_ok = bool(self.report.get("early_warning", {}).get("validated", False))

        out["risk_tier"] = assign_risk_tier(out, ew_validated=ew_ok)

        out = pd.concat(
            [out, investment_risk_score(out, ew_validated=ew_ok)],
            axis=1
        )       
        return out

    # ---- persistence
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RiskModels":
        return joblib.load(path)


def _frame(df: pd.DataFrame, cols) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan for c in cols},
                        index=df.index)


# --------------------------------------------------------------------------- transparent rules
def assign_risk_tier(p: pd.DataFrame,ew_validated:bool = True) -> pd.Series:
    """Hand-written, auditable rules (NOT learned):

    High   : Z + M both flag, OR early-warning prob >= 0.50, OR anomaly percentile >= 95
    Watch  : Z'' zone = Distress, OR M flag at either cut-off, OR anomaly percentile >= 85, OR prob >= 0.25
    Low    : otherwise
    Insufficient data : fewer than 60% of model features available
    A structural break (merger year) caps the tier at Watch, because the M indices are not comparable
    across a merger and would otherwise read as manipulation.
    """
    if ew_validated:
        ew = p["early_warning_prob"].fillna(0)
    else:
        ew = pd.Series(0.0, index=p.index)
    high = p["joint_signal"].eq("Z + M both flag") | (ew >= 0.5) | (p["anomaly_percentile"] >= 95)
    watch = (p["z_zone"].eq("Distress") | p["m_flag_a"].eq("Flag") | p["m_flag_b"].eq("Flag")
             | (p["anomaly_percentile"] >= 85) | (ew >= 0.25))
    tier = np.where(high, "High", np.where(watch, "Watch", "Low"))
    tier = pd.Series(tier, index=p.index)
    if "structural_break" in p.columns:
        tier = tier.where(~(p["structural_break"] & tier.eq("High")), "Watch")
    tier = tier.where(p["feature_coverage"] >= 0.6, "Insufficient data")
    return tier


# --------------------------------------------------------------------------- investor-facing score
RISK_WEIGHTS = {"z": 0.35, "m": 0.20, "anom": 0.25, "ew": 0.20}
DRIVER_LABELS = {"z": "Altman Z'' (financial weakness)", "m": "Beneish M (earnings-quality / manipulation signal)",
                 "anom": "Unusual pattern vs. peer banks (anomaly model)", "ew": "Early-warning model"}


def investment_risk_score(p: pd.DataFrame, ew_validated: bool = False, weights: Optional[dict] = None) -> pd.DataFrame:
    """0-100 composite, higher = riskier.  Weights are a JUDGEMENT CALL (state this in any write-up).

    z    : Z''-EM, >= 5.85 (safe) -> 0 ; 4.35 (distress line) -> 0.5 ; keeps rising to 1 at Z'' = 1.35,
           so very weak banks score higher than marginal ones. Z'' zones over-trigger on banks, hence 0.5 not 1.
    m    : Beneish M mapped linearly, <= -2.22 -> 0 ; >= -1.78 -> 1
    anom : anomaly percentile / 100
    ew   : early-warning probability. Its weight is forced to 0 until the model is validated
           (RiskModels.report['early_warning']['validated']), because with few distress events it is in-sample only.
    Components that cannot be computed are dropped and the remaining weights are re-normalised.
    A merger year (structural_break) is capped at 'Medium'; < 60% feature coverage -> 'Insufficient data'.
    """
    w = dict(weights or RISK_WEIGHTS)
    if not ew_validated:
        w["ew"] = 0.0
    wt = pd.Series(w)
    parts = pd.DataFrame(index=p.index)
    z = pd.to_numeric(p["z_em"], errors="coerce")
    parts["z"] = np.where(z >= 5.85, 0.0, np.where(z >= 4.35, (5.85 - z) / (5.85 - 4.35) * 0.5,
                                                    0.5 + ((4.35 - z) / (4.35 - 1.35)).clip(0, 1) * 0.5))
    parts.loc[z.isna(), "z"] = np.nan
    parts["m"] = ((pd.to_numeric(p["m_score"], errors="coerce") + 2.22) / (-1.78 + 2.22)).clip(0, 1)
    parts["anom"] = (pd.to_numeric(p["anomaly_percentile"], errors="coerce") / 100).clip(0, 1)
    parts["ew"] = pd.to_numeric(p["early_warning_prob"], errors="coerce")
    contrib = parts.fillna(0) * wt
    avail = parts.notna() * wt
    denom = avail.sum(axis=1).replace(0, np.nan)
    score = (contrib.sum(axis=1) / denom * 100).round(1)

    band = pd.Series(np.where(score >= 66, "High", np.where(score >= 33, "Medium", "Low")), index=p.index, dtype=object)
    band[score.isna()] = "Insufficient data"
    if "structural_break" in p.columns:
        band = band.where(~(p["structural_break"].astype(bool) & band.eq("High")), "Medium")
    if "feature_coverage" in p.columns:
        band = band.where(p["feature_coverage"] >= 0.6, "Insufficient data")
    main = contrib.idxmax(axis=1).map(DRIVER_LABELS)
    main[score.isna() | (contrib.sum(axis=1) == 0)] = None
    out = pd.DataFrame({"risk_score": score, "risk_band": band, "main_driver": main})
    for k in ["z", "m", "anom", "ew"]:
        out[f"risk_part_{k}"] = (parts[k] * 100).round(1)
    return out


def _lobo_auc(scored: pd.DataFrame, y: pd.Series) -> Optional[float]:
    """Leave-one-bank-out AUC of the early-warning model (pooled out-of-sample predictions)."""
    preds, ys = [], []
    for bank in scored["bank"].unique():
        te = scored["bank"] == bank
        tr = ~te
        if y[tr].sum() < 2 or (y[tr] == 0).sum() < 1:
            return None
        m = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scale", StandardScaler()),
                      ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, random_state=42))])
        m.fit(_frame(scored[tr], EARLY_WARNING_FEATURES), y[tr])
        preds.append(m.predict_proba(_frame(scored[te], EARLY_WARNING_FEATURES))[:, 1])
        ys.append(y[te].to_numpy())
    yy, pp = np.concatenate(ys), np.concatenate(preds)
    return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None


# --------------------------------------------------------------------------- validation vs real events
def event_validation(pred: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """TP/FP/FN/TN + sensitivity/specificity of each signal against the event labels, plus AUC.

    Rows where a signal is not computable (e.g. FY2018 has no M-score) are excluded for that signal only.
    """
    y = labels.reindex(pred.index)
    # tier recomputed WITHOUT the (in-sample trained) early-warning model, so this row is not circular
    # Tier without early-warning, for the rules + anomaly validation rows
    tier_unsup = assign_risk_tier(pred, ew_validated=False)

    # Separate row that deliberately includes early-warning,
    # even when it is in-sample and not validated
    tier_with_ew = assign_risk_tier(pred, ew_validated=True)

    signals = {
        "Z'' zone = Distress": pred["z_zone"].eq("Distress"),
        "M flag (cut-off -1.78)": pred["m_flag_a"].eq("Flag"),
        "M flag (cut-off -2.22)": pred["m_flag_b"].eq("Flag"),
        "Z + M both flag": pred["joint_signal"].eq("Z + M both flag"),
        "Anomaly pct >= 90": pred["anomaly_percentile"] >= 90,
        "Tier High (rules + anomaly only)": tier_unsup.eq("High"),
        "Tier High or Watch (rules + anomaly only)": tier_unsup.isin(["High", "Watch"]),
        "Tier High (incl. in-sample early-warning)": tier_with_ew.eq("High"),
    }
    computable = {
        "Z'' zone = Distress": pred["z_zone"].ne("n/a"),
        "M flag (cut-off -1.78)": pred["m_flag_a"].ne("n/a"),
        "M flag (cut-off -2.22)": pred["m_flag_b"].ne("n/a"),
        "Z + M both flag": pred["joint_signal"].ne("n/a"),
        "Anomaly pct >= 90": pd.Series(True, index=pred.index),
        "Tier High (rules + anomaly only)": tier_unsup.ne("Insufficient data"),
        "Tier High or Watch (rules + anomaly only)": tier_unsup.ne("Insufficient data"),
        "Tier High (incl. in-sample early-warning)": pred["risk_tier"].ne("Insufficient data"),
    }
    rows = []
    for name, flag in signals.items():
        ok = computable[name]
        yy, ff = y[ok], flag[ok]
        tp = int(((ff) & (yy == 1)).sum()); fp = int(((ff) & (yy == 0)).sum())
        fn = int(((~ff) & (yy == 1)).sum()); tn = int(((~ff) & (yy == 0)).sum())
        rows.append({"signal": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
                     "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                     "n_scored": int(ok.sum())})
    out = pd.DataFrame(rows)

    # threshold-free ranking quality (higher = riskier)
    auc = {}
    for name, score in {"-Z''-EM": -pred["z_em"], "M-score": pred["m_score"],
                        "Anomaly score": pred["anomaly_score"]}.items():
        ok = score.notna() & y.notna()
        auc[name] = roc_auc_score(y[ok], score[ok]) if y[ok].nunique() == 2 else np.nan
    out.attrs["auc"] = auc
    return out