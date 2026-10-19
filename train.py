"""Train + score + validate.   python train.py [--panel data/bank_panel.csv] [--events data/events.csv] [--horizon 1]"""
import argparse

import pandas as pd

from bank_risk.pipeline import DATA, train_and_save

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DATA / "bank_panel.csv"))
    ap.add_argument("--events", default=str(DATA / "events.csv"))
    ap.add_argument("--horizon", type=int, default=1, help="label = distress event within [t, t+horizon] FY")
    a = ap.parse_args()
    models, pred, labels, val = train_and_save(a.panel, a.events, a.horizon)
    pd.set_option("display.width", 200)
    print("\nEvent validation (rows labelled 1 = within horizon of a distress event):")
    print(val.round(3).to_string(index=False))
    print("\nAUC (higher score = riskier):", {k: round(v, 3) for k, v in val.attrs["auc"].items()})
    print("\nEarly-warning model:", models.report["early_warning"]["note"])
    print("\nRisk tier counts:\n", pred["risk_tier"].value_counts().to_string())
