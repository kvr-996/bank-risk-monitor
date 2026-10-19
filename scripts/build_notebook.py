"""Builds notebooks/bank_risk_model.ipynb (run once; then execute the notebook in Jupyter)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md = lambda t: nb.cells.append(nbf.v4.new_markdown_cell(t))
code = lambda t: nb.cells.append(nbf.v4.new_code_cell(t))

md("""# Bank distress & manipulation screener: Altman Z'' + Beneish M + ML layer

**What this notebook does**

1. Loads the six-bank panel exported from your research workbook (`data/bank_panel.csv`).
2. Computes **Altman Z''** (bank-adapted, emerging-market) and **Beneish M** (bank-mapped) with the exact
   formulas, switches and cut-offs from the workbook's `Parameters` tab, and checks them against the workbook's `Ratios` tab.
3. Fits the **ML layer**: an unsupervised Isolation-Forest anomaly model and a small supervised early-warning model.
4. Validates every signal against the Yes Bank event (RBI moratorium, 5 Mar 2020).
5. Shows how the website back-end scores a new bank.

**Important:** Z'' and M are closed-form formulas, so they are computed *exactly*, not learned. A learned model would only approximate them.
The ML sits on top and answers a different question: *does this bank-year look unusual, and does it resemble the run-up to a known distress event?*

**Data caveat:** a large share of the inputs in the workbook are mechanical estimates (see `n_estimated_fields`, `Estimate_Log`), and FY2025-26 values are projections.
Results show the method working; they are not audited findings.""")

code("""import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from bank_risk.scoring import Params, score_panel
from bank_risk.ml import RiskModels, load_events, make_labels, event_validation
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

raw = pd.read_csv(ROOT / "data" / "bank_panel.csv")
events = load_events(ROOT / "data" / "events.csv")
print(raw.shape, "| banks:", raw.bank.unique().tolist())
raw[["bank","fy","total_assets","net_profit","market_cap","gnpa","crar","n_estimated_fields"]].head(10)""")

md("## 1. Exact scores: Altman Z'' and Beneish M")
code("""p = Params()          # workbook defaults: X1 weight 6.56, +3.25 EM constant, cut-offs 5.85 / 4.35, M cut-off -1.78
scored = score_panel(raw, p)
cols = ["bank","fy","x1_liquidity","x2_reserves_ta","x3_profit_ta","x4_mcap_liab","z_em","z_zone","m_score","m_flag_a","joint_signal"]
scored[cols].round(3).head(18)""")

md("### Parity check against the workbook's own `Ratios` tab")
code("""wb = pd.read_csv(ROOT/"data"/"workbook_ratios.csv", keep_default_na=False, na_values=[""]).rename(columns={"Bank":"bank","FY":"fy"})
m = scored.merge(wb, on=["bank","fy"])
pairs = {"Z''-EM (+3.25)":"z_em", "Beneish M-score":"m_score", "DSRI":"dsri", "GMI":"gmi", "AQI":"aqi",
         "SGI":"sgi", "DEPI":"depi", "SGAI":"sgai", "LVGI":"lvgi", "TATA":"tata"}
rows = []
for w, c in pairs.items():
    a, b = pd.to_numeric(m[c], errors="coerce"), pd.to_numeric(m[w], errors="coerce")
    ok = a.notna() & b.notna()
    rows.append({"metric": w, "rows compared": int(ok.sum()), "max abs diff": float((a[ok]-b[ok]).abs().max())})
pd.DataFrame(rows)""")

md("## 2. What the scores say")
code("""fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for bank, g in scored.groupby("bank"):
    ax[0].plot(g.fy, g.z_em, marker="o", label=bank, lw=2.5 if bank=="Yes Bank" else 1.2)
    ax[1].plot(g.fy, g.m_score, marker="o", label=bank, lw=2.5 if bank=="Yes Bank" else 1.2)
ax[0].axhline(p.z_safe, ls="--", c="g"); ax[0].axhline(p.z_distress, ls="--", c="r")
ax[0].set_title("Altman Z''-EM (dashed: safe 5.85 / distress 4.35)")
ax[1].axhline(p.m_cutoff_a, ls="--", c="r"); ax[1].axhline(p.m_cutoff_b, ls=":", c="orange")
ax[1].set_title("Beneish M (dashed: -1.78, dotted: -2.22)")
for a in ax: a.tick_params(axis="x", rotation=45)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()""")
code("""print("Z'' zones:", scored.z_zone.value_counts().to_dict())
print("M flags @ -1.78:", scored.m_flag_a.value_counts().to_dict(), "| @ -2.22:", scored.m_flag_b.value_counts().to_dict())
print("\\nLowest Z''-EM bank-years:"); print(scored.nsmallest(5, "z_em")[["bank","fy","z_em","z_zone","gnpa","crar"]].round(3).to_string(index=False))
print("\\nHighest M bank-years:");   print(scored.nlargest(5, "m_score")[["bank","fy","m_score","m_flag_b"]].round(3).to_string(index=False))""")

md("""**Reading this honestly**

* Z''-EM puts Yes Bank FY2020 (the moratorium year) far below every other bank-year, but it also labels *Axis* and *IDFC First* as "Distress" in some years: the zone cut-offs are not calibrated for banks, the *ranking* is more informative than the label.
* Beneish M never crosses -1.78 for any bank-year. The -2.22 cut-off flags some rows, mostly false positives. M was built for non-financial firms, so on banks it is a weak signal.
* Merger years (HDFC Bank FY2024, IDFC First FY2019) distort the M indices; they are tagged as structural breaks.""")

md("## 3. ML layer")
code("""models = RiskModels.fit(scored, events, horizon=1)
pred = models.predict(scored, events)
labels = make_labels(scored, events, horizon=1)
print("labelled-positive rows:", pred.loc[labels==1, ["bank","fy"]].values.tolist())
print("\\nearly-warning model:", models.report["early_warning"]["note"])
pred[["bank","fy","z_em","m_score","anomaly_percentile","early_warning_prob","structural_break","risk_tier"]].round(3).query("bank in ['Yes Bank','IDFC First Bank']")""")
code("""piv = pred.pivot(index="bank", columns="fy", values="anomaly_percentile")
fig, ax = plt.subplots(figsize=(10, 3.2))
im = ax.imshow(piv.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45)
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, "" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, label="anomaly percentile"); ax.set_title("Isolation-Forest anomaly percentile (no labels used)"); plt.tight_layout(); plt.show()""")

md("## 4. Validation against the Yes Bank event")
code("""val = event_validation(pred, labels)
print("AUC (higher = riskier):", {k: round(v, 3) for k, v in val.attrs["auc"].items()})
val.round(3)""")
md("""Only **2 labelled rows** (Yes Bank FY2019 and FY2020) exist, so sensitivity/specificity are indicative at best.
The row "Tier High (incl. in-sample early-warning)" is *circular*: that model was trained on those same two rows. Use the "rules + anomaly only" rows for an honest read.""")

md("## 5. Sensitivity to the bank-adaptation switches")
code("""rows = []
for x3 in (1, 2):
    for recv in (1, 2):
        for cut in (1, 2):
            q = Params(x3_definition=x3, receivables_proxy=recv, joint_cutoff=cut)
            s = score_panel(raw, q)
            yb = s[(s.bank=="Yes Bank") & (s.fy=="FY2020")].iloc[0]
            rows.append({"X3 def": x3, "recv proxy": recv, "M cut-off": cut,
                         "Yes FY20 Z''-EM": round(yb.z_em, 2), "Yes FY20 M": None if pd.isna(yb.m_score) else round(yb.m_score, 2),
                         "rows Z distress": int((s.z_zone=="Distress").sum()),
                         "rows joint Z+M": int((s.joint_signal=="Z + M both flag").sum())})
pd.DataFrame(rows)""")
md("`receivables_proxy = 1` needs interest-receivable data that is not loaded for any bank, so those rows show blank M-scores. That is the model refusing to guess.")

md("## 6. Scoring a new bank (what the website will do)")
code("""from bank_risk.schema import REQUIRED_MINIMUM
new = raw[(raw.bank=="Yes Bank") & raw.fy.isin(["FY2019","FY2020","FY2021"])][REQUIRED_MINIMUM]   # minimal form fields only
q = Params(neutral_fill=True)
out = models.predict(score_panel(new, q), events)
out[["bank","fy","z_em","z_zone","m_score","anomaly_percentile","risk_tier","input_notes"]].round(3)""")
md("""With only the minimum columns, several inputs are *derived* (listed in `input_notes`) and missing Beneish indices become neutral, so the M-score is approximate.
The API warns about this. The full column set gives results identical to section 1.""")

md("## 7. Save the trained model")
code("""from bank_risk.pipeline import MODEL_PATH
MODEL_PATH.parent.mkdir(exist_ok=True); models.save(MODEL_PATH)
print("saved ->", MODEL_PATH)""")

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "bank_risk_model.ipynb"
nbf.write(nb, out); print("wrote", out)
