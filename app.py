"""Streamlit front-end for the bank distress & manipulation screener.

Place this file in the project root (next to api.py and train.py) and keep
.streamlit/config.toml next to it (it forces a light theme so every widget,
table and chart is readable), then run:
    streamlit run app.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bank_risk.pipeline import (DATA, OUTPUTS, assess_company, read_table, score_frame, train_and_save)
from bank_risk.schema import INVESTOR_OPTIONAL, INVESTOR_REQUIRED, PERCENT_FIELDS, REQUIRED_MINIMUM
from bank_risk.scoring import Params

st.set_page_config(page_title="Bank Risk Monitor", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- palette
NAVY, BLUE, INK, MUTED, LINE = "#0f2a4a", "#1e4fa3", "#1b2a41", "#53647c", "#d9e1ec"
BAND = {  # fg = text, bg = soft tint, solid = strong accent
    "Low": dict(fg="#05603a", bg="#d1fadf", solid="#12915b"),
    "Medium": dict(fg="#7a2e0e", bg="#fef0c7", solid="#dc6803"),
    "High": dict(fg="#912018", bg="#fee4e2", solid="#d92d20"),
    "Insufficient data": dict(fg="#344054", bg="#eaecf0", solid="#667085"),
}
BAND["Watch"] = BAND["Medium"]
CELL_BAND = {"Safe": "Low", "Grey": "Medium", "Distress": "High", "OK": "Low", "Flag": "High"}
PALETTE = ["#1e4fa3", "#0e8f83", "#c2410c", "#7c3aed", "#b42318", "#475467", "#0891b2", "#a16207"]
BAND_MSG = {
    "Low": "Few warning signs in the submitted financials.",
    "Medium": "Some warning signs. Review the drivers below before relying on this company.",
    "High": "Several strong warning signs. Treat with caution and investigate the drivers below.",
    "Insufficient data": "Not enough usable data to produce a reliable score. Fill in the missing fields.",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root {{ --navy:{NAVY}; --blue:{BLUE}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; color-scheme: light; }}

/* ---------- base: always light, always readable ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background:#f4f7fb !important; color:var(--ink) !important;
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {{
  font-family:'DM Sans',-apple-system,'Segoe UI',Roboto,sans-serif;
}}
.stApp h1,.stApp h2,.stApp h3,.stApp h4 {{ font-family:'Manrope','DM Sans',sans-serif; color:var(--navy) !important; letter-spacing:-.02em; }}
.stApp h3 {{ font-size:1.15rem; font-weight:800; margin:1.2rem 0 .6rem; }}
.block-container {{ padding:1.2rem 2rem 3rem; max-width:1400px; }}
header[data-testid="stHeader"] {{ background:transparent; }}

[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, label p {{ color:var(--ink) !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:var(--muted) !important; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background:#ffffff !important; border-right:1px solid var(--line); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p {{ color:var(--ink) !important; }}
[data-testid="stSidebar"] h3 {{ margin-top:0; }}

/* ---------- tabs ---------- */
[data-testid="stTabs"] [role="tablist"] {{ gap:.35rem; border-bottom:2px solid var(--line); }}
[data-testid="stTabs"] button[role="tab"] {{ padding:.7rem 1.15rem; font-weight:700; font-size:.95rem; color:var(--muted) !important; }}
[data-testid="stTabs"] button[role="tab"] p {{ color:inherit !important; font-weight:700; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color:var(--blue) !important; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top:1.2rem; }}

/* ---------- hero ---------- */
.hero {{ background:linear-gradient(115deg,#0f2a4a 0%,#17407a 65%,#1e4fa3 100%); border-radius:18px;
         padding:1.5rem 1.8rem; margin-bottom:.9rem; box-shadow:0 6px 20px rgba(15,42,74,.18); }}
.hero .eyebrow {{ color:#b9d3f7 !important; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700; }}
.hero .title {{ font-family:'Manrope',sans-serif; font-size:1.85rem; font-weight:800; letter-spacing:-.03em; margin:.3rem 0; color:#ffffff !important; }}
.hero .copy {{ color:#e3eefc !important; max-width:880px; font-size:.97rem; line-height:1.5; }}
.notice {{ background:#fffaeb; border:1px solid #fedf89; border-left:5px solid #f79009; color:#6b3f04 !important;
           border-radius:10px; padding:.65rem 1rem; font-size:.87rem; margin-bottom:1.1rem; }}
.notice b {{ color:#6b3f04 !important; }}

/* ---------- cards ---------- */
.kpi {{ background:#ffffff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem; min-height:128px;
        display:flex; flex-direction:column; justify-content:space-between; box-shadow:0 2px 6px rgba(15,42,74,.06); }}
.kpi-l {{ font-size:.73rem; font-weight:700; color:var(--muted) !important; text-transform:uppercase; letter-spacing:.07em; }}
.kpi-v {{ font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; color:var(--navy) !important; line-height:1.1; margin:.2rem 0; }}
.kpi-v small {{ font-size:.9rem; font-weight:600; color:var(--muted) !important; margin-left:.25rem; }}
.kpi-s {{ font-size:.82rem; color:var(--muted) !important; min-height:1.4rem; }}
.pill {{ display:inline-block; padding:2px 11px; border-radius:99px; font-weight:700; font-size:.78rem; }}
.verdict {{ border-radius:12px; padding:1rem 1.2rem; margin:.4rem 0 1rem; display:flex; flex-wrap:wrap; gap:.4rem 1rem; align-items:center; }}
.verdict b {{ font-family:'Manrope',sans-serif; font-size:1.2rem; }}
.meter {{ background:#ffffff; border:1px solid var(--line); border-radius:14px; padding:.95rem 1.25rem 1.05rem; margin:1rem 0;
          box-shadow:0 2px 6px rgba(15,42,74,.06); }}
.meter-top {{ display:flex; justify-content:space-between; font-size:.8rem; font-weight:700; color:var(--muted) !important;
              text-transform:uppercase; letter-spacing:.07em; margin-bottom:.8rem; }}
.meter-wrap {{ position:relative; }}
.meter-segs {{ display:flex; height:12px; border-radius:99px; overflow:hidden; }}
.meter-marker {{ position:absolute; top:-6px; width:6px; height:24px; background:var(--navy); border:2px solid #fff;
                 border-radius:4px; transform:translateX(-50%); box-shadow:0 1px 4px rgba(0,0,0,.4); }}
.meter-scale {{ display:flex; font-size:.75rem; color:var(--muted) !important; margin-top:.45rem; }}
.step {{ font-family:'Manrope',sans-serif; font-weight:800; color:var(--navy) !important; margin:.2rem 0 .15rem; font-size:1.05rem; }}
.step span {{ background:var(--blue); color:#ffffff !important; border-radius:99px; padding:2px 10px; margin-right:.55rem; font-size:.8rem; }}
.hint {{ color:var(--muted) !important; font-size:.87rem; margin-bottom:.6rem; line-height:1.45; }}
.empty {{ background:#ffffff; border:2px dashed #b9c7da; border-radius:14px; padding:1.6rem; text-align:center;
          color:var(--muted) !important; margin-top:.6rem; }}
.empty b {{ color:var(--navy) !important; font-family:'Manrope',sans-serif; font-size:1.05rem; }}

/* ---------- containers, expanders, tables ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:14px; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ border-color:var(--line); }}
[data-testid="stExpander"] {{ border-radius:12px; background:#ffffff !important; border:1px solid var(--line) !important; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color:var(--ink) !important; font-weight:600; }}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#ffffff; }}
[data-testid="stFileUploaderDropzone"] {{ background:#ffffff !important; border:2px dashed #9fb3cf !important; border-radius:12px; }}
[data-testid="stFileUploaderDropzone"] * {{ color:var(--ink) !important; }}

/* ---------- inputs ---------- */
.stApp input, .stApp textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
  background:#ffffff !important; color:#172b4d !important; border-color:#a9b9cf !important;
}}
.stApp input::placeholder {{ color:#6b7a90 !important; opacity:1 !important; }}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"] {{ background:#ffffff !important; color:#172b4d !important; }}
[role="option"] * {{ color:#172b4d !important; }}
[data-baseweb="tag"] {{ background:#dbe7f8 !important; }}
[data-baseweb="tag"] * {{ color:#0f2a4a !important; }}

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button {{
  border-radius:10px; min-height:2.7rem; font-weight:700; color:#174ea6 !important; background:#ffffff !important; border:1.5px solid #8ea6c6 !important;
}}
.stButton>button p, .stDownloadButton>button p {{ color:inherit !important; font-weight:700; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:#eaf1fb !important; border-color:var(--blue) !important; color:var(--navy) !important; }}
.stButton>button[kind="primary"] {{ background:var(--blue) !important; color:#ffffff !important; border-color:var(--blue) !important; font-size:1.02rem; }}
.stButton>button[kind="primary"] p {{ color:#ffffff !important; }}
.stButton>button[kind="primary"]:hover {{ background:#163d80 !important; color:#ffffff !important; }}

[data-testid="stAlert"] {{ border-radius:10px; }}
@media(max-width:768px) {{ .block-container{{padding:1rem;}} .hero .title{{font-size:1.35rem;}} }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def pill(text: str, band: str) -> str:
    s = BAND.get(band, BAND["Insufficient data"])
    return f'<span class="pill" style="background:{s["bg"]};color:{s["fg"]}">{html.escape(str(text))}</span>'


def kpi(label: str, value: str, sub: str = "", accent: str = BLUE) -> str:
    return (f'<div class="kpi" style="border-top:4px solid {accent}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


def risk_meter(score: float) -> str:
    pos = max(0.0, min(100.0, score))
    return (f'<div class="meter"><div class="meter-top"><span>Risk meter</span><span>{score:.0f} / 100</span></div>'
            f'<div class="meter-wrap"><div class="meter-segs">'
            f'<div style="width:33%;background:{BAND["Low"]["solid"]}"></div>'
            f'<div style="width:33%;background:{BAND["Medium"]["solid"]}"></div>'
            f'<div style="width:34%;background:{BAND["High"]["solid"]}"></div></div>'
            f'<div class="meter-marker" style="left:{pos}%"></div></div>'
            f'<div class="meter-scale"><span style="width:33%">Low (0-33)</span><span style="width:33%">Medium (33-66)</span>'
            f'<span style="width:34%;text-align:right">High (66-100)</span></div></div>')


def cell_style(v):
    key = CELL_BAND.get(v, v)
    if key in BAND:
        s = BAND[key]
        return f"background-color:{s['bg']};color:{s['fg']};font-weight:600"
    return ""


def show_table(df: pd.DataFrame, cols: list[str]):
    cols = [c for c in cols if c in df.columns]
    sub = [c for c in ("risk_tier", "risk_band", "z_zone", "m_flag_a", "m_flag_b") if c in cols]
    styled = df[cols].style.map(cell_style, subset=sub).format(precision=2, na_rep="–")
    cfg = {}
    if "risk_score" in cols:
        cfg["risk_score"] = st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f")
    if "anomaly_percentile" in cols:
        cfg["anomaly_percentile"] = st.column_config.ProgressColumn("Anomaly pctl", min_value=0, max_value=100, format="%.0f")
    st.dataframe(styled, width="stretch", hide_index=True, column_config=cfg)


def band_cards(df: pd.DataFrame):
    cols = st.columns(4)
    total = max(len(df), 1)
    for col, t in zip(cols, ["High", "Medium", "Low", "Insufficient data"]):
        n = int((df["risk_band"] == t).sum())
        col.markdown(kpi(t, str(n), f"{n / total:.0%} of bank-years", BAND[t]["solid"]), unsafe_allow_html=True)


def style_chart(ch):
    return (ch.configure(background="#ffffff", padding={"left": 8, "right": 12, "top": 10, "bottom": 6})
              .configure_axis(labelColor=INK, titleColor=INK, gridColor="#e6ecf4", domainColor="#b9c7da", tickColor="#b9c7da",
                              titleFontWeight=600, labelFontSize=12, titleFontSize=12)
              .configure_legend(labelColor=INK, orient="bottom", title=None, labelFontSize=12)
              .configure_view(strokeWidth=0, fill="#ffffff"))


def show_chart(ch):
    """Render a chart inside a bordered white card with our own (non-Streamlit) theme."""
    with st.container(border=True):
        st.altair_chart(ch, width="stretch", theme=None)


def _bank_color():
    return alt.Color("bank:N", scale=alt.Scale(range=PALETTE), legend=alt.Legend(title=None, orient="bottom"))


def _prep(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.dropna(subset=[col]).copy()
    d["fy_label"] = d["fy"].astype(str)
    return d


def _ref_line(y: float, color: str, label: str):
    data = pd.DataFrame({"y": [y], "t": [label]})
    rule = alt.Chart(data).mark_rule(strokeDash=[5, 4], color=color, strokeWidth=1.6).encode(y="y:Q")
    txt = alt.Chart(data).mark_text(align="left", baseline="bottom", dx=2, dy=-4, color=color, fontSize=11,
                                    fontWeight=700).encode(y="y:Q", text="t:N", x=alt.value(6))
    return rule + txt


def risk_chart(df: pd.DataFrame, height=300):
    d = _prep(df, "risk_score")
    bands = pd.DataFrame({"lo": [0, 33, 66], "hi": [33, 66, 100], "c": ["#e1f6e9", "#fff1c9", "#fde3e1"]})
    bg = alt.Chart(bands).mark_rect().encode(y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 100])), y2="hi:Q", color=alt.Color("c:N", scale=None))
    ln = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2.4).encode(
        x=alt.X("fy_label:N", title="Fiscal year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("risk_score:Q", title="Risk score (0-100)", scale=alt.Scale(domain=[0, 100])), color=_bank_color(),
        tooltip=["bank", "fy_label", alt.Tooltip("risk_score:Q", format=".0f"), "risk_band"])
    return style_chart((bg + ln).properties(height=height))


def z_chart(df: pd.DataFrame, height=300):
    d = _prep(df, "z_em")
    ln = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2.2).encode(
        x=alt.X("fy_label:N", title="Fiscal year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("z_em:Q", title="Altman Z''-EM", scale=alt.Scale(zero=False)), color=_bank_color(),
        tooltip=["bank", "fy_label", alt.Tooltip("z_em:Q", format=".2f"), "z_zone"])
    return style_chart((ln + _ref_line(5.85, BAND["Low"]["solid"], "Safe > 5.85")
                        + _ref_line(4.35, BAND["High"]["solid"], "Distress < 4.35")).properties(height=height))


def m_chart(df: pd.DataFrame, cutoff: float, height=300):
    d = _prep(df, "m_score")
    ln = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2.2).encode(
        x=alt.X("fy_label:N", title="Fiscal year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("m_score:Q", title="Beneish M-score", scale=alt.Scale(zero=False)), color=_bank_color(),
        tooltip=["bank", "fy_label", alt.Tooltip("m_score:Q", format=".2f"), "m_flag_a"])
    return style_chart((ln + _ref_line(cutoff, BAND["High"]["solid"], f"Flag above {cutoff}")).properties(height=height))


def parts_chart(parts: dict):
    names = {"z": "Altman Z''", "m": "Beneish M", "anom": "Anomaly model", "ew": "Early-warning"}
    d = pd.DataFrame({"Component": [names[k] for k in parts], "Risk": list(parts.values())})
    d["label"] = d["Risk"].map(lambda v: f"{v:.0f}")
    base = alt.Chart(d).encode(y=alt.Y("Component:N", sort=None, title=None))
    bars = base.mark_bar(cornerRadiusEnd=5, size=22).encode(
        x=alt.X("Risk:Q", scale=alt.Scale(domain=[0, 100]), title="Component risk (0 = none, 100 = maximum)"),
        color=alt.Color("Risk:Q", legend=None, scale=alt.Scale(type="threshold", domain=[33, 66],
                                                                range=[BAND["Low"]["solid"], BAND["Medium"]["solid"], BAND["High"]["solid"]])),
        tooltip=["Component", alt.Tooltip("Risk:Q", format=".0f")])
    txt = base.mark_text(align="left", dx=6, color=INK, fontWeight=700).encode(x="Risk:Q", text="label:N")
    return style_chart((bars + txt).properties(height=max(46 * len(d), 130)))


@st.cache_data(show_spinner="Scoring bundled panel...")
def get_demo_panel() -> pd.DataFrame:
    path = OUTPUTS / "scores_and_predictions.csv"
    if not path.exists() or "risk_score" not in pd.read_csv(path, nrows=1).columns:
        train_and_save(verbose=False)       # outputs were built before the risk score existed
    return pd.read_csv(path)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("### ⚙️ Analysis settings")
st.sidebar.caption("Assumptions used by **Assess a bank** and **Bulk analysis**.")
x3 = st.sidebar.radio("Altman X3 definition", [1, 2], format_func=lambda v: {1: "PBT / total assets", 2: "(PBT + interest) / total assets"}[v])
recv = st.sidebar.radio("Beneish receivables proxy", [2, 1], format_func=lambda v: {1: "Interest receivable", 2: "Net / gross advances"}[v])
jc = st.sidebar.radio("Beneish cut-off", [1, 2], format_func=lambda v: {1: "-1.78 (standard)", 2: "-2.22 (stricter)"}[v])
tax = st.sidebar.slider("Assumed tax rate", 0.0, 0.5, 0.25, 0.01, help="Only used when profit before tax is not entered.")
neutral = st.sidebar.checkbox("Neutral-fill missing Beneish indices", value=False, help="Treat missing indices as 1.0.")
params = Params(x3_definition=x3, receivables_proxy=recv, joint_cutoff=jc, assumed_tax_rate=tax, neutral_fill=neutral)
m_cut = params.m_cutoff() if callable(params.m_cutoff) else params.m_cutoff

# ----------------------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <div class="eyebrow">Financial intelligence · Risk analytics</div>
  <div class="title">Bank Risk &amp; Integrity Monitor</div>
  <div class="copy">Screen a bank's financials for distress signals (Altman Z''), earnings-quality red flags (Beneish M) and unusual patterns (ML).</div>
</div>
<div class="notice"><b>Research use only.</b> Bundled figures are partly estimated (FY2025-26 are projections) and the early-warning model
is not validated (one distressed bank in the panel). Outputs are screening signals, not audited findings or investment advice.</div>
""", unsafe_allow_html=True)

tab_assess, tab_demo, tab_batch, tab_val, tab_about = st.tabs(
    ["🏦 Assess a bank", "📊 Portfolio overview", "📁 Bulk analysis", "✅ Model validation", "📘 Methodology"])

# ============================================================================= Assess a bank
LABELS = {
    "total_assets": "Total assets", "total_liabilities": "Total liabilities", "equity": "Equity (net worth)",
    "cash": "Cash & bank balances", "investments": "Investments", "deposits": "Deposits",
    "total_income": "Total income", "net_profit": "Net profit", "operating_cash_flow": "Operating cash flow",
    "market_cap": "Market capitalisation", "depreciation": "Depreciation", "fixed_assets": "Fixed assets",
    "interest_expense": "Interest expense", "operating_expenses": "Operating expenses",
    "gnpa": "GNPA (%)", "nnpa": "NNPA (%)", "crar": "Capital adequacy / CRAR (%)",
    "gross_advances": "Gross advances", "net_advances": "Net advances", "interest_income": "Interest income",
    "other_income": "Fee / other income", "net_interest_income": "Net interest income",
    "equity_capital": "Equity share capital", "reserves": "Reserves & surplus", "pbt": "Profit before tax",
    "borrowings": "Borrowings", "other_liabilities": "Other liabilities", "earning_assets": "Earning assets",
    "interest_receivable": "Interest receivable", "cet1": "CET1 ratio (%)", "pcr": "Provision coverage (%)",
}
REQ_FIELDS = [c for c in INVESTOR_REQUIRED if c not in ("bank", "fy")]
GROUPS = {
    "Balance sheet": ["total_assets", "total_liabilities", "equity", "cash", "investments", "deposits", "fixed_assets"],
    "Income & cash flow": ["total_income", "interest_expense", "operating_expenses", "depreciation", "net_profit", "operating_cash_flow"],
    "Market & asset quality": ["market_cap", "gnpa", "nnpa", "crar"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(REQ_FIELDS), "GROUPS must cover INVESTOR_REQUIRED"
_half = (len(INVESTOR_OPTIONAL) + 1) // 2
OPT_GROUPS = {"Balance sheet & P&L (optional)": INVESTOR_OPTIONAL[:_half], "More detail (optional)": INVESTOR_OPTIONAL[_half:]}


def _example_table(fields):
    raw = pd.read_csv(DATA / "bank_panel.csv")
    ex = raw[(raw["bank"] == "Yes Bank") & raw["fy"].isin(["FY2019", "FY2020"])].sort_values("fy")
    t = pd.DataFrame({"Previous FY": [ex.iloc[0].get(f) for f in fields], "Current FY": [ex.iloc[1].get(f) for f in fields]},
                     index=[LABELS[f] for f in fields], dtype=float)
    for f in fields:
        if f in PERCENT_FIELDS:
            t.loc[LABELS[f]] *= 100
    return t


def _blank_table(fields):
    return pd.DataFrame({"Previous FY": np.nan, "Current FY": np.nan}, index=[LABELS[f] for f in fields], dtype=float)


def _to_frame(company, fy_prev, fy_cur, req_tbl, opt_tbl):
    rows = []
    for col, fy in (("Previous FY", fy_prev), ("Current FY", fy_cur)):
        r = {"bank": company, "fy": fy}
        for tbl, fields in ((req_tbl, REQ_FIELDS), (opt_tbl, INVESTOR_OPTIONAL)):
            for f in fields:
                v = tbl.loc[LABELS[f], col]
                r[f] = (v / 100 if f in PERCENT_FIELDS else v) if pd.notna(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _reset_form(example: bool):
    ss = st.session_state
    ss.req_tbl = _example_table(REQ_FIELDS) if example else _blank_table(REQ_FIELDS)
    ss.opt_tbl = _example_table(INVESTOR_OPTIONAL) if example else _blank_table(INVESTOR_OPTIONAL)
    ss.company, ss.fy_prev, ss.fy_cur = ("Yes Bank", "FY2019", "FY2020") if example else ("", "FY2023", "FY2024")
    ss.tbl_ver = ss.get("tbl_ver", 0) + 1
    ss.assess_out = None


def _editors(master: pd.DataFrame, groups: dict, prefix: str, fy_prev: str, fy_cur: str) -> pd.DataFrame:
    cfg = {"Previous FY": st.column_config.NumberColumn(f"Previous ({fy_prev or '-'})", format="%.2f", width="small"),
           "Current FY": st.column_config.NumberColumn(f"Current ({fy_cur or '-'})", format="%.2f", width="small")}
    cols = st.columns(len(groups), gap="medium")
    out = []
    for col, (title, fields) in zip(cols, groups.items()):
        with col:
            st.markdown(f"**{title}**")
            out.append(st.data_editor(master.loc[[LABELS[f] for f in fields]], column_config=cfg, width="stretch",
                                      key=f"{prefix}_{title}_{st.session_state.tbl_ver}"))
    return pd.concat(out)


def render_assessment(res: dict):
    band = res["risk_band"]
    s = BAND.get(band, BAND["Insufficient data"])
    score = res["risk_score"]
    st.markdown(f"### Result · {html.escape(res['company'])} · {html.escape(str(res['fy']))}")
    head = f"{band} risk" if band != "Insufficient data" else "Insufficient data"
    st.markdown(f'<div class="verdict" style="background:{s["bg"]};border-left:6px solid {s["solid"]}">'
                f'<b style="color:{s["fg"]}">{head}</b><span style="color:{s["fg"]}">{BAND_MSG.get(band, "")}</span></div>',
                unsafe_allow_html=True)

    zb = CELL_BAND.get(res["z_zone"], "Insufficient data")
    mflag = res["m_flag"]
    mb = CELL_BAND.get(mflag, "Insufficient data")
    k = st.columns(4)
    k[0].markdown(kpi("Risk score", "n/a" if score is None else f"{score:.0f}<small>/ 100</small>", pill(band, band), s["solid"]), unsafe_allow_html=True)
    k[1].markdown(kpi("Altman Z''-EM", "n/a" if res["z_em"] is None else f"{res['z_em']:.2f}",
                      pill(f"{res['z_zone']} zone", zb) + " &nbsp;<span style='font-size:.75rem'>Safe &gt; 5.85</span>", BAND[zb]["solid"]), unsafe_allow_html=True)
    k[2].markdown(kpi("Beneish M-score", "n/a" if res["m_score"] is None else f"{res['m_score']:.2f}",
                      pill("Flagged" if mflag == "Flag" else "No flag" if mflag == "OK" else "n/a", mb)
                      + f" &nbsp;<span style='font-size:.75rem'>cut-off {m_cut}</span>", BAND[mb]["solid"]), unsafe_allow_html=True)
    ap = res["anomaly_percentile"]
    k[3].markdown(kpi("Anomaly percentile", "n/a" if ap is None else f"{ap:.0f}<small>th</small>",
                      "More unusual than this % of training bank-years", BLUE), unsafe_allow_html=True)

    if score is not None:
        st.markdown(risk_meter(score), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Why this score**")
            if res["main_driver"]:
                st.markdown(f"Biggest contributor: **{res['main_driver']}**")
            for r in res["reasons"]:
                st.markdown(f"- {r}")
    with right:
        with st.container(border=True):
            st.markdown("**Score breakdown**")
            parts = {k_: v for k_, v in res["score_parts"].items() if v is not None}
            if parts:
                st.altair_chart(parts_chart(parts), width="stretch", theme=None)
            else:
                st.caption("No components could be computed.")

    important = [w for w in res["warnings"] if "early-warning" not in w]
    with st.expander(f"Data checks & notes ({len(res['warnings'])})", expanded=bool(important)):
        for w in res["warnings"]:
            (st.warning if w in important else st.caption)(w)
        if not res["warnings"]:
            st.caption("No issues found.")
    with st.expander("Component ratios and assumptions"):
        comp = pd.DataFrame({"value": res["components"]}).rename_axis("component")
        st.dataframe(comp.style.format(precision=3, na_rep="–"), width="stretch")
        if res["input_notes"]:
            st.caption("Derived / assumed: " + res["input_notes"])


with tab_assess:
    ss = st.session_state
    if "req_tbl" not in ss:
        _reset_form(example=False)

    # ---- step 1
    st.markdown('<div class="step"><span>1</span>Company and fiscal years</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">Use the <b>latest two</b> consecutive years, same units in both (e.g. Rs crore). '
                'Beneish M compares the current year with the previous one.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 3], vertical_alignment="bottom")
        company = c1.text_input("Company name", key="company", placeholder="e.g. Example Bank Ltd")
        fy_prev = c2.text_input("Previous fiscal year", key="fy_prev")
        fy_cur = c3.text_input("Current fiscal year", key="fy_cur")
        with c4:
            b1, b2 = st.columns(2)
            b1.button("Load example", width="stretch", help="Yes Bank FY2019-FY2020", on_click=_reset_form, args=(True,))
            b2.button("Clear form", width="stretch", on_click=_reset_form, args=(False,))

    # ---- step 2
    st.markdown('<div class="step" style="margin-top:1.3rem"><span>2</span>Financials</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">Type or paste into the tables. Percent fields are typed as percents (5 for 5%). '
                'All 17 core fields are required for both years.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        req_tbl = _editors(ss.req_tbl, GROUPS, "req", fy_prev, fy_cur)
    with st.expander("Additional financials (optional, improves accuracy)"):
        st.caption("Net advances matters most: without it the Beneish receivables proxy is weaker. Blank fields are derived where possible.")
        opt_tbl = _editors(ss.opt_tbl, OPT_GROUPS, "opt", fy_prev, fy_cur)

    # ---- step 3
    st.markdown('<div class="step" style="margin-top:1.3rem"><span>3</span>Run</div>', unsafe_allow_html=True)
    if st.button("Run risk assessment", type="primary", width="stretch"):
        if not company.strip():
            ss.assess_out = {"errors": ["Enter a company name."]}
        else:
            try:
                ss.assess_out = {"res": assess_company(_to_frame(company.strip(), fy_prev, fy_cur, req_tbl, opt_tbl), params)}
            except ValueError as e:
                ss.assess_out = {"errors": list(e.args[0])}

    out = ss.get("assess_out")
    st.divider()
    if not out:
        st.markdown('<div class="empty"><b>Your result will appear here</b><br>'
                    'Fill in the tables above (or click <i>Load example</i>) and press <i>Run risk assessment</i>.</div>',
                    unsafe_allow_html=True)
    elif "errors" in out:
        for msg in out["errors"]:
            st.error(msg)
    else:
        render_assessment(out["res"])
    st.caption("Screening aid only, not investment advice. Scores combine published formulas with a small ML layer trained on a handful of Indian banks.")

# ============================================================================= Portfolio overview
with tab_demo:
    demo = get_demo_panel()
    banks = sorted(demo["bank"].unique())
    sel = st.multiselect("Banks to show", banks, default=banks)
    d = demo[demo["bank"].isin(sel)]
    if d.empty:
        st.info("Select at least one bank.")
    else:
        band_cards(d)
        st.markdown("### Risk score over time")
        show_chart(risk_chart(d))
        st.caption("Shaded bands: Low 0-33, Medium 33-66, High 66-100.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("### Altman Z''-EM")
            show_chart(z_chart(d))
        with c2:
            st.markdown("### Beneish M-score")
            show_chart(m_chart(d, -1.78))
        st.markdown("### Bank-level detail")
        show_table(d, ["bank", "fy", "risk_score", "risk_band", "z_em", "z_zone", "m_score", "m_flag_a", "joint_signal",
                       "anomaly_percentile", "structural_break", "n_estimated_fields"])
        st.download_button("⬇ Download scores (CSV)", d.to_csv(index=False), "demo_scores.csv", "text/csv")

# ============================================================================= Bulk analysis
RESULT_COLS = ["bank", "fy", "risk_score", "risk_band", "z_em", "z_zone", "m_score", "m_flag_a", "m_flag_b", "joint_signal",
               "anomaly_percentile", "early_warning_prob", "risk_tier", "structural_break", "x1_liquidity", "x2_reserves_ta",
               "x3_profit_ta", "x4_mcap_liab", "dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata",
               "boyd_graham_z", "roa", "nim", "feature_coverage", "input_notes"]
with tab_batch:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Upload a portfolio")
        st.markdown('<div class="hint">CSV or XLSX with <b>at least 2 consecutive years per bank</b> (3 for Boyd-Graham). '
                    'Column names can be snake_case or the original workbook headers.</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xlsm", "xls"], label_visibility="collapsed")
        use_demo = st.checkbox("Use the bundled panel instead (data/bank_panel.csv)", value=up is None)
    with right:
        st.markdown("### Template and columns")
        template = pd.read_csv(DATA / "bank_panel.csv")
        template = template[(template["bank"] == "Yes Bank") & template["fy"].isin(["FY2019", "FY2020"])].drop(columns=["n_estimated_fields"], errors="ignore")
        st.download_button("⬇ Download CSV template", template.to_csv(index=False), "template.csv", "text/csv", width="stretch")
        with st.expander("Required columns"):
            st.write("**Minimum:**", ", ".join(REQUIRED_MINIMUM))
            st.write("**For a reliable M-score and ML features:**", ", ".join(c for c in INVESTOR_REQUIRED if c not in REQUIRED_MINIMUM))
            st.caption("gnpa / nnpa / crar are fractions (0.05 = 5%). FY format: FY2019, 2019 or 2018-19.")

    raw = None
    if up is not None:
        try:
            raw = read_table(up, up.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read file: {e}")
    elif use_demo:
        raw = pd.read_csv(DATA / "bank_panel.csv")

    if raw is not None:
        st.divider()
        with st.expander(f"Input preview ({len(raw)} rows)"):
            st.dataframe(raw.head(50), width="stretch")
        try:
            pred, warns = score_frame(raw, params, strict=False)
        except ValueError as e:
            for msg in e.args[0]:
                st.error(msg)
        else:
            if warns:
                with st.expander(f"Data checks & notes ({len(warns)})"):
                    for w in warns:
                        st.caption("• " + w)
            band_cards(pred)
            b = st.selectbox("Focus on one bank", ["All banks"] + sorted(pred["bank"].astype(str).unique()))
            view = pred if b == "All banks" else pred[pred["bank"].astype(str) == b]
            st.markdown("### Risk score over time")
            show_chart(risk_chart(view, 260))
            c1, c2 = st.columns(2, gap="large")
            with c1:
                show_chart(z_chart(view, 260))
            with c2:
                show_chart(m_chart(view, m_cut, 260))
            st.markdown("### Results")
            show_table(view, RESULT_COLS)
            st.download_button("⬇ Download results (CSV)", view[[c for c in RESULT_COLS if c in view.columns]].to_csv(index=False),
                               "scored_results.csv", "text/csv")

# ============================================================================= Model validation
with tab_val:
    v, r = OUTPUTS / "event_validation.csv", OUTPUTS / "training_report.json"
    if not (v.exists() and r.exists()):
        train_and_save(verbose=False)
    report = json.loads(r.read_text())
    ew = report["early_warning"]
    st.markdown("### How well do the signals catch the Yes Bank FY2020 event?")
    auc = report.get("validation_auc", {})
    cols = st.columns(max(len(auc), 1))
    for col, (name, val) in zip(cols, auc.items()):
        col.markdown(kpi(f"AUC · {name}", "n/a" if val is None else f"{val:.2f}", "1.0 = perfect ranking, 0.5 = chance", BLUE), unsafe_allow_html=True)
    st.markdown("")
    st.dataframe(pd.read_csv(v).style.format(precision=3, na_rep="–"), width="stretch", hide_index=True)

    st.markdown("### Early-warning model")
    (st.success if ew.get("validated") else st.warning)(ew["note"])
    coefs = ew.get("standardised_coefficients")
    if coefs:
        cd = pd.DataFrame({"feature": list(coefs), "coefficient": list(coefs.values())})
        ch = alt.Chart(cd).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("coefficient:Q", title="Standardised coefficient (+ raises risk)"),
            color=alt.condition("datum.coefficient > 0", alt.value(BAND["High"]["solid"]), alt.value(BAND["Low"]["solid"])),
            tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f")]).properties(height=260)
        show_chart(style_chart(ch))
    if st.button("Retrain models from data/"):
        train_and_save(verbose=False)
        st.cache_data.clear()
        st.success("Retrained. Reload the page to see updated outputs.")

# ============================================================================= Methodology
with tab_about:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("### How the risk score is built")
        st.dataframe(pd.DataFrame({
            "Component": ["Altman Z''-EM", "Beneish M-score", "Anomaly model", "Early-warning model"],
            "Weight": ["35%", "20%", "25%", "20% (off until validated)"],
            "What it measures": ["Financial strength: Safe > 5.85, Distress < 4.35", "Earnings-quality / manipulation signal, flag above -1.78",
                                 "How unusual the financial pattern is vs. training banks (Isolation Forest)", "Probability of a distress event within a year"]}),
            width="stretch", hide_index=True)
        st.markdown("Weights are a judgement call, not learned. **Low** < 33 ≤ **Medium** < 66 ≤ **High**. Merger years are capped at Medium "
                    "because Beneish indices are not comparable across a merger.")
    with c2:
        with st.container(border=True):
            st.markdown("### Limitations")
            st.markdown("1. Much of the input data is estimated.\n2. One distressed bank, so the early-warning model cannot be validated.\n"
                        "3. Z'' over-triggers on banks; Beneish M is weak on banks.\n4. Merger years are tagged `structural_break`.")