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
from contextlib import nullcontext
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st





ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bank_risk.pipeline import (DATA, OUTPUTS, assess_company, explain_row, read_table, score_frame, train_and_save)
from bank_risk.schema import INVESTOR_OPTIONAL, INVESTOR_REQUIRED, PERCENT_FIELDS, REQUIRED_MINIMUM
from bank_risk.scoring import Params

st.set_page_config(page_title="Bank Risk Monitor", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

_cfg = ROOT / ".streamlit" / "config.toml"
if not _cfg.exists():          # takes effect on the next restart
    try:
        _cfg.parent.mkdir(exist_ok=True)
        _cfg.write_text('[theme]\nbase = "light"\nprimaryColor = "#1e4fa3"\nbackgroundColor = "#f4f7fb"\n'
                        'secondaryBackgroundColor = "#e9eff7"\ntextColor = "#1b2a41"\n', encoding="utf-8")
    except OSError:
        pass
st.markdown("""
<style>
/* Modern investor-facing typography */

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --dashboard-font: 'IBM Plex Sans', sans-serif;
}

/* Main application text — do not override icon fonts */
.stApp,
.stApp p,
.stApp label,
.stApp input,
.stApp textarea,
.stApp button,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"] {
    font-family: var(--dashboard-font) !important;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3 {
    font-family: var(--dashboard-font) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    color: #102A43;
}

/* KPI card labels */
.kpi-label {
    font-family: var(--dashboard-font) !important;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.2px;
}

/* KPI numbers */
.kpi-value {
    font-family: var(--dashboard-font) !important;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -1px;
    color: #102A43;
}

/* Preserve Streamlit's built-in icon fonts */
[data-testid="stIconMaterial"],
.material-icons,
.material-symbols-rounded,
.material-symbols-outlined {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}
</style>
""", unsafe_allow_html=True)

try:
    DARK_THEME = st.context.theme.type == "dark"
except Exception:  # noqa: BLE001  (older Streamlit)
    DARK_THEME = False

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
[data-testid="stTabs"] [role="tablist"] {{ gap:.8rem; border-bottom:2px solid var(--line); }}
[data-testid="stTabs"] button[role="tab"] {{ padding:.7rem 1.35rem; font-weight:700; font-size:.95rem; color:var(--muted) !important; }}
[data-testid="stTabs"] button[role="tab"] p {{ color:inherit !important; font-weight:700; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color:var(--blue) !important; }}
[data-testid="stTabs"] [role="tabpanel"] {{ padding-top:1.2rem; }}

/* ---------- hero ---------- */
.hero {{ background:transparent; padding:.5rem 0 1rem; margin-bottom:.9rem; }}
.hero .eyebrow {{ color:#b9d3f7 !important; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700; }}
.hero .title {{ display:inline-block; font-family:'Manrope',sans-serif; font-size:1.85rem; font-weight:800; letter-spacing:-.03em;
                margin:.3rem 0; padding-bottom:.45rem; color:var(--navy) !important; border-bottom:4px solid #1e4fa3; }}
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
.meter-segs {{ display:flex; height:10px; border-radius:99px; overflow:hidden; }}
.meter-wrap {{ position:relative; margin-top:2.1rem; }}
.meter-marker {{ position:absolute; top:-4px; width:18px; height:18px; background:#5b5be6; border:3px solid #fff;
                 border-radius:50%; transform:translateX(-50%); box-shadow:0 2px 8px rgba(91,91,230,.55); }}
.meter-bubble {{ position:absolute; top:-40px; transform:translateX(-50%); background:#fff; border:1px solid #e5e9f0;
                 border-radius:10px; padding:2px 10px; font-weight:700; font-size:.85rem; color:var(--navy) !important;
                 box-shadow:0 6px 18px rgba(15,42,74,.14); white-space:nowrap; }}
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


st.markdown("""
<style>
/* buttons: newer Streamlit uses data-testid="stBaseButton-*" */
.stApp button[data-testid^="stBaseButton"], [data-testid="stFileUploaderDropzone"] button {
  background:#ffffff !important; color:#174ea6 !important; border:1.5px solid #8ea6c6 !important; opacity:1 !important;
}
.stApp button[data-testid^="stBaseButton"] *, [data-testid="stFileUploaderDropzone"] button * { color:#174ea6 !important; }
.stApp button[data-testid^="stBaseButton"]:hover, [data-testid="stFileUploaderDropzone"] button:hover {
  background:#eaf1fb !important; border-color:#1e4fa3 !important;
}
.stApp button[data-testid="stBaseButton-primary"] { background:#1e4fa3 !important; border-color:#1e4fa3 !important; }
.stApp button[data-testid="stBaseButton-primary"] * { color:#ffffff !important; }
.stApp button[data-testid="stBaseButton-primary"]:hover { background:#163d80 !important; }
.stApp button:disabled { opacity:.55 !important; }

/* read-only tables (plain HTML, always light) */
.tbl-wrap { overflow:auto; background:#ffffff; border:1px solid #d9e1ec; border-radius:10px; margin-bottom:.6rem; }
.tbl-wrap table { border-collapse:collapse; width:100%; font-size:.86rem; color:#172b4d; }
.tbl-wrap th { position:sticky; top:0; z-index:1; background:#e9eff7; color:#0f2a4a; text-align:left; padding:.55rem .75rem;
               font-weight:700; white-space:nowrap; border-bottom:2px solid #c5d2e3; }
.tbl-wrap td { padding:.45rem .75rem; border-bottom:1px solid #e6ecf4; color:#172b4d; white-space:nowrap; }
.tbl-wrap tbody tr:nth-child(even) td { background:#f8fafd; }
.tbl-wrap tbody tr:hover td { background:#eef4fc; }
</style>
""", unsafe_allow_html=True)
if DARK_THEME:   # canvas grids ignore CSS colours; flip them back to light if the dark theme is active
    st.markdown("<style>[data-testid='stDataEditor']{filter:invert(1) hue-rotate(180deg);background:#fff;}</style>",
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------- helpers
def pill(text: str, band: str) -> str:
    s = BAND.get(band, BAND["Insufficient data"])
    return f'<span class="pill" style="background:{s["bg"]};color:{s["fg"]}">{html.escape(str(text))}</span>'


def kpi(label: str, value: str, sub: str = "", accent: str = BLUE) -> str:
    return (f'<div class="kpi" style="border-top:4px solid {accent}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


def risk_meter(score: float) -> str:
    pos = max(0.0, min(100.0, score))
    lo, md, hi = BAND["Low"]["solid"], BAND["Medium"]["solid"], BAND["High"]["solid"]
    return (f'<div class="meter"><div class="meter-top"><span>Risk meter</span><span>{score:.0f} / 100</span></div>'
            f'<div class="meter-wrap"><div class="meter-segs" '
            f'style="background:linear-gradient(90deg,{lo} 0%,{lo} 22%,{md} 50%,{hi} 78%,{hi} 100%)"></div>'
            f'<div class="meter-bubble" style="left:{pos}%">{score:.0f}</div>'
            f'<div class="meter-marker" style="left:{pos}%"></div></div>'
            f'<div class="meter-scale"><span style="width:33%">Low (0-33)</span><span style="width:33%">Medium (33-66)</span>'
            f'<span style="width:34%;text-align:right">High (66-100)</span></div></div>')


def cell_style(v):
    key = CELL_BAND.get(v, v)
    if key in BAND:
        s = BAND[key]
        return f"background-color:{s['bg']};color:{s['fg']};font-weight:600"
    return ""


def light_table(df: pd.DataFrame, index: bool = False, precision: int = 2, colour_cols=(), bar_cols=(), max_h: int = 520,
                fmt: dict | None = None, value_styles: dict | None = None):
    """Read-only table rendered as HTML so it is always visible, whatever the Streamlit theme."""
    df = df.copy()
    for year_col in ("fy", "FY"):
        if year_col in df.columns:
            df[year_col] = df[year_col].map(display_fy)
    colour_cols = [c for c in colour_cols if c in df.columns]
    bar_cols = [c for c in bar_cols if c in df.columns]
    sty = df.style.format(precision=precision, na_rep="–", escape="html")
    if fmt:
        sty = sty.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="–", escape="html")
    if colour_cols:
        sty = sty.map(cell_style, subset=colour_cols)
    if value_styles:
        for column, style in value_styles.items():
            if column in df.columns:
                sty = sty.map(style, subset=[column])
    if bar_cols:
        sty = sty.bar(subset=bar_cols, vmin=0, vmax=100, color="#d6d8fb")
    if not index:
        sty = sty.hide(axis="index")
    flat = " ".join(sty.to_html().split())
    st.markdown(f'<div class="tbl-wrap" style="max-height:{max_h}px">{flat}</div>', unsafe_allow_html=True)


def show_table(df: pd.DataFrame, cols: list[str]):
    cols = [c for c in cols if c in df.columns]
    sub = [c for c in ("risk_tier", "risk_band", "z_zone", "m_flag_a", "m_flag_b") if c in cols]
    light_table(df[cols], colour_cols=sub, bar_cols=("risk_score", "anomaly_percentile"))


# --- Score + ratio table: risk score | zone | M score | M ratios | Z score | Z ratios -------------------------
M_RATIOS = {"dsri": "M: DSRI", "gmi": "M: GMI", "aqi": "M: AQI", "sgi": "M: SGI",
            "depi": "M: DEPI", "sgai": "M: SGAI", "lvgi": "M: LVGI", "tata": "M: TATA"}
Z_RATIOS = {"x1_liquidity": "Z: X1 liquidity", "x2_reserves_ta": "Z: X2 reserves/TA",
            "x3_profit_ta": "Z: X3 profit/TA", "x4_mcap_liab": "Z: X4 mcap/liabilities",
            "x5_income_ta": "Z: X5 income/TA"}


def score_ratio_table(df: pd.DataFrame):
    """Cumulative (Z''+M+anomaly) risk score, its zone, then M-score with its 8 ratios, then Z-score with its Z'' ratios (X1-X4), X5 and the classic 5-ratio Z."""
    cols = {"bank": "Bank", "fy": "FY", "risk_score": "Risk score", "risk_band": "Zone", "m_score": "M score",
            **M_RATIOS, "z_em": "Z score", **Z_RATIOS, "z_classic": "Classic Z (5 ratios)"}
    cols = {k: v for k, v in cols.items() if k in df.columns}
    out = df[list(cols)].rename(columns=cols)
    if "FY" in out.columns:
        out["FY"] = out["FY"].map(display_fy)
    fmt = {"Risk score": "{:.0f}", "M score": "{:.3f}", "Z score": "{:.3f}", "Classic Z (5 ratios)": "{:.3f}",
           **{v: "{:.3f}" for v in list(M_RATIOS.values()) + list(Z_RATIOS.values())}}

    def classic_z_style(value):
        if pd.isna(value):
            return ""
        return cell_style("Safe" if value > 2.99 else "Distress" if value < 1.81 else "Grey")

    def m_score_style(value):
        if pd.isna(value):
            return ""
        return cell_style("Flag" if value > m_cut else "OK")

    def z_score_style(value):
        if pd.isna(value):
            return ""
        zone = "Safe" if value > params.z_safe else "Distress" if value < params.z_distress else "Grey"
        return cell_style(zone)

    light_table(out, colour_cols=("Zone",), bar_cols=("Risk score",), fmt=fmt,
                value_styles={"M score": m_score_style, "Z score": z_score_style,
                                          "Classic Z (5 ratios)": classic_z_style})
    st.caption(f"Score colors: M score red = above cutoff ({m_cut}), green = at/below cutoff; "
               "Z score green = safe, amber = grey zone, red = distress. "
               "Z score = Altman Z'' (uses X1-X4 and drives the risk score). X5 is the fifth ratio of the classic Altman Z; "
               "Classic Z (5 ratios) = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5 (safe > 2.99, distress < 1.81), shown for reference.")
    return out


def band_cards(df):
    bands = ["Low", "Medium", "High", "Insufficient data"]
    cols = st.columns(len(bands))
    total = max(len(df), 1)

    for col, t in zip(cols, bands):
        n = int((df["risk_band"] == t).sum())
        col.markdown(
            kpi(t, str(n), "&nbsp;", BAND[t]["solid"]),
            unsafe_allow_html=True
        )

CHART_FONT = "DM Sans, -apple-system, Segoe UI, Roboto, sans-serif"
INDIGO = "#5b5be6"
GRID = "#eceff5"
AXIS_TXT = "#6b7280"
TRACK = "#eef1f8"
CHART_PALETTE = [INDIGO, "#0e8f83", "#e0782c", "#c026d3", "#0891b2", "#b42318", "#64748b", "#a16207"]

# White rounded hover card (Vega tooltips live in the main page DOM, so plain CSS can restyle them)
st.markdown("""
<style>
#vg-tooltip-element { background:#ffffff !important; border:1px solid #e5e9f0 !important; border-radius:12px !important;
  box-shadow:0 8px 24px rgba(15,42,74,.14) !important; color:#1b2a41 !important; padding:10px 14px !important;
  font-family:'DM Sans',-apple-system,'Segoe UI',sans-serif !important; font-size:13px !important; }
#vg-tooltip-element td.key { color:#8a94a6 !important; font-weight:500 !important; padding-right:12px !important; }
#vg-tooltip-element td.value { color:#1b2a41 !important; font-weight:700 !important; text-align:left !important; }
</style>
""", unsafe_allow_html=True)


def style_chart(ch):
    """Shared look: white card, thin solid pale gridlines, no axis frames, light-grey labels."""
    return (ch.configure(background="#ffffff", font=CHART_FONT,
                         padding={"left": 10, "right": 20, "top": 14, "bottom": 8})
              .configure_view(strokeWidth=0, fill="#ffffff")
              .configure_axis(labelColor=AXIS_TXT, titleColor="#9aa3b2", labelFont=CHART_FONT, titleFont=CHART_FONT,
                              labelFontSize=12, titleFontSize=12, titleFontWeight=400, titlePadding=12,
                              labelPadding=8, ticks=False, domain=False, gridColor=GRID, gridWidth=1)
              .configure_axisX(grid=False)
              .configure_axisY(grid=True)
              .configure_legend(labelColor="#374151", labelFont=CHART_FONT, orient="bottom", title=None,
                                labelFontSize=13, symbolType="square", symbolSize=170, padding=6,
                                columnPadding=22, offset=16, labelLimit=260))


def show_chart(ch):
    """Render a chart inside a bordered white card with our own (non-Streamlit) theme."""
    with st.container(border=True):
        st.altair_chart(ch, width="stretch", theme=None)


def _bank_color():
    return alt.Color("bank:N", scale=alt.Scale(range=CHART_PALETTE), legend=alt.Legend(title=None, orient="bottom"))


def _prep(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.dropna(subset=[col]).copy()
    d["fy_label"] = d["fy"].map(display_fy)
    d["fy_axis_label"] = d["fy_label"]
    return d


def _ref_line(y: float, color: str, label: str):
    data = pd.DataFrame({"y": [y], "t": [label]})
    rule = alt.Chart(data).mark_rule(strokeDash=[6, 5], color=color, strokeWidth=1.5, opacity=.8).encode(y="y:Q")
    txt = alt.Chart(data).mark_text(align="left", baseline="bottom", dx=2, dy=-5, color=color, fontSize=11,
                                    fontWeight=700, font=CHART_FONT).encode(y="y:Q", text="t:N", x=alt.value(8))
    return [rule, txt]


def _trend(d: pd.DataFrame, ycol: str, ytitle: str, scale, tooltip, height: int, color=None, solid=INDIGO,
           xfield: str = "fy_axis_label", xsort="ascending", xtitle=None):
    """Smooth thin line + small dots, vertical hover crosshair, enlarged dot on hover.
    With several banks, clicking a legend entry toggles that bank (like the checkboxes in the reference)."""
    multi = color is not None
    xenc = alt.X(f"{xfield}:N", title=xtitle, sort=xsort, axis=alt.Axis(labelAngle=0, labelPadding=10))
    base = alt.Chart(d).encode(x=xenc, y=alt.Y(f"{ycol}:Q", title=ytitle, scale=scale), tooltip=tooltip)

    hover = alt.selection_point(fields=[xfield], nearest=True, on="mouseover", clear="mouseout", empty=False)
    enc = color if multi else alt.value(solid)
    if multi:
        leg = alt.selection_point(fields=["bank"], bind="legend")
        op = alt.condition(leg, alt.value(1), alt.value(.15))
    else:
        leg, op = None, alt.value(1)

    cross = alt.Chart(d).mark_rule(color="#aab4c5", strokeWidth=1).encode(
        x=alt.X(f"{xfield}:N", sort=xsort),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    line = base.mark_line(interpolate="monotone", strokeWidth=2.4, strokeCap="round",
                          strokeJoin="round").encode(color=enc, opacity=op)
    if leg is not None:
        line = line.add_params(leg)
    pts = base.mark_point(filled=True, stroke="#ffffff", strokeWidth=2).encode(
        color=enc, opacity=op, size=alt.condition(hover, alt.value(170), alt.value(60)))
    catcher = base.mark_point(size=600).encode(opacity=alt.value(0)).add_params(hover)
    return [cross, line, pts, catcher]


def risk_chart(df: pd.DataFrame, height=300):
    d = df.copy()
    # Exclude rows without sufficient feature coverage
    if "risk_band" in d.columns:
        d = d[d["risk_band"] != "Insufficient data"].copy()
    d = _prep(d, "risk_score")
    bands = pd.DataFrame({
        "lo": [0, 33, 66], "hi": [33, 66, 100], "mid": [16.5, 49.5, 83],
        "name": ["LOW", "MEDIUM", "HIGH"],
        "c": ["#f1faf5", "#fffaf0", "#fef3f2"],
        "solid": [BAND["Low"]["solid"], BAND["Medium"]["solid"], BAND["High"]["solid"]]})
    ysc = alt.Scale(domain=[0, 100])
    bg = alt.Chart(bands).mark_rect().encode(
        y=alt.Y("lo:Q", scale=ysc), y2="hi:Q", color=alt.Color("c:N", scale=None, legend=None))
    lab = alt.Chart(bands).mark_text(align="left", fontSize=10, fontWeight=700, opacity=.55, font=CHART_FONT).encode(
        y=alt.Y("mid:Q", scale=ysc), text="name:N", x=alt.value(10), color=alt.Color("solid:N", scale=None, legend=None))
    ln = _trend(d, "risk_score", "Risk score (0-100)", ysc,
                ["bank", "fy_label", alt.Tooltip("risk_score:Q", format=".0f"), "risk_band"], height,
                color=_bank_color())
    return style_chart(alt.layer(bg, lab, *ln).properties(height=height))


def z_chart(df: pd.DataFrame, height=300):
    d = _prep(df, "z_em")
    ln = _trend(d, "z_em", "Altman Z''-EM", alt.Scale(zero=False),
                ["bank", "fy_label", alt.Tooltip("z_em:Q", format=".2f"), "z_zone"], height, color=_bank_color())
    return style_chart(alt.layer(*ln, *_ref_line(5.85, BAND["Low"]["solid"], "Safe > 5.85"),
                                 *_ref_line(4.35, BAND["High"]["solid"], "Distress < 4.35")).properties(height=height))


def m_chart(df: pd.DataFrame, cutoff: float, height=300):
    d = _prep(df, "m_score")
    ln = _trend(d, "m_score", "Beneish M-score", alt.Scale(zero=False),
                ["bank", "fy_label", alt.Tooltip("m_score:Q", format=".2f"), "m_flag_a"], height, color=_bank_color())
    return style_chart(alt.layer(*ln, *_ref_line(cutoff, BAND["High"]["solid"], f"Flag above {cutoff}")).properties(height=height))


def parts_chart(parts: dict):
    names = {"z": "Altman Z''", "m": "Beneish M", "anom": "Anomaly model", "ew": "Early-warning"}
    d = pd.DataFrame({"Component": [names[k] for k in parts], "Risk": list(parts.values())})
    d["label"] = d["Risk"].map(lambda v: f"{v:.0f}")
    d["track"] = 100
    T = "Component risk (0 = none, 100 = maximum)"
    xs = alt.Scale(domain=[0, 100])
    base = alt.Chart(d).encode(y=alt.Y("Component:N", sort=None, title=None,
                                       axis=alt.Axis(grid=False, labelColor=INK, labelFontWeight=600, labelFontSize=13)))
    track = base.mark_bar(cornerRadius=10, size=20, color=TRACK).encode(
        x=alt.X("track:Q", scale=xs, title=T, axis=alt.Axis(grid=True, values=[0, 33, 66, 100])))
    bars = base.mark_bar(cornerRadius=10, size=20).encode(
        x=alt.X("Risk:Q", scale=xs, title=T),
        color=alt.Color("Risk:Q", legend=None, scale=alt.Scale(type="threshold", domain=[33, 66],
                                                                range=[BAND["Low"]["solid"], BAND["Medium"]["solid"], BAND["High"]["solid"]])),
        tooltip=["Component", alt.Tooltip("Risk:Q", format=".0f")])
    txt = base.mark_text(align="left", dx=8, color=INK, fontWeight=700, fontSize=12, font=CHART_FONT).encode(
        x=alt.X("Risk:Q", scale=xs, title=T), text="label:N")
    return style_chart((track + bars + txt).properties(height=max(50 * len(d), 140)))


@st.cache_data(show_spinner="Scoring bundled panel...")
def get_demo_panel() -> pd.DataFrame:
    path = OUTPUTS / "scores_and_predictions.csv"
    if not path.exists() or not {"risk_score", "x5_income_ta"} <= set(pd.read_csv(path, nrows=1).columns):
        train_and_save(verbose=False)       # outputs were built before the risk score existed
    return pd.read_csv(path)


# ============================================================================= Screener-style home + bank pages
SC_BAND = {k: (v["bg"], v["fg"], v["solid"]) for k, v in BAND.items()}

BARS_SVG = ('<svg viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="20" width="7" height="12" rx="2" fill="#6ac11d"/>'
            '<rect x="13" y="12" width="7" height="20" rx="2" fill="#6ac11d"/><rect x="24" y="2" width="7" height="30" rx="2" fill="#6ac11d"/></svg>')


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading scores...")
def sc_load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = OUTPUTS / "scores_and_predictions.csv"
    if not path.exists() or "risk_score" not in pd.read_csv(path, nrows=1).columns:
        train_and_save(verbose=False)
    scored = pd.read_csv(path)
    raw = pd.read_csv(DATA / "bank_panel.csv")
    keep = ["bank", "fy"] + [c for c in raw.columns if c not in scored.columns]
    merged = scored.merge(raw[keep], on=["bank", "fy"], how="left")
    merged = merged.sort_values(["bank", "fy_num"]).reset_index(drop=True)
    return merged, raw


def inr(x, decimals: int = 0) -> str:
    """Indian digit grouping: 1234567 -> 12,34,567."""
    if x is None or pd.isna(x):
        return "–"
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = whole + (("." + frac) if frac else "")
    return ("-" if neg else "") + out


def pct(x, d: int = 2) -> str:
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def display_fy(value) -> str:
    return str(value).strip().removeprefix("FY").removeprefix("fy").strip()


def num(x, d: int = 2) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"


def sc_pill(text: str, band: str | None = None) -> str:
    bg, fg, _ = SC_BAND.get(band or text, SC_BAND["Insufficient data"])
    return f'<span class="sc-pill" style="background:{bg};color:{fg}">{html.escape(str(text))}</span>'


def sc_table(df: pd.DataFrame, fmt: dict | None = None, band_cols=(), highlight=None, index=False, max_h=520):
    """Light HTML table (renders the same under any Streamlit theme)."""
    fmt = fmt or {}
    d = df.copy()
    for year_col in ("fy", "FY"):
        if year_col in d.columns:
            d[year_col] = d[year_col].map(display_fy)
    sty = d.style.format(precision=2, na_rep="–", escape="html")
    if fmt:
        sty = sty.format({k: v for k, v in fmt.items() if k in d.columns}, na_rep="–", escape="html")
    bc = [c for c in band_cols if c in d.columns]
    if bc:
        sty = sty.map(lambda v: (f"background-color:{SC_BAND[v][0]};color:{SC_BAND[v][1]};font-weight:600" if v in SC_BAND else ""), subset=bc)
    if highlight is not None:
        sty = sty.apply(lambda r: ["background-color:#f1eeff;font-weight:600" if highlight(r) else "" for _ in r], axis=1)
    if not index:
        sty = sty.hide(axis="index")
    flat = " ".join(sty.to_html().split())
    st.markdown(f'<div class="sc-tbl" style="max-height:{max_h}px">{flat}</div>', unsafe_allow_html=True)


def sc_css():
    st.markdown(f"""
    <style>
    footer {{ visibility: hidden; }}
    html {{ scroll-behavior:smooth; }}
    .block-container {{ max-width: 1080px; padding-top: 1.2rem; padding-bottom: 4rem; }}

    /* ---------- home ---------- */
    .home-spacer {{ height: 14vh; }}
    .brand {{ text-align: center; font-size: 4.2rem; font-weight: 500; letter-spacing: -2px; line-height: 1; color: #2b2b2b; }}
    .brand svg {{ height: 3.1rem; margin-left: 6px; vertical-align: -2px; }}
    .tagline {{ text-align: center; font-size: 1.12rem; margin: 1.1rem 0 2rem 0; color: {INK}; }}
    .or-label {{ text-align: right; padding-top: .45rem; font-size: 1.02rem; }}

    /* search box (selectbox) */
    div[data-baseweb="select"] > div {{ background: #fff; border: 1px solid #cfd6e2; border-radius: 10px; min-height: 3.5rem;
        font-size: 1.15rem; padding-left: .6rem; }}
    div[data-baseweb="select"] input {{ font-size: 1.15rem; }}
    /* home search box (plain text input, no dropdown arrow) */
    .st-key-search [data-baseweb="input"] > div {{ background:#fff; border:1px solid #cfd6e2; border-radius:10px; min-height:3.5rem; padding-left:.6rem; }}
    .st-key-search input {{ font-size:1.15rem; }}
    /* home search box: no dropdown arrow */
    body:has(.home-spacer) div[data-baseweb="select"] div:has(> svg), body:has(.home-spacer) div[data-baseweb="select"] svg {{ display:none !important; }}
    .st-key-search [data-baseweb="select"] > div > div:last-child,
    .st-key-search [data-baseweb="select"] [role="button"],
    .st-key-search [data-baseweb="select"] [aria-hidden="true"] {{
        display:none !important; width:0 !important; min-width:0 !important; padding:0 !important;
    }}
    .st-key-search [data-baseweb="select"] > div {{ padding-right:0 !important; }}
    .st-key-search [data-baseweb="select"] > div > div:first-child {{ width:100% !important; }}
    .st-key-search [data-baseweb="select"] svg {{ display:none !important; }}
    /* belt and braces: whatever markup Streamlit uses for the arrow, paint a white strip over the right end of the search box
       (clicks pass straight through it, so the box still opens normally) */
    .st-key-search [data-baseweb="select"] > div, body:has(.home-spacer) div[data-baseweb="select"] > div {{
        position:relative !important; overflow:hidden !important; }}
    .st-key-search [data-baseweb="select"] > div::after, body:has(.home-spacer) div[data-baseweb="select"] > div::after {{
        content:""; position:absolute; top:0; right:0; bottom:0; width:3.6rem; background:#ffffff; z-index:50; pointer-events:none; }}

    /* home search box with live suggestions */
    .bs-wrap {{ position:relative; width:100%; font-family:'DM Sans',-apple-system,'Segoe UI',sans-serif; }}
    .bs-input {{ width:100%; box-sizing:border-box; height:3.5rem; padding:0 1.1rem; font-size:1.15rem; color:#1b2a41; background:#fff;
                border:1px solid #cfd6e2; border-radius:10px; outline:none; font-family:inherit;
                transition:border-color .15s ease, box-shadow .15s ease; }}
    .bs-input::placeholder {{ color:#6b7a90; opacity:1; }}
    .bs-input:focus {{ border-color:#1e4fa3; box-shadow:0 0 0 3px rgba(30,79,163,.12); }}
    .bs-list {{ position:absolute; top:calc(100% + 6px); left:0; right:0; z-index:5000; display:none; background:#fff;
               border:1px solid #dfe5ef; border-radius:12px; box-shadow:0 12px 30px rgba(15,42,74,.16); max-height:340px;
               overflow-y:auto; padding:.35rem; }}
    .bs-item {{ padding:.7rem .9rem; border-radius:8px; cursor:pointer; font-size:1.05rem; color:#1b2a41; }}
    .bs-item b {{ color:#1e4fa3; font-weight:700; }}
    .bs-item.is-on, .bs-item:hover {{ background:#eef2fb; }}
    [data-testid="stHtml"], [data-testid="stElementContainer"]:has(.bs-wrap) {{ overflow:visible !important; }}
    .bs-empty {{ padding:.7rem .9rem; color:#6b7a90; font-size:1rem; }}

    /* home search: Streamlit's own box (typing filters the bank list live) inside a white frame; the arrow end of the select is clipped away */
    body:has(.home-spacer) [data-testid="stSelectbox"] {{
        overflow:hidden; background:#f1f3f7 !important; border:1px solid #d5dae3 !important; border-radius:10px;
        transition:background .18s ease, border-color .18s ease, box-shadow .18s ease;
    }}
    body:has(.home-spacer) [data-testid="stSelectbox"] [data-baseweb="select"], .st-key-search [data-baseweb="select"] {{
        clip-path:inset(0 3.6rem 0 0) !important; }}                 /* cuts off the arrow end of the box, whatever its markup */
    .st-key-search {{ overflow:hidden; background:#f1f3f7 !important; border:1px solid #d5dae3 !important; border-radius:10px; }}
    body:has(.home-spacer) [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        border:none !important; background:#f1f3f7 !important; box-shadow:none !important; min-height:3.5rem;
    }}
    body:has(.home-spacer) [data-testid="stSelectbox"]:hover,
    body:has(.home-spacer) [data-testid="stSelectbox"]:focus-within {{
        background:#eef4fc !important; border-color:{BLUE} !important; box-shadow:0 0 0 3px rgba(30,79,163,.12);
    }}
    body:has(.home-spacer) [data-testid="stSelectbox"]:hover [data-baseweb="select"] > div,
    body:has(.home-spacer) [data-testid="stSelectbox"]:focus-within [data-baseweb="select"] > div {{
        background:#eef4fc !important;
    }}

    /* home buttons (bank chips + Tools): grey at rest, colour appears on hover */
    body:has(.home-spacer) .stButton > button,
    body:has(.home-spacer) .stButton > button[data-testid^="stBaseButton"] {{
        background:#f1f3f7 !important; border:1px solid #d5dae3 !important; color:#7a8496 !important; box-shadow:none !important;
        transition:background .18s ease, border-color .18s ease, color .18s ease, box-shadow .18s ease, transform .18s ease;
    }}
    body:has(.home-spacer) .stButton > button *,
    body:has(.home-spacer) .stButton > button[data-testid^="stBaseButton"] * {{ color:#7a8496 !important; }}
    body:has(.home-spacer) .stButton > button:hover,
    body:has(.home-spacer) .stButton > button:focus-visible {{
        background:#eef4fc !important; border-color:{BLUE} !important; color:{BLUE} !important;
        box-shadow:0 4px 14px rgba(30,79,163,.16) !important; transform:translateY(-1px);
    }}
    body:has(.home-spacer) .stButton > button:hover *,
    body:has(.home-spacer) .stButton > button:focus-visible * {{ color:{BLUE} !important; }}

    /* ---------- detail ---------- */
    .topbar {{ display:flex; align-items:center; gap:.4rem; padding:.35rem 0; }}
    .topbar .logo {{ font-size: 1.9rem; font-weight: 500; letter-spacing: -1px; color:#2b2b2b; }}
    .topbar svg {{ height: 1.4rem; }}
    .bank-title {{ font-size: 2.3rem; font-weight: 500; margin: 0; line-height: 1.15; }}
    .bank-sub {{ color: {MUTED}; font-size: .95rem; margin-top: .2rem; }}
    .bank-score-divider {{ border-top:1px solid #dfe5ef; margin:1rem 0 .45rem; }}
    .sc-pill {{ display:inline-block; padding: .2rem .8rem; border-radius: 999px; font-weight: 600; font-size: .9rem; }}
    .price-row {{ display:flex; align-items:baseline; gap:.7rem; margin: .8rem 0 .2rem 0; }}
    .price {{ font-size: 2rem; font-weight: 600; }}
    .delta {{ font-size: .95rem; font-weight: 600; }}
    .sc-cap {{ color: {MUTED}; font-size: .85rem; }}
    .sc-sec {{ font-size: 1.05rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 1.6rem 0 .5rem 0; }}
    .about {{ font-size: 1.05rem; line-height: 1.65; color: #3b4658; }}
    .kp li {{ margin-bottom: .4rem; line-height: 1.55; color: #3b4658; }}
    .sc-card {{ background:#fff; border:1px solid #dfe5ef; border-radius: 12px; padding: .4rem 1.4rem; margin-top: 1rem; }}
    .sc-kv {{ display:flex; justify-content:space-between; padding: .8rem 0; border-bottom: 1px solid #e8ecf3; font-size: 1.02rem; }}
    .sc-kv:last-child {{ border-bottom: none; }}
    .sc-kv span:first-child {{ color: #4a5568; }}
    .sc-kv b {{ font-weight: 600; }}
    .sc-unit {{ color: {MUTED}; font-weight: 400; }}
    .sc-tbl {{ overflow:auto; border:1px solid #dfe5ef; border-radius:10px; background:#fff; }}
    .sc-tbl table {{ border-collapse: collapse; width:100%; font-size:.9rem; }}
    .sc-tbl th {{ background:#f3f6fb; color:#4a5568; font-weight:600; text-align:right; padding:.5rem .7rem; position:sticky; top:0; white-space:nowrap; }}
    .sc-tbl td {{ padding:.45rem .7rem; text-align:right; border-top:1px solid #edf0f6; color:{INK}; white-space:nowrap; }}
    .sc-tbl th:first-child, .sc-tbl td:first-child {{ text-align:left; }}
    .sc-tbl tr.me td {{ background:#f1eeff; font-weight:600; }}

    /* every st.markdown("<style>...") is an empty element; Streamlit's 1rem row-gap between them stacks into a big blank
       strip at the top of the page. Collapse them (the CSS inside still applies). */
    [data-testid="stElementContainer"]:has(style) {{ display:none; }}

    /* ---------- bank header: logo | section tabs | Home | Tools, flush at the very top, sticky, full-width ---------- */
    body:has(.section-nav-wrap) header[data-testid="stHeader"] {{ display:none; }}          /* removes the gap above the bar */
    body:has(.section-nav-wrap) .block-container {{ padding-top:0 !important; }}
    /* sticky must sit on an element whose parent is the tall page block (cover both Streamlit wrapper layouts) */
    [data-testid="stLayoutWrapper"]:has(> .st-key-topbar), .st-key-topbar {{
        position:sticky; top:0; z-index:1000;
    }}
    .st-key-topbar {{
        position:relative; background:#fff; margin-bottom:.4rem;
    }}
    .st-key-topbar::after {{
        content:""; position:absolute; left:0; right:0; bottom:0; height:1px;
        background:#e3e8f0; box-shadow:0 3px 10px rgba(15,42,74,.10); pointer-events:none;
    }}
    .st-key-topbar [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; gap:.9rem !important; align-items:center; }}
    .st-key-topbar [data-testid="stHorizontalBlock"] > div {{ width:auto !important; min-width:0 !important; flex:0 0 auto !important; }}
    .st-key-topbar [data-testid="stHorizontalBlock"] > div:nth-child(2) {{ flex:1 1 0 !important; }}   /* tabs take the space */
    .st-key-topbar .topbar {{ padding:0; white-space:nowrap; }}
    .st-key-topbar .topbar .logo {{ font-size:1.55rem; }}
    .st-key-topbar .topbar svg {{ height:1.2rem; }}
    .st-key-topbar .stButton > button {{ min-height:2.2rem; padding:.2rem .9rem; font-size:.88rem; white-space:nowrap; }}
    .section-nav-wrap {{ width:100%; }}
    .section-nav {{
        display:flex; gap:.25rem; overflow-x:auto; white-space:nowrap;
        scrollbar-width:none;
    }}
    .section-nav::-webkit-scrollbar {{ display:none; }}
    .section-nav a {{
        display:inline-block; padding:1rem 1rem .8rem; border-bottom:3px solid transparent;
        color:#5b6b82; font-size:.95rem; font-weight:600; text-decoration:none;
        transition:color .15s ease,border-color .15s ease;
    }}
    .section-nav a:hover, .section-nav a:focus, .section-nav a.is-active {{
        color:#1e4fa3; border-bottom-color:#1e4fa3;
    }}
    .section-anchor {{ scroll-margin-top:4.5rem; height:1px; }}
    @media(max-width:768px) {{
        .section-nav a {{ padding:.85rem .7rem .65rem; font-size:.88rem; }}
        .st-key-topbar [data-testid="stHorizontalBlock"] > div:nth-child(1) {{ display:none; }}   /* hide logo on phones */
        .st-key-topbar [data-testid="stHorizontalBlock"] {{ gap:.4rem !important; }}
        .st-key-topbar .stButton > button {{ padding:.2rem .5rem; font-size:.8rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)
_BANK_SEARCH_HTML = """
<div class="bs-wrap" data-banks="__BANKS__">
  <input id="bs-input" class="bs-input" type="text" placeholder="Search for a bank" autocomplete="off" spellcheck="false">
  <div id="bs-list" class="bs-list"></div>
</div>
<script>
(() => {
  const S = (window.__bs = window.__bs || { shown: [], on: -1 });
  if (S.off) S.off();                                   /* drop handlers from a previous run */
  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const els = () => {
    const input = document.getElementById("bs-input");
    const list = document.getElementById("bs-list");
    let banks = [];
    try { banks = JSON.parse(input.closest(".bs-wrap").dataset.banks); } catch (err) {}
    return { input, list, banks };
  };
  const go = (name) => {
    const u = new URL(window.location.href);
    u.searchParams.set("bank", name);
    window.location.href = u.toString();
  };
  const paint = (list) => [...list.children].forEach((el, i) => el.classList.toggle("is-on", i === S.on));
  const render = () => {
    const { input, list, banks } = els();
    if (!input || !list) return;
    const q = input.value.trim().toLowerCase();
    if (!q) { list.style.display = "none"; S.shown = []; return; }
    S.shown = banks.filter((b) => b.toLowerCase().includes(q)).slice(0, 8);
    S.on = S.shown.length ? 0 : -1;
    if (!S.shown.length) {
      list.innerHTML = '<div class="bs-empty">No bank found</div>';
    } else {
      list.innerHTML = S.shown.map((b, i) => {
        const k = b.toLowerCase().indexOf(q);
        return '<div class="bs-item" data-i="' + i + '">' + esc(b.slice(0, k)) + "<b>" + esc(b.slice(k, k + q.length)) + "</b>" + esc(b.slice(k + q.length)) + "</div>";
      }).join("");
    }
    list.style.display = "block";
    paint(list);
  };
  const isBox = (e) => e.target && e.target.id === "bs-input";
  const onInput = (e) => { if (isBox(e)) render(); };
  const onFocus = (e) => { if (isBox(e)) render(); };
  const onKey = (e) => {
    if (!isBox(e)) return;
    const { list } = els();
    if (e.key === "ArrowDown" && S.shown.length) { e.preventDefault(); S.on = (S.on + 1) % S.shown.length; paint(list); }
    else if (e.key === "ArrowUp" && S.shown.length) { e.preventDefault(); S.on = (S.on - 1 + S.shown.length) % S.shown.length; paint(list); }
    else if (e.key === "Enter" && S.shown.length) { e.preventDefault(); go(S.shown[Math.max(S.on, 0)]); }
    else if (e.key === "Escape") { list.style.display = "none"; }
  };
  const onDown = (e) => {
    const el = e.target.closest && e.target.closest(".bs-item");
    if (el) { e.preventDefault(); go(S.shown[+el.dataset.i]); }
  };
  const onClick = (e) => {
    if (e.target.closest && e.target.closest(".bs-wrap")) return;
    const list = document.getElementById("bs-list");
    if (list) list.style.display = "none";
  };
  document.addEventListener("input", onInput, true);
  document.addEventListener("focusin", onFocus, true);
  document.addEventListener("keydown", onKey, true);
  document.addEventListener("mousedown", onDown, true);
  document.addEventListener("click", onClick, true);
  S.off = () => {
    document.removeEventListener("input", onInput, true);
    document.removeEventListener("focusin", onFocus, true);
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("mousedown", onDown, true);
    document.removeEventListener("click", onClick, true);
  };
})();
</script>
"""


def bank_search_html(banks) -> str:
    """Home search box with live suggestions (type 'hdfc' -> every bank containing 'hdfc'). No dropdown arrow."""
    data = html.escape(json.dumps([str(b) for b in banks]), quote=True)
    return _BANK_SEARCH_HTML.replace("__BANKS__", data)


def render_home(banks):
    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="brand">bankrisk{BARS_SVG}</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Distress and manipulation screening for banks in India.</div>', unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        st.selectbox("Search", banks, index=None, placeholder="Search for a bank", key="search",
                     label_visibility="collapsed", on_change=on_search, args=(banks,))

    # "Or analyse:" chips (3 per row, label on the first row)
    _, mid, _ = st.columns([0.1, 7.8, 0.1])
    with mid:
        for r0 in range(0, len(banks), 3):
            row = st.columns([1.15, 1, 1, 1, 1.15])
            if r0 == 0:
                row[0].markdown('<div class="or-label">Or analyse:</div>', unsafe_allow_html=True)
            for j, b in enumerate(banks[r0:r0 + 3]):
                row[1 + j].button(b, key=f"chip_{b}", on_click=open_bank, args=(b,), width="stretch")
    st.markdown('')
    _, tools_col, _ = st.columns([2.2, 2, 2.2])
    tools_col.button('⚙ Tools: assess your own data, bulk analysis, validation', key='tools_btn', on_click=open_tools, width='stretch')

def render_bank(bank, data, banks):
    g = data[data["bank"] == bank].sort_values("fy_num").reset_index(drop=True)
    FYS = g["fy"].tolist()

    nav_items = [
        ("section-summary", "Summary"),
        ("section-charts", "Charts"),
        ("section-analysis", "Analysis"),
        ("section-peers", "Peers"),
        ("section-financials", "Financials"),
        ("section-method", "Method"),
    ]
    nav_links = []
    for i, (anchor, label) in enumerate(nav_items):
        active_class = "is-active" if i == 0 else ""
        nav_links.append(f'<a href="#{anchor}" class="{active_class}">{label}</a>')
    nav = "".join(nav_links)
    nav_html = f"""
    <div class="section-nav-wrap"><nav class="section-nav">{nav}</nav></div>
    <script>
    (() => {{
      const scroller = () => document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
      const updateActiveSection = () => {{
        const anchors = [...document.querySelectorAll(".section-anchor")];
        const links = [...document.querySelectorAll(".section-nav a")];
        if (!anchors.length || !links.length) return;

        const threshold = 110;                       /* just below the sticky bar */
        let current = anchors[0];
        for (const anchor of anchors) {{
          if (anchor.getBoundingClientRect().top <= threshold) current = anchor;
          else break;
        }}
        const sc = scroller();                       /* Streamlit scrolls inside stMain, not the window */
        if (sc.scrollHeight > sc.clientHeight + 50 && sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 2) {{
          current = anchors[anchors.length - 1];
        }}
        links.forEach(link => link.classList.toggle("is-active", link.hash === `#${{current.id}}`));
      }};

      if (window.__bankRiskSectionScrollHandler) {{
        document.removeEventListener("scroll", window.__bankRiskSectionScrollHandler, true);
        window.removeEventListener("resize", window.__bankRiskSectionScrollHandler);
      }}
      let scheduled = false;
      window.__bankRiskSectionScrollHandler = () => {{
        if (scheduled) return;
        scheduled = true;
        window.requestAnimationFrame(() => {{
          updateActiveSection();
          scheduled = false;
        }});
      }};
      document.addEventListener("scroll", window.__bankRiskSectionScrollHandler, {{ capture: true, passive: true }});
      window.addEventListener("resize", window.__bankRiskSectionScrollHandler);
      updateActiveSection();
      setTimeout(updateActiveSection, 600);          /* the page is still rendering on first paint */
      setTimeout(updateActiveSection, 1800);
    }})();
    </script>
    """
    # one sticky bar: logo | section tabs | Home | Tools
    with st.container(key="topbar"):
        c_logo, c_nav, c_home, c_tools = st.columns([1.5, 7, 1.2, 1.2], vertical_alignment="center")
        c_logo.markdown(f'<div class="topbar"><span class="logo">bankrisk</span>{BARS_SVG}</div>', unsafe_allow_html=True)
        with c_nav:
            st.html(nav_html, unsafe_allow_javascript=True)
        c_home.button("← Home", key="home_btn", on_click=go_home, width="stretch")
        c_tools.button("⚙ Tools", key="tools_btn2", on_click=open_tools, width="stretch")

    h1, h2 = st.columns([3, 1.1])
    with h2:
        fy = st.selectbox("Financial year", FYS, index=len(FYS) - 1, key="fy_sel", format_func=display_fy)
    cur_i = FYS.index(fy)
    r = g.iloc[cur_i]
    prev = g.iloc[cur_i - 1] if cur_i > 0 else None
    with h1:
        seg = r.get("segment")
        sub = " · ".join(str(x) for x in [seg, r.get("basis")] if isinstance(x, str) and x)
        st.markdown(f'<div class="bank-title">{html.escape(bank)}</div><div class="bank-sub">{html.escape(sub)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="bank-score-divider"></div>', unsafe_allow_html=True)
    tab_sum, tab_chart, tab_an, tab_peer, tab_fin, tab_method = (
        nullcontext() for _ in range(6)
    )

    # ----------------------------------------------------------------------------- Summary
    with tab_sum:
        st.markdown('<div id="section-summary" class="section-anchor"></div>', unsafe_allow_html=True)
        score, band = r["risk_score"], r["risk_band"]
        delta_html = ""
        if prev is not None and pd.notna(score) and pd.notna(prev["risk_score"]):
            d = score - prev["risk_score"]
            col = "#d93025" if d > 0 else "#2e9e5b"       # higher risk = red
            arrow = "▲" if d > 0 else "▼"
            delta_html = f'<span class="delta" style="color:{col}">{arrow} {abs(d):.1f} vs {display_fy(prev["fy"])}</span>'
        st.markdown(
            f'<div class="price-row"><span class="price">{"–" if pd.isna(score) else f"{score:.0f}"}'
            f'<span class="sc-unit" style="font-size:1rem"> / 100</span></span>{sc_pill(band, band)}{delta_html}</div>'
            f'<div class="sc-cap">Cumulative risk score · {display_fy(fy)} · higher = riskier</div>', unsafe_allow_html=True)

        st.markdown('<div class="sc-sec">About</div>', unsafe_allow_html=True)
        ta, np_ = r.get("total_assets"), r.get("net_profit")
        bits = [f"{bank} is a {str(seg).lower() + ' sector ' if isinstance(seg, str) else ''}bank in the research panel."]
        if pd.notna(ta):
            bits.append(f"In {display_fy(fy)} it reported total assets of ₹ {inr(ta)} Cr.")
        if pd.notna(np_):
            bits.append(f"and a net {'profit' if np_ >= 0 else 'loss'} of ₹ {inr(abs(np_))} Cr.")
        st.markdown(f'<div class="about">{" ".join(bits).replace(".  ", ". ")}</div>', unsafe_allow_html=True)
        # if r.get("n_estimated_fields", 0) and r["n_estimated_fields"] > 0:
        #     st.caption(f"⚠️ {int(r['n_estimated_fields'])} input field(s) for {fy} are estimates, not reported figures.")

        st.markdown('<div class="sc-sec">Key points</div>', unsafe_allow_html=True)
        pts = explain_row(r)
        if pd.notna(r.get("main_driver")) and r.get("main_driver"):
            pts = [f"Biggest contributor to the score: <b>{html.escape(str(r['main_driver']))}</b>."] + pts
        st.markdown('<ul class="kp">' + "".join(f"<li>{p}</li>" for p in pts) + "</ul>", unsafe_allow_html=True)

        def kv(label, value, unit=""):
            u = f' <span class="sc-unit">{unit}</span>' if unit else ""
            return f'<div class="sc-kv"><span>{label}</span><b>{value}{u}</b></div>'

        mcap = r.get("market_cap")
        rows = [
            kv("Market Cap", f"₹ {inr(mcap)}", "Cr.") if pd.notna(mcap) else kv("Market Cap", "–"),
            kv("Total Assets", f"₹ {inr(ta)}", "Cr."),
            kv("Deposits", f"₹ {inr(r.get('deposits'))}", "Cr."),
            kv("Net Profit", f"₹ {inr(np_)}", "Cr."),
            kv("Return on Assets", pct(r.get("roa"))),
            kv("Net Interest Margin", pct(r.get("nim"))),
            kv("Gross NPA / Net NPA", f"{pct(r.get('gnpa'))} / {pct(r.get('nnpa'))}"),
            kv("Capital Adequacy (CRAR) / CET1", f"{pct(r.get('crar'))} / {pct(r.get('cet1'))}"),
            kv("Provision Coverage", pct(r.get("pcr"))),
            kv("Overall risk score", ("–" if (pd.isna(r.get("risk_score")) or r["risk_band"] == "Insufficient data") else f"{r['risk_score']:.0f} <span class='sc-unit'>/ 100</span>")
               + " " + sc_pill(r["risk_band"], r["risk_band"])),
            kv("Is it good risk?", sc_pill(
                {"Low": "Yes: low risk", "Medium": "Fair: some warning signs", "High": "No: high risk",
                 "Insufficient data": "Cannot tell: not enough data"}.get(r["risk_band"], "Cannot tell"), r["risk_band"])),
            kv("Altman Z''-EM", f"{num(r.get('z_em'))} " + (sc_pill(r['z_zone'], {'Safe': 'Low', 'Grey': 'Medium', 'Distress': 'High'}.get(r['z_zone'])) if r['z_zone'] in ('Safe', 'Grey', 'Distress') else "")),
            kv("Classic Altman Z (5 ratios)", f"{num(r.get('z_classic'))} {sc_pill(r['z_classic_zone'], {'Safe': 'Low', 'Grey': 'Medium', 'Distress': 'High'}.get(r['z_classic_zone']))}") if 'z_classic' in r.index else "",
            kv("Beneish M-score", f"{num(r.get('m_score'))} <span class='sc-unit'>(flag above -1.78)</span>"),
            kv("Anomaly percentile", "–" if pd.isna(r.get("anomaly_percentile")) else f"{r['anomaly_percentile']:.0f}"),
        ]
        st.markdown(f'<div class="sc-card">{"".join(rows)}</div>', unsafe_allow_html=True)
        if r.get("structural_break"):
            st.info("Merger year: Beneish indices are not comparable across a merger, so the band is capped at Medium.")

    # ----------------------------------------------------------------------------- Charts
    def sc_line_chart(df, col, title, refs=(), domain=None, fmt=".2f"):
        d = df.dropna(subset=[col])
        if d.empty:
            st.caption(f"{title}: not enough data")
            return
        d = d.copy()
        d["fy_axis_label"] = d["fy"].map(display_fy)
        scale = alt.Scale(domain=domain) if domain else alt.Scale(zero=False)
        display_fys = [display_fy(year) for year in FYS]
        layers = _trend(d, col, title, scale, ["fy_axis_label", alt.Tooltip(f"{col}:Q", format=fmt)], 260,
                        xfield="fy_axis_label", xsort=display_fys, xtitle=None)
        for y, color, label in refs:
            layers += _ref_line(y, color, label)
        show_chart(style_chart(alt.layer(*layers).properties(height=260)))


    with tab_chart:
        st.markdown('<div id="section-charts" class="section-anchor"></div>', unsafe_allow_html=True)
        st.markdown(f"#### {bank}")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Cumulative risk score**")
            sc_line_chart(g, "risk_score", "Risk score (0-100)", refs=[(33, "#e0a100", "Medium 33"), (66, "#d93025", "High 66")], domain=[0, 100], fmt=".0f")
        with c2:
            st.markdown("**Altman Z''-EM**")
            sc_line_chart(g, "z_em", "Z''-EM", refs=[(5.85, "#2e9e5b", "Safe > 5.85"), (4.35, "#d93025", "Distress < 4.35")])
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.markdown("**Beneish M-score**")
            sc_line_chart(g, "m_score", "M-score", refs=[(-1.78, "#d93025", "Flag above -1.78")])
        with c4:
            st.markdown("**Gross NPA**")
            gg = g.assign(gnpa_pct=g["gnpa"] * 100)
            sc_line_chart(gg, "gnpa_pct", "Gross NPA (%)")

    # ----------------------------------------------------------------------------- Analysis
    with tab_an:
        st.markdown('<div id="section-analysis" class="section-anchor"></div>', unsafe_allow_html=True)
        st.markdown(f"#### Score breakdown · {display_fy(fy)}")
        parts = pd.DataFrame({
            "Component": ["Altman Z''", "Beneish M", "Anomaly model", "Early-warning"],
            "Risk": [r.get("risk_part_z"), r.get("risk_part_m"), r.get("risk_part_anom"), r.get("risk_part_ew")]})
        parts["Risk"] = pd.to_numeric(parts["Risk"], errors="coerce")
        if parts["Risk"].notna().any():
            _keys = ["z", "m", "anom", "ew"]
            show_chart(parts_chart({k: float(v) for k, v in zip(_keys, parts["Risk"]) if pd.notna(v)}))
            st.caption("The early-warning model is excluded from the cumulative score until it can be validated; it is shown for reference.")
        else:
            st.info("Not enough data to break this year's score into components.")

        st.markdown("#### Cumulative score, zone, M-score and Z-score with their ratios")
        cols = {"fy": "FY", "risk_score": "Cumulative score", "risk_band": "Zone", "m_score": "M score",
                "dsri": "DSRI", "gmi": "GMI", "aqi": "AQI", "sgi": "SGI", "depi": "DEPI", "sgai": "SGAI", "lvgi": "LVGI", "tata": "TATA",
                "z_em": "Z score", "x1_liquidity": "X1 liquidity", "x2_reserves_ta": "X2 reserves/TA",
                "x3_profit_ta": "X3 profit/TA", "x4_mcap_liab": "X4 mcap/liab", "x5_income_ta": "X5 income/TA",
                "z_classic": "Classic Z (5 ratios)"}
        cols = {k: v for k, v in cols.items() if k in g.columns}
        t = g[list(cols)].rename(columns=cols)
        t["FY"] = t["FY"].map(display_fy)
        fmt = {c: "{:.3f}" for c in t.columns if c not in ("FY", "Zone", "Cumulative score")}
        fmt["Cumulative score"] = "{:.0f}"
        sc_table(t, fmt, band_cols=("Zone",), highlight=lambda row: row["FY"] == display_fy(fy))
        st.caption("Highlighted row = selected year. M ratios: receivables, margin, asset quality, income growth, depreciation, SG&A, leverage, accruals. "
                   "Z ratios: X1 liquidity, X2 reserves, X3 profit, X4 market value of equity, X5 income/assets. "
                   "'Z score' is Altman Z'' (X1-X4, drives the risk score); 'Classic Z (5 ratios)' adds X5 and is shown for reference.")

    # ----------------------------------------------------------------------------- Peers
    with tab_peer:
        st.markdown('<div id="section-peers" class="section-anchor"></div>', unsafe_allow_html=True)
        st.markdown("#### Other banks")
        p = data[data["fy"] == fy][["bank", "risk_score", "risk_band", "z_em", "z_zone", "m_score", "gnpa", "crar", "roa"]].copy()
        p = p.sort_values("risk_score", ascending=False, na_position="last")
        for c in ("gnpa", "crar", "roa"):
            p[c] = p[c] * 100
        p = p.rename(columns={"bank": "Bank", "risk_score": "Risk score", "risk_band": "Zone", "z_em": "Z score", "z_zone": "Z zone",
                              "m_score": "M score", "gnpa": "GNPA %", "crar": "CRAR %", "roa": "ROA %"})
        sc_table(p, {"Risk score": "{:.0f}", "Z score": "{:.2f}", "M score": "{:.2f}", "GNPA %": "{:.2f}", "CRAR %": "{:.2f}", "ROA %": "{:.2f}"},
                   band_cols=("Zone",), highlight=lambda row: row["Bank"] == bank)
        st.markdown("**Open another bank**")
        cols_ = st.columns(3)
        for i, b in enumerate([x for x in banks if x != bank]):
            cols_[i % 3].button(b, key=f"peer_{b}", on_click=open_bank, args=(b,), width="stretch")

    # ----------------------------------------------------------------------------- Financials
    with tab_fin:
        st.markdown('<div id="section-financials" class="section-anchor"></div>', unsafe_allow_html=True)
        st.markdown(f"#### {bank} · Financial snapshot")
        st.caption("Figures are from the bank panel; amounts are ₹ crore and percentage changes are year over year. "
                   "Historical estimates and projections may be included.")

        metric_specs = [
            ("Total assets", "total_assets", "amount", False),
            ("Deposits", "deposits", "amount", False),
            ("Net profit", "net_profit", "amount", False),
            ("Operating cash flow", "operating_cash_flow", "amount", False),
            ("Gross NPA", "gnpa", "percent", True),
            ("Capital adequacy · CRAR", "crar", "percent", False),
        ]

        def financial_metric_value(value, kind):
            
            if pd.isna(value):
                return "–"
            return f"{pct(value)}" if kind == "percent" else f"₹ {inr(value)} Cr"

        def financial_metric_delta(current, previous, kind):
            if previous is None or pd.isna(current) or pd.isna(previous):
                return None, None
            if kind == "percent":
                return f"{(current - previous) * 100:+.2f} pp", "inverse"
            if previous == 0:
                return None, None
            return f"{(current / previous - 1) * 100:+.1f}%", "inverse" if current < previous else "normal"

        metric_rows = [st.columns(3, gap="large"), st.columns(3, gap="large")]
        for i, (label, field, kind, inverse) in enumerate(metric_specs):
            value = r.get(field)
            previous_value = prev.get(field) if prev is not None else None
            delta, default_delta_color = financial_metric_delta(value, previous_value, kind)
            metric_rows[i // 3][i % 3].metric(
                label,
                financial_metric_value(value, kind),
                delta,
                delta_color="inverse" if inverse else (default_delta_color or "normal"),
                help=f"Selected year: {display_fy(fy)}" if prev is None else f"Change from {display_fy(prev['fy'])} to {display_fy(fy)}",
            )

        # st.markdown("#### Financial statements and bank ratios")
        # st.caption("Each table lists available metrics by row and the bank's fiscal years in columns.")
        # groups = {
        #     "Balance sheet": ["total_assets", "total_liabilities", "equity", "deposits", "borrowings", "gross_advances", "investments", "cash"],
        #     "Profit & loss": ["interest_income", "other_income", "total_income", "interest_expense", "net_interest_income",
        #                       "operating_expenses", "pbt", "net_profit"],
        #     "Cash flow": ["operating_cash_flow", "depreciation"],
        #     "Asset quality & capital (%)": ["gnpa", "nnpa", "pcr", "crar", "cet1"],
        #     "Market": ["market_cap"],
        # }
        # labels = {"total_assets": "Total assets", "total_liabilities": "Total liabilities", "equity": "Equity", "deposits": "Deposits",
        #           "borrowings": "Borrowings", "gross_advances": "Gross advances", "investments": "Investments", "cash": "Cash & bank balances",
        #           "interest_income": "Interest income", "other_income": "Fee / other income", "total_income": "Total income",
        #           "interest_expense": "Interest expense", "net_interest_income": "Net interest income", "operating_expenses": "Operating expenses",
        #           "pbt": "Profit before tax", "net_profit": "Net profit", "operating_cash_flow": "Operating cash flow",
        #           "depreciation": "Depreciation", "gnpa": "Gross NPA", "nnpa": "Net NPA", "pcr": "Provision coverage", "crar": "CRAR", "cet1": "CET1",
        #           "market_cap": "Market cap"}
        # for title, cs in groups.items():
        #     cs = [c for c in cs if c in g.columns]
        #     if not cs:
        #         continue
        #     st.markdown(f'<div class="sc-sec" style="font-size:.9rem">{title}</div>', unsafe_allow_html=True)
        #     pv = g.set_index("fy")[cs].T
        #     pv.index = [labels[c] for c in cs]
        #     pv.index.name = ""
        #     pv = pv.reset_index().rename(columns={"index": ""})
        #     if "%" in title:
        #         pv = pv.assign(**{y: pv[y] * 100 for y in FYS})
        #         sc_table(pv, {y: "{:.2f}" for y in FYS}, max_h=300)
        #     else:
        #         sc_table(pv, {y: (lambda v: inr(v)) for y in FYS}, max_h=420)

    # ----------------------------------------------------------------------------- Method
    with tab_method:
        st.markdown('<div id="section-method" class="section-anchor"></div>', unsafe_allow_html=True)
        st.markdown("""
    **Cumulative risk score (0-100, higher = riskier)** blends three signals:
    Altman Z''-EM (bank-adapted), Beneish M (bank-mapped) and an unsupervised anomaly model. The early-warning model is excluded until it can be validated.
    Bands: Low 0-33, Medium 33-66, High 66-100.

    **Altman Z''-EM** = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4 + 3.25, with Safe > 5.85 and Distress < 4.35.
    **Five-ratio Z.** Altman Z'' deliberately drops X5 (sales/assets) for non-manufacturers, so the score uses X1-X4. The classic Altman Z (1968) uses all five ratios (1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5; safe > 2.99, distress < 1.81) and is shown for reference; it was built for US manufacturers.

**Beneish M** uses the eight-variable model; a score above -1.78 is flagged.

    **Limits.** Z'' zones over-trigger on banks and M is a weak signal for banks. The panel contains one distressed bank (Yes Bank, FY2020),
    so validation is indicative only. Merger years are capped at Medium. Many inputs are estimates; this screener is a research tool, not investment advice.
    """)


def open_bank(name: str):
    st.session_state["bank"] = name
    st.session_state["view"] = "bank"
    st.session_state["search"] = ""
    st.session_state.pop("search_error", None)
    try:
        st.query_params["bank"] = name
    except Exception:  # noqa: BLE001
        pass


def go_home():
    st.session_state["bank"] = None
    st.session_state["view"] = "home"
    st.session_state["search"] = ""
    st.session_state.pop("search_error", None)
    try:
        st.query_params.clear()
    except Exception:  # noqa: BLE001
        pass


def open_tools():
    st.session_state["view"] = "tools"


def on_search(banks):
    query = str(st.session_state.get("search") or "").strip()
    if not query:
        st.session_state.pop("search_error", None)
        return
    match = next((name for name in banks if name.casefold() == query.casefold()), None)
    if match is None:
        partial = [name for name in banks if query.casefold() in name.casefold()]
        match = partial[0] if len(partial) == 1 else None
    if match:
        open_bank(match)
    else:
        st.session_state.pop("search_error", None)


@st.cache_data(show_spinner="Loading scores...")
def sc_load_scored() -> pd.DataFrame:
    scored = get_demo_panel()
    raw = pd.read_csv(DATA / "bank_panel.csv")
    keep = ["bank", "fy"] + [c for c in raw.columns if c not in scored.columns]
    return scored.merge(raw[keep], on=["bank", "fy"], how="left").sort_values(["bank", "fy_num"]).reset_index(drop=True)


# Default scoring settings (used by the bank pages). The Tools page below lets the user change them.
params = Params()
m_cut = params.m_cutoff()

SC_DATA = sc_load_scored()
SC_BANKS = sorted(SC_DATA["bank"].unique())
if "view" not in st.session_state:
    _qp = st.query_params.get("bank") if hasattr(st, "query_params") else None
    st.session_state["bank"] = _qp if _qp in SC_BANKS else None
    st.session_state["view"] = "bank" if st.session_state["bank"] else "home"

if st.session_state["view"] == "home":
    sc_css()
    render_home(SC_BANKS)
    st.stop()
elif st.session_state["view"] == "bank" and st.session_state.get("bank") in SC_BANKS:
    sc_css()
    render_bank(st.session_state["bank"], SC_DATA, SC_BANKS)
    st.stop()
# otherwise: "tools" view -> the original tabbed app continues below


# ----------------------------------------------------------------------------- settings (simple filter)
st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.caption("Applies to single-bank and bulk analysis.")

jc = st.sidebar.radio(
    "Manipulation check (Beneish M)", [1, 2],
    format_func=lambda v: {1: "Standard (flag above -1.78)", 2: "More sensitive (flag above -2.22)"}[v],
    help="A lower cut-off flags more companies.")

with st.sidebar.expander("Advanced"):
    x3 = st.radio("Profit used in Altman Z''", [1, 2],
                  format_func=lambda v: {1: "Profit before tax", 2: "Profit before tax + interest"}[v])
    recv = st.radio("Receivables measure", [2, 1],
                    format_func=lambda v: {1: "Interest receivable", 2: "Advances"}[v])
    tax = st.slider("Assumed tax rate", 0.0, 0.5, 0.25, 0.01, help="Only used when profit before tax is not entered.")
    neutral = st.checkbox("Fill missing Beneish values neutrally", value=False)

params = Params(x3_definition=x3, receivables_proxy=recv, joint_cutoff=jc, assumed_tax_rate=tax, neutral_fill=neutral)
m_cut = params.m_cutoff()

# Same sticky header as the home / bank pages: logo | section nav | Home | Tools
TOOLS_NAV = [("section-assess", "Assess a bank"), ("section-portfolio", "Portfolio overview"),
             ("section-bulk", "Bulk analysis"), ("section-validation", "Model validation")]
st.markdown("""
<style>
html { scroll-behavior:smooth; }
[data-testid="stToolbar"] { display:none; }                       /* keeps the bar clear of Streamlit's own top-right menu */
body:has(.st-key-topbar) .block-container { padding-top:0 !important; }
/* Streamlit's transparent top header sits on top of the bar and swallows every click: let clicks fall through it,
   but keep its own buttons (e.g. re-open sidebar) working */
header[data-testid="stHeader"] { background:transparent !important; pointer-events:none !important; }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a,
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { pointer-events:auto !important; }
/* every <style> markdown is an empty element and Streamlit's row-gap between them piles up into a blank strip above the bar */
[data-testid="stElementContainer"]:has(style) { display:none; }
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar), .st-key-topbar { position:sticky; top:0; z-index:1000; }
.st-key-topbar { position:relative; background:#fff; margin-bottom:.9rem; }
.st-key-topbar::after { content:""; position:absolute; left:0; right:0; bottom:0; height:1px;
    background:#e3e8f0; box-shadow:0 3px 10px rgba(15,42,74,.10); pointer-events:none; }
.st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; gap:.9rem !important; align-items:center; }
.st-key-topbar [data-testid="stHorizontalBlock"] > div { width:auto !important; min-width:0 !important; flex:0 0 auto !important; }
.st-key-topbar [data-testid="stHorizontalBlock"] > div:nth-child(2) { flex:1 1 0 !important; }
.topbar { display:flex; align-items:center; gap:.4rem; padding:0; white-space:nowrap; }
.topbar .logo { font-size:1.55rem; font-weight:500; letter-spacing:-1px; color:#2b2b2b; }
.topbar svg { height:1.2rem; }
.st-key-topbar .stButton > button { min-height:2.2rem; padding:.2rem .9rem; font-size:.88rem; white-space:nowrap; }
.section-nav-wrap { width:100%; }
.section-nav { display:flex; gap:.25rem; overflow-x:auto; white-space:nowrap; scrollbar-width:none; }
.section-nav::-webkit-scrollbar { display:none; }
.section-nav a { display:inline-block; padding:1rem 1rem .8rem; border-bottom:3px solid transparent; color:#5b6b82;
    font-size:.95rem; font-weight:600; text-decoration:none; transition:color .15s ease,border-color .15s ease; }
.section-nav a:hover, .section-nav a:focus, .section-nav a.is-active { color:#1e4fa3; border-bottom-color:#1e4fa3; }
.section-anchor { scroll-margin-top:4.5rem; height:1px; }
/* Match the home-page gray controls and blue hover treatment throughout Tools. */
body:has(#section-assess) .stTextInput input,
body:has(#section-assess) .stTextArea textarea,
body:has(#section-assess) [data-baseweb="select"] > div,
body:has(#section-assess) [data-baseweb="input"] > div,
body:has(#section-assess) [data-testid="stFileUploaderDropzone"] {
  background:#f1f3f7 !important; border-color:#d5dae3 !important;
  transition:background .18s ease, border-color .18s ease, box-shadow .18s ease;
}
body:has(#section-assess) .stTextInput input:hover,
body:has(#section-assess) .stTextArea textarea:hover,
body:has(#section-assess) [data-baseweb="select"] > div:hover,
body:has(#section-assess) [data-baseweb="input"] > div:hover,
body:has(#section-assess) .stTextInput input:focus,
body:has(#section-assess) .stTextArea textarea:focus,
body:has(#section-assess) [data-baseweb="select"] > div:focus-within,
body:has(#section-assess) [data-baseweb="input"] > div:focus-within,
body:has(#section-assess) [data-testid="stFileUploaderDropzone"]:hover {
  background:#eef4fc !important; border-color:#1e4fa3 !important;
  box-shadow:0 0 0 3px rgba(30,79,163,.12);
}
body:has(#section-assess) [data-testid="stDataEditor"] {
  border-color:#d5dae3 !important; transition:border-color .18s ease, box-shadow .18s ease;
}
body:has(#section-assess) [data-testid="stDataEditor"]:hover,
body:has(#section-assess) [data-testid="stDataEditor"]:focus-within {
  border-color:#1e4fa3 !important; box-shadow:0 0 0 3px rgba(30,79,163,.12);
}
body:has(#section-assess) .stButton > button:not([data-testid="stBaseButton-primary"]),
body:has(#section-assess) .stDownloadButton > button {
  background:#f1f3f7 !important; border:1px solid #d5dae3 !important; color:#7a8496 !important;
  box-shadow:none !important;
  transition:background .18s ease, border-color .18s ease, color .18s ease, box-shadow .18s ease, transform .18s ease;
}
body:has(#section-assess) .stButton > button:not([data-testid="stBaseButton-primary"]) *,
body:has(#section-assess) .stDownloadButton > button * { color:#7a8496 !important; }
body:has(#section-assess) .stButton > button:not([data-testid="stBaseButton-primary"]):hover,
body:has(#section-assess) .stDownloadButton > button:hover {
  background:#eef4fc !important; border-color:#1e4fa3 !important; color:#1e4fa3 !important;
  box-shadow:0 4px 14px rgba(30,79,163,.16) !important; transform:translateY(-1px);
}
body:has(#section-assess) .stButton > button:not([data-testid="stBaseButton-primary"]):hover *,
body:has(#section-assess) .stDownloadButton > button:hover * { color:#1e4fa3 !important; }
@media(max-width:768px) {
  .section-nav a { padding:.85rem .7rem .65rem; font-size:.88rem; }
  .st-key-topbar [data-testid="stHorizontalBlock"] > div:nth-child(1) { display:none; }
  .st-key-topbar .stButton > button { padding:.2rem .5rem; font-size:.8rem; }
}
</style>
""", unsafe_allow_html=True)

_TOOLS_NAV_HTML = """
<div class="section-nav-wrap"><nav class="section-nav">__NAV__</nav></div>
<script>
(() => {
  const scroller = () => document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
  const updateActiveSection = () => {
    const anchors = [...document.querySelectorAll(".section-anchor")];
    const links = [...document.querySelectorAll(".section-nav a")];
    if (!anchors.length || !links.length) return;
    const threshold = 110;                       /* just below the sticky bar */
    let current = anchors[0];
    for (const anchor of anchors) {
      if (anchor.getBoundingClientRect().top <= threshold) current = anchor;
      else break;
    }
    const sc = scroller();                       /* Streamlit scrolls inside stMain, not the window */
    if (sc.scrollHeight > sc.clientHeight + 50 && sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 2) {
      current = anchors[anchors.length - 1];
    }
    links.forEach(link => link.classList.toggle("is-active", link.hash === `#${current.id}`));
  };
  if (window.__bankRiskSectionScrollHandler) {
    document.removeEventListener("scroll", window.__bankRiskSectionScrollHandler, true);
    window.removeEventListener("resize", window.__bankRiskSectionScrollHandler);
  }
  let scheduled = false;
  window.__bankRiskSectionScrollHandler = () => {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(() => { updateActiveSection(); scheduled = false; });
  };
  document.addEventListener("scroll", window.__bankRiskSectionScrollHandler, { capture: true, passive: true });
  window.addEventListener("resize", window.__bankRiskSectionScrollHandler);
  updateActiveSection();
  setTimeout(updateActiveSection, 600);
  setTimeout(updateActiveSection, 1800);
})();
</script>
"""
_nav_links = "".join(f'<a href="#{a_}" class="{"is-active" if i == 0 else ""}">{t_}</a>' for i, (a_, t_) in enumerate(TOOLS_NAV))
with st.container(key="topbar"):
    _c_logo, _c_nav, _c_home, _c_tools = st.columns([1.5, 7, 1.2, 1.2], vertical_alignment="center")
    _c_logo.markdown(f'<div class="topbar"><span class="logo">bankrisk</span>{BARS_SVG}</div>', unsafe_allow_html=True)
    with _c_nav:
        st.html(_TOOLS_NAV_HTML.replace("__NAV__", _nav_links), unsafe_allow_javascript=True)
    _c_home.button("← Home", key="tools_home", on_click=go_home, width="stretch")
    _c_tools.button("⚙ Tools", key="tools_active", disabled=True, width="stretch")

# The four sections are stacked on one page (like the bank pages); the bar above scrolls to them and follows your position.
tab_assess, tab_demo, tab_batch, tab_val = (nullcontext() for _ in range(4))

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
    cfg = {"Previous FY": st.column_config.NumberColumn(f"Previous ({display_fy(fy_prev) if fy_prev else '-'})", format="%.2f", width="small"),
           "Current FY": st.column_config.NumberColumn(f"Current ({display_fy(fy_cur) if fy_cur else '-'})", format="%.2f", width="small")}
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
    st.markdown(f"### Result · {html.escape(res['company'])} · {html.escape(display_fy(res['fy']))}")
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
        light_table(comp.reset_index(), precision=3)
        if res["input_notes"]:
            st.caption("Derived / assumed: " + res["input_notes"])


with tab_assess:
    st.markdown('<div id="section-assess" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown("## Assess a bank")
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
    st.markdown('<div id="section-portfolio" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown("## Portfolio overview")
    demo = get_demo_panel()
    banks = sorted(demo["bank"].unique())
    st.selectbox("Open a bank's page", banks, index=None, placeholder="Search for a bank…", key="demo_open",
                 on_change=lambda: open_bank(st.session_state["demo_open"]) if st.session_state.get("demo_open") else None)
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
        score_ratio_table(d)
        st.download_button("⬇ Download scores (CSV)", d.to_csv(index=False), "demo_scores.csv", "text/csv")

# ============================================================================= Bulk analysis
RESULT_COLS = ["bank", "fy", "risk_score", "risk_band", "z_em", "z_zone", "m_score", "m_flag_a", "m_flag_b", "joint_signal",
               "anomaly_percentile", "early_warning_prob", "risk_tier", "structural_break", "x1_liquidity", "x2_reserves_ta",
               "x3_profit_ta", "x4_mcap_liab", "x5_income_ta", "z_classic", "z_classic_zone", "dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata",
               "boyd_graham_z", "roa", "nim", "feature_coverage", "input_notes"]

def clear_saved_portfolio():
    saved_upload_dir = OUTPUTS / "saved_upload"

    if saved_upload_dir.exists():
        for f in saved_upload_dir.glob("portfolio_upload.*"):
            f.unlink(missing_ok=True)

    # Use a separate flag instead of modifying the uploader's state
    st.session_state["portfolio_cleared"] = True
with tab_batch:
    st.markdown('<div id="section-bulk" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown("## Bulk analysis")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Upload a portfolio")
        st.markdown('<div class="hint">CSV or XLSX with <b>at least 2 consecutive years per bank</b> (3 for Boyd-Graham). '
                    'Column names can be snake_case or the original workbook headers.</div>', unsafe_allow_html=True)
        if st.session_state.pop("portfolio_cleared", False):
            st.session_state.pop("portfolio_upload", None)

        up = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx", "xlsm", "xls"],
            label_visibility="collapsed",
            key="portfolio_upload",
        )

        use_demo = st.checkbox(
            "Use the bundled panel instead (data/bank_panel.csv)",
            value=False,
        )
        st.button("Clear uploaded portfolio", key="clear_portfolio", on_click=clear_saved_portfolio)

        saved_upload_dir = OUTPUTS / "saved_upload"
        saved_upload_dir.mkdir(parents=True, exist_ok=True)
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
source_name = None

if up is not None:
    # Save the uploaded file
    suffix = Path(up.name).suffix.lower()
    saved_path = saved_upload_dir / f"portfolio_upload{suffix}"

    # Remove an older upload with a different extension
    for old_file in saved_upload_dir.glob("portfolio_upload.*"):
        if old_file != saved_path:
            old_file.unlink(missing_ok=True)

    saved_path.write_bytes(up.getvalue())
    source_name = up.name

    try:
        raw = read_table(saved_path, source_name)
    except Exception as e:
        st.error(f"Could not read file: {e}")

elif use_demo:
    # Load demo data only when explicitly selected
    try:
        raw = pd.read_csv(DATA / "bank_panel.csv")
    except Exception as e:
        st.error(f"Could not load demo data: {e}")

else:
    # Restore the previous upload after a browser refresh
    saved_files = list(saved_upload_dir.glob("portfolio_upload.*"))

    if saved_files:
        saved_path = saved_files[0]
        source_name = saved_path.name

        try:
            raw = read_table(saved_path, source_name)
        except Exception as e:
            st.error(f"Could not restore the uploaded file: {e}")

# Display dashboard results only when data is available
if raw is not None:
    st.divider()

    with st.expander(f"Input preview ({len(raw)} rows)"):
        light_table(raw.head(50))

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

        b = st.selectbox(
            "**Focus on one bank**",
            ["All banks"] + sorted(pred["bank"].astype(str).unique()),
        )

        view = (
            pred if b == "All banks"
            else pred[pred["bank"].astype(str) == b]
        )

        st.markdown("### Risk score over time")
        show_chart(risk_chart(view, 260))

        c1, c2 = st.columns(2, gap="large")

        with c1:
            show_chart(z_chart(view, 260))

        with c2:
            show_chart(m_chart(view, m_cut, 260))

        st.markdown("### Results")
        score_ratio_table(view)

        st.download_button(
            "⬇ Download results (CSV)",
            view[
                [c for c in RESULT_COLS if c in view.columns]
            ].to_csv(index=False),
            "scored_results.csv",
            "text/csv",
        )

else:
    st.info("Upload a portfolio CSV or Excel file to view risk statistics and results.")


# ============================================================================= Model validation
with tab_val:
    st.markdown('<div id="section-validation" class="section-anchor"></div>', unsafe_allow_html=True)
    st.markdown("## Model validation")
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
    light_table(pd.read_csv(v), precision=3)

    st.markdown("### Early-warning model")
    (st.success if ew.get("validated") else st.warning)(ew["note"])
    coefs = ew.get("standardised_coefficients")
    if coefs:
        cd = pd.DataFrame({"feature": list(coefs), "coefficient": list(coefs.values())})
        cd["label"] = cd["coefficient"].map(lambda v: f"{v:+.2f}")
        order = cd.sort_values("coefficient", ascending=False)["feature"].tolist()
        T = "Standardised coefficient (+ raises risk)"
        base = alt.Chart(cd).encode(y=alt.Y("feature:N", sort=order, title=None,
                                            axis=alt.Axis(grid=False, labelColor=INK, labelFontWeight=600, labelLimit=240)))
        bars = base.mark_bar(cornerRadiusEnd=7, size=20).encode(
            x=alt.X("coefficient:Q", title=T, axis=alt.Axis(grid=True)),
            color=alt.condition("datum.coefficient > 0", alt.value(BAND["High"]["solid"]), alt.value(BAND["Low"]["solid"])),
            tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f")])
        zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK, strokeWidth=1.3, opacity=.55).encode(x=alt.X("z:Q", title=T))
        lab_kw = dict(color=INK, fontWeight=700, fontSize=11, font=CHART_FONT)
        pos = base.mark_text(align="left", dx=7, **lab_kw).encode(
            x=alt.X("coefficient:Q", title=T), text="label:N").transform_filter("datum.coefficient >= 0")
        neg = base.mark_text(align="right", dx=-7, **lab_kw).encode(
            x=alt.X("coefficient:Q", title=T), text="label:N").transform_filter("datum.coefficient < 0")
        show_chart(style_chart((bars + zero + pos + neg).properties(height=max(34 * len(cd) + 50, 220))))
    if st.button("Retrain models from data"):
        train_and_save(verbose=False)
        st.cache_data.clear()
        st.success("Retrained. Reload the page to see updated outputs.")