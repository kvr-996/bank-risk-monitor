"""Column schema: the raw fields a user supplies for each bank-year."""

# workbook header -> snake_case
RAW_COLUMN_MAP = {
    "Bank": "bank",
    "Segment": "segment",
    "FY": "fy",
    "Basis": "basis",
    "Total assets": "total_assets",
    "Total liabilities": "total_liabilities",
    "Equity": "equity",
    "Cash & bank balances": "cash",
    "Investments": "investments",
    "Gross advances": "gross_advances",
    "Deposits": "deposits",
    "Interest income": "interest_income",
    "Fee / other income": "other_income",
    "Total income": "total_income",
    "Operating expenses": "operating_expenses",
    "Net interest income": "net_interest_income",
    "Net profit": "net_profit",
    "Operating cash flow": "operating_cash_flow",
    "Depreciation": "depreciation",
    "Receivables / interest receivable": "interest_receivable",
    "Earning assets": "earning_assets",
    "CRAR / CAR": "crar",
    "CET1": "cet1",
    "GNPA %": "gnpa",
    "NNPA %": "nnpa",
    "Provision coverage %": "pcr",
    "Market cap": "market_cap",
    "Borrowings (added)": "borrowings",
    "Other liabilities (added)": "other_liabilities",
    "Equity capital (added)": "equity_capital",
    "Reserves (added)": "reserves",
    "Fixed assets (added)": "fixed_assets",
    "Interest expense (added)": "interest_expense",
    "Profit before tax (added)": "pbt",
    "Net advances (added)": "net_advances",
}

ID_COLUMNS = ["bank", "fy"]
TEXT_COLUMNS = ["bank", "segment", "fy", "basis"]
NUMERIC_COLUMNS = [c for c in RAW_COLUMN_MAP.values() if c not in TEXT_COLUMNS]

# Minimum a website user must supply for Altman Z'' + Beneish M to be computable.
# Everything else is derived by fallbacks in scoring.fill_derived_fields().
REQUIRED_MINIMUM = [
    "bank", "fy", "total_assets", "total_liabilities", "equity", "cash", "investments",
    "deposits", "total_income", "net_profit", "operating_cash_flow", "market_cap",
]

# ---- investor workflow (one company, two consecutive years) -------------------------------------
# Everything an investor MUST type in for a reliable Z'' + Beneish M + ML risk score.
# Beneish needs the extra four P&L / fixed-asset fields; the ML layer needs gnpa, nnpa and crar
# (without them feature coverage drops below 60% and the tier becomes "Insufficient data").
INVESTOR_REQUIRED = REQUIRED_MINIMUM + [
    "depreciation", "fixed_assets", "interest_expense", "operating_expenses",
    "gnpa", "nnpa", "crar",
]
# Nice to have: improves accuracy, otherwise derived (see scoring.fill_derived_fields).
INVESTOR_OPTIONAL = [
    "gross_advances", "net_advances", "interest_income", "other_income", "net_interest_income",
    "equity_capital", "reserves", "pbt", "borrowings", "other_liabilities",
    "earning_assets", "interest_receivable", "cet1", "pcr",
]
# Fields entered as percentages in the UI and divided by 100 before scoring.
PERCENT_FIELDS = ["gnpa", "nnpa", "crar", "cet1", "pcr"]
MIN_YEARS_FOR_INVESTOR = 2   # Beneish compares with the prior year