"""Export the Inputs / Ratios / Estimate_Log tabs of the research workbook to CSV.

Usage:
    python scripts/export_from_workbook.py path/to/workbook.xlsx

Writes:
    data/bank_panel.csv        raw bank-year inputs (snake_case columns, cached values incl. estimates)
    data/workbook_ratios.csv   the workbook's own Z''/M outputs (used only by tests/test_parity.py)
"""
import sys
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bank_risk.schema import RAW_COLUMN_MAP  # noqa: E402


def sheet_df(wb, name, max_row=None):
    ws = wb[name]
    rows = list(ws.iter_rows(min_row=1, max_row=max_row, values_only=True))
    return pd.DataFrame(rows[1:], columns=rows[0])


def main(path):
    out = Path(__file__).resolve().parents[1] / "data"
    wb = openpyxl.load_workbook(path, data_only=True)

    inputs = sheet_df(wb, "Inputs", max_row=55)
    inputs = inputs[inputs["Bank"].notna() & inputs["FY"].notna()].copy()
    inputs = inputs.rename(columns=RAW_COLUMN_MAP)[list(RAW_COLUMN_MAP.values())]

    # how many fields per bank-year are mechanical estimates (not reported data)
    log = sheet_df(wb, "Estimate_Log")
    n_est = log.groupby(["Bank", "FY"]).size().rename("n_estimated_fields").reset_index()
    n_est = n_est.rename(columns={"Bank": "bank", "FY": "fy"})
    inputs = inputs.merge(n_est, on=["bank", "fy"], how="left")
    inputs["n_estimated_fields"] = inputs["n_estimated_fields"].fillna(0).astype(int)
    inputs.to_csv(out / "bank_panel.csv", index=False)

    ratios = sheet_df(wb, "Ratios", max_row=55)
    ratios = ratios[ratios.iloc[:, 0].notna()]
    ratios.to_csv(out / "workbook_ratios.csv", index=False)
    print(f"bank_panel.csv: {inputs.shape}, workbook_ratios.csv: {ratios.shape}")


if __name__ == "__main__":
    main(sys.argv[1])
