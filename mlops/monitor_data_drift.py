
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "transactions.csv"
RESULTS_DIR = ROOT / "results"
REPORT_PATH = RESULTS_DIR / "data_drift_report.json"

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

SAMPLE_ROWS = 100_000
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Sample rows to limit memory use.
    df = pd.read_csv(
        DATA_PATH,
        usecols=NUMERIC_COLUMNS + ["type", "isFraud"],
        nrows=SAMPLE_ROWS,
    )

    report = {
        "dataset": str(DATA_PATH.name),
        "rows_checked": int(len(df)),
        "sampling_note": (
            "First 100,000 rows; this is a baseline diagnostic, "
            "not a production traffic drift measurement."
        ),
        "missing_values": {
            column: int(count)
            for column, count in df.isnull().sum().items()
        },
        "fraud_rate": float(df["isFraud"].mean()),
        "transaction_type_distribution": {
            str(key): float(value)
            for key, value in
            df["type"].value_counts(normalize=True).items()
        },
        "numeric_summary": {},
    }

    for column in NUMERIC_COLUMNS:
        series = pd.to_numeric(df[column], errors="coerce")
        report["numeric_summary"][column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Monitoring baseline saved to: {REPORT_PATH}")
    print(f"Rows checked: {len(df)}")
    print(f"Fraud rate: {report['fraud_rate']:.6f}")
    print("This is a baseline report, not an automated drift alert.")


if __name__ == "__main__":
    main()

