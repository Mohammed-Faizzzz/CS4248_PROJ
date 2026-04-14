"""
Standardise every prediction CSV in model_predictions/:
  - Add `confidence` if missing (= prob of the predicted class via pred_label)
  - Drop any extra columns (e.g. pred_label)
  - Enforce column order: pred, prob_negative, prob_neutral, prob_positive, confidence
"""

from pathlib import Path
import pandas as pd

PRED_DIR = Path(__file__).parent.parent / "model_predictions"
LABEL_TO_COL = {
    "negative": "prob_negative",
    "neutral":  "prob_neutral",
    "positive": "prob_positive",
}
FINAL_COLS = ["pred", "prob_negative", "prob_neutral", "prob_positive", "confidence"]

for fp in sorted(PRED_DIR.glob("*.csv")):
    df = pd.read_csv(fp)
    changed = False

    if "pred" not in df.columns:
        for alias in ("predicted_label", "pred_label"):
            if alias in df.columns:
                df = df.rename(columns={alias: "pred"})
                changed = True
                break

    if "confidence" not in df.columns:
        prob_cols = df[list(LABEL_TO_COL.values())]
        col_index = df["pred"].map(LABEL_TO_COL)
        df["confidence"] = [prob_cols.loc[i, col] for i, col in col_index.items()]
        changed = True

    df = df[FINAL_COLS]
    df.to_csv(fp, index=False)

    if changed:
        print(f"fixed {fp.name}  (added confidence, standardised columns)")
    else:
        print(f"fixed {fp.name}  (standardised columns)")
