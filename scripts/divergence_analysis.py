"""
Divergence analysis between feature-based classifiers and fine-tuned RoBERTa
on manually annotated Elon Musk tweets.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2


LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_MAP = {name: i for i, name in enumerate(LABEL_NAMES)}


# Linguistic feature taggers

def has_question(text: str) -> bool:
    return "?" in str(text)


def has_exclamation(text: str) -> bool:
    return "!" in str(text)


def has_emoticon(text: str) -> bool:
    """Detect common text emoticons (not emoji)."""
    pattern = r"[:;=8][\-']?[)(DPpOo/\\|@#\*]|[)(DPp][\-']?[:;=8]|<3|</3|[xX][Dd]"
    return bool(re.search(pattern, str(text)))


def has_emoji(text: str) -> bool:
    """Detect Unicode emoji."""
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"   # supplemental symbols
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    return bool(emoji_pattern.search(str(text)))


def has_allcaps_word(text: str) -> bool:
    """Word with 3+ uppercase letters (likely emphasis)."""
    return bool(re.search(r"\b[A-Z]{3,}\b", str(text)))


def has_negation(text: str) -> bool:
    negation_cues = {
        "not", "no", "never", "neither", "nobody", "nothing",
        "nowhere", "nor", "hardly", "barely", "scarcely",
        "don't", "doesn't", "didn't", "isn't", "wasn't",
        "weren't", "won't", "wouldn't", "shouldn't", "couldn't",
        "hasn't", "haven't", "hadn't", "aren't", "can't", "ain't",
    }
    tokens = set(str(text).lower().split())
    return bool(tokens & negation_cues)


def has_superlative(text: str) -> bool:
    """Detect superlatives and intensifiers."""
    pattern = (
        r"\b(best|worst|most|least|greatest|amazing|terrible"
        r"|incredible|awful|fantastic|horrible|excellent|disgusting)\b"
    )
    return bool(re.search(pattern, str(text).lower()))


def has_sarcasm_marker(text: str) -> bool:
    """Heuristic sarcasm markers: /s tag, common ironic phrases, ellipsis + emphasis."""
    t = str(text).lower()
    markers = ["/s", "yeah right", "oh great", "sure thing", "totally", "obviously"]
    if any(m in t for m in markers):
        return True
    if "..." in t and ("!" in text or has_allcaps_word(text)):
        return True
    return False


def has_url(text: str) -> bool:
    return bool(re.search(r"https?://\S+", str(text)))


def has_mention(text: str) -> bool:
    return bool(re.search(r"@\w+", str(text)))


def has_hashtag(text: str) -> bool:
    return bool(re.search(r"#\w+", str(text)))


def is_short(text: str, threshold: int = 5) -> bool:
    """Tweets with <= threshold words."""
    return len(str(text).split()) <= threshold


FEATURE_TAGGERS = {
    "question": has_question,
    "exclamation": has_exclamation,
    "emoticon": has_emoticon,
    "emoji": has_emoji,
    "allcaps": has_allcaps_word,
    "negation": has_negation,
    "superlative": has_superlative,
    "sarcasm_marker": has_sarcasm_marker,
    "url": has_url,
    "mention": has_mention,
    "hashtag": has_hashtag,
    "short_text": is_short,
}


# Divergence metrics

def agreement_rate(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    """Fraction of instances where both models agree."""
    return float(np.mean(preds_a == preds_b))


def error_overlap_rate(preds_a: np.ndarray, preds_b: np.ndarray,
                       gold: np.ndarray) -> dict:
    """
    Compute error overlap between two models.
    
    Returns counts for errors unique to A, unique to B, and shared.
    """
    errors_a = set(np.where(preds_a != gold)[0])
    errors_b = set(np.where(preds_b != gold)[0])

    intersection = errors_a & errors_b
    union = errors_a | errors_b

    return {
        "overlap_jaccard": len(intersection) / len(union) if union else 0.0,
        "errors_a_only": len(errors_a - errors_b),
        "errors_b_only": len(errors_b - errors_a),
        "errors_both": len(intersection),
        "errors_a_total": len(errors_a),
        "errors_b_total": len(errors_b),
    }


def per_class_agreement(preds_a: np.ndarray, preds_b: np.ndarray,
                        gold: np.ndarray, label_names: list[str]) -> dict:
    """Agreement rate broken down by gold label."""
    results = {}
    for i, name in enumerate(label_names):
        mask = gold == i
        if mask.sum() == 0:
            continue
        results[name] = {
            "count": int(mask.sum()),
            "agreement": float(np.mean(preds_a[mask] == preds_b[mask])),
        }
    return results


def disagreement_confusion(preds_a: np.ndarray, preds_b: np.ndarray,
                           label_names: list[str]) -> pd.DataFrame:
    """
    Cross-tabulation: rows = model A, cols = model B.
    """
    return pd.crosstab(
        pd.Series(preds_a, name="Feature Model"),
        pd.Series(preds_b, name="RoBERTa"),
    ).rename(
        index={i: label_names[i] for i in range(len(label_names))},
        columns={i: label_names[i] for i in range(len(label_names))},
    )


def mcnemars_test(preds_a: np.ndarray, preds_b: np.ndarray, gold: np.ndarray) -> dict:
    """Calculate McNemar's test for statistical significance of disagreement."""
    b = ((preds_a == gold) & (preds_b != gold)).sum()
    c = ((preds_a != gold) & (preds_b == gold)).sum()
    
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0}
        
    statistic = ((np.abs(b - c) - 1)**2) / (b + c)
    p_value = chi2.sf(statistic, df=1)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def expected_calibration_error(preds: np.ndarray, confidences: np.ndarray, gold: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across confidence bins."""
    ece = 0.0
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences >= bin_lower) & (confidences < bin_upper)
            
        n = mask.sum()
        if n > 0:
            acc = (preds[mask] == gold[mask]).mean()
            conf = confidences[mask].mean()
            ece += (n / len(preds)) * np.abs(acc - conf)
    return float(ece)


def average_js_divergence(probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    """Calculate average Jensen-Shannon divergence between probability distributions."""
    js_divs = [jensenshannon(p_a, p_b) for p_a, p_b in zip(probs_a, probs_b)]
    valid_js = [x for x in js_divs if not np.isnan(x)]
    return float(np.mean(valid_js)) if valid_js else 0.0


# Linguistic feature analysis

def tag_linguistic_features(texts: pd.Series) -> pd.DataFrame:
    """Tag each text with binary linguistic feature indicators."""
    features = {}
    for name, fn in FEATURE_TAGGERS.items():
        features[name] = texts.map(fn).astype(int)
    return pd.DataFrame(features)


def feature_divergence_breakdown(
    feature_df: pd.DataFrame,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    gold: np.ndarray,
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> pd.DataFrame:
    """
    For each linguistic feature, compute agreement and F1 for tweets
    with vs without the feature. Sorted by delta_agreement ascending
    (features that hurt agreement most appear first).
    """
    disagree = preds_a != preds_b
    rows = []

    for feat in feature_df.columns:
        mask = feature_df[feat] == 1
        n = mask.sum()
        if n == 0:
            continue

        agree_with = float(np.mean(~disagree[mask]))
        agree_without = float(np.mean(~disagree[~mask])) if (~mask).sum() > 0 else None

        f1_a = f1_score(gold[mask], preds_a[mask], average="macro", zero_division=0)
        f1_b = f1_score(gold[mask], preds_b[mask], average="macro", zero_division=0)

        rows.append({
            "feature": feat,
            "count": int(n),
            "pct": round(n / len(feature_df) * 100, 1),
            "agreement": round(agree_with, 4),
            "agreement_without": round(agree_without, 4) if agree_without is not None else None,
            "delta_agreement": round(agree_with - agree_without, 4) if agree_without is not None else None,
            f"f1_{name_a}": round(f1_a, 4),
            f"f1_{name_b}": round(f1_b, 4),
            "f1_delta (b-a)": round(f1_b - f1_a, 4),
        })

    return pd.DataFrame(rows).sort_values("delta_agreement", ascending=True)


# Markdown helpers

def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


def _md_matrix(names: list[str], matrix: dict, fmt: str = ".4f") -> str:
    """Render a symmetric named matrix as a markdown table."""
    header = "| |" + "".join(f" `{n}` |" for n in names)
    sep    = "|---|" + "---|" * len(names)
    rows   = []
    for a in names:
        row = f"| `{a}` |"
        for b in names:
            if a == b:
                row += " — |"
            else:
                key = (min(a, b), max(a, b))
                row += f" {matrix[key]:{fmt}} |"
        rows.append(row)
    return "\n".join([header, sep] + rows)


# Main

def main():
    parser = argparse.ArgumentParser(description="All-pairs divergence analysis")
    parser.add_argument("--annotations", required=True,
                        help="CSV with columns: text, label (and optionally clean_text)")
    parser.add_argument("--predictions-dir", default="model_predictions",
                        help="Directory of per-model prediction CSVs "
                             "(see README for naming convention)")
    parser.add_argument("--output-dir", default="results/divergence")
    parser.add_argument("--text-col", default="text",
                        help="Raw text column (used for linguistic feature tagging)")
    parser.add_argument("--label-col", default="label",
                        help="Gold label column (integer 0/1/2)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.predictions_dir)

    # ── Load annotations ────────────────────────────────────────────────────
    print("Loading annotations...")
    df = pd.read_csv(args.annotations)
    texts_raw = df[args.text_col]
    gold = df[args.label_col].values
    n = len(df)
    print(f"  Tweets: {n}")
    print(f"  Label distribution: {dict(Counter(gold))}")

    # ── Discover and load model predictions ─────────────────────────────────
    pred_files = sorted(pred_dir.glob("*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No CSV files found in {pred_dir}")

    PROB_COLS = ["prob_negative", "prob_neutral", "prob_positive"]
    models: dict[str, dict] = {}
    for fp in pred_files:
        name = fp.stem
        mdf = pd.read_csv(fp)
        has_probs = all(c in mdf.columns for c in PROB_COLS)
        models[name] = {
            "preds": mdf["pred"].values,
            "probs": mdf[PROB_COLS].values if has_probs else None,
            "conf":  mdf["confidence"].values if has_probs else None,
        }
        assert len(models[name]["preds"]) == n, (
            f"{name}: {len(models[name]['preds'])} rows != annotations {n}"
        )

    model_names = list(models.keys())
    pairs = [(a, b) for i, a in enumerate(model_names)
             for b in model_names[i + 1:]]

    print(f"Loaded {len(model_names)} models: {model_names}")
    print(f"All-pairs comparisons: {len(pairs)}")

    # ── Linguistic features (computed once) ──────────────────────────────────
    feat_df = tag_linguistic_features(texts_raw)

    # ── Build markdown report ────────────────────────────────────────────────
    lines: list[str] = []

    lines += [
        "# Divergence Analysis Report",
        "",
        f"**Dataset:** `{args.annotations}` | **N:** {n}  ",
        f"**Models:** {', '.join(f'`{m}`' for m in model_names)}  ",
        "**Label distribution:** " +
        " | ".join(f"{LABEL_NAMES[i]}: {int((gold == i).sum())}" for i in range(3)),
        "",
    ]

    # ── Per-model performance ────────────────────────────────────────────────
    lines += ["## Model Performance", ""]
    perf_rows = []
    for name, m in models.items():
        preds = m["preds"]
        acc = accuracy_score(gold, preds)
        f1  = f1_score(gold, preds, average="macro", zero_division=0)
        ece = (expected_calibration_error(preds, m["conf"], gold)
               if m["conf"] is not None else float("nan"))
        perf_rows.append({
            "model": f"`{name}`",
            "accuracy": f"{acc:.4f}",
            "macro_f1": f"{f1:.4f}",
            "ece": f"{ece:.4f}" if not np.isnan(ece) else "N/A",
        })
    lines += [_md_table(pd.DataFrame(perf_rows)), ""]

    # Per-model classification reports in collapsible blocks
    lines += ["### Full Classification Reports", ""]
    for name, m in models.items():
        report = classification_report(gold, m["preds"],
                                       target_names=LABEL_NAMES, zero_division=0)
        lines += [
            f"<details><summary><code>{name}</code></summary>",
            "",
            "```",
            report.strip(),
            "```",
            "",
            "</details>",
            "",
        ]

    # ── Pairwise agreement matrix ────────────────────────────────────────────
    agree_mat: dict[tuple, float] = {}
    kappa_mat: dict[tuple, float] = {}
    js_mat:    dict[tuple, float] = {}
    for a, b in pairs:
        key = (a, b)
        agree_mat[key] = agreement_rate(models[a]["preds"], models[b]["preds"])
        kappa_mat[key] = cohen_kappa_score(models[a]["preds"], models[b]["preds"])
        if models[a]["probs"] is not None and models[b]["probs"] is not None:
            js_mat[key] = average_js_divergence(models[a]["probs"], models[b]["probs"])

    lines += ["## Pairwise Agreement Rate", "", _md_matrix(model_names, agree_mat), ""]
    lines += ["## Pairwise Cohen's Kappa",  "", _md_matrix(model_names, kappa_mat), ""]

    if js_mat:
        js_names = [n for n in model_names
                    if any(k[0] == n or k[1] == n for k in js_mat)]
        lines += ["## Pairwise JS Divergence (soft probabilities)",
                  "", _md_matrix(js_names, js_mat), ""]

    # ── McNemar's test ───────────────────────────────────────────────────────
    lines += ["## McNemar's Test", ""]
    mcn_rows = []
    for a, b in pairs:
        res = mcnemars_test(models[a]["preds"], models[b]["preds"], gold)
        mcn_rows.append({
            "pair": f"`{a}` vs `{b}`",
            "statistic": f"{res['statistic']:.4f}",
            "p-value":   f"{res['p_value']:.4e}",
            "significant (p<0.05)": "yes" if res["p_value"] < 0.05 else "no",
        })
    lines += [_md_table(pd.DataFrame(mcn_rows)), ""]

    # ── Error overlap ────────────────────────────────────────────────────────
    lines += ["## Error Overlap", ""]
    eo_rows = []
    for a, b in pairs:
        eo = error_overlap_rate(models[a]["preds"], models[b]["preds"], gold)
        eo_rows.append({
            "pair":           f"`{a}` vs `{b}`",
            "jaccard":        f"{eo['overlap_jaccard']:.4f}",
            "only-a errors":  eo["errors_a_only"],
            "only-b errors":  eo["errors_b_only"],
            "shared errors":  eo["errors_both"],
            "total-a errors": eo["errors_a_total"],
            "total-b errors": eo["errors_b_total"],
        })
    lines += [_md_table(pd.DataFrame(eo_rows)), ""]

    # ── Per-pair detail sections ─────────────────────────────────────────────
    lines += ["## Per-Pair Details", ""]
    for a, b in pairs:
        pa, pb = models[a]["preds"], models[b]["preds"]
        lines += [f"### `{a}` vs `{b}`", ""]

        # Per-class agreement
        pca = per_class_agreement(pa, pb, gold, LABEL_NAMES)
        pca_rows = [{"class": cls, "n": s["count"],
                     "agreement": f"{s['agreement']:.4f}"}
                    for cls, s in pca.items()]
        lines += ["**Per-class agreement**", "", _md_table(pd.DataFrame(pca_rows)), ""]

        # Cross-tabulation
        ct = disagreement_confusion(pa, pb, LABEL_NAMES)
        ct_header = f"| `{a}` \\ `{b}` |" + "".join(f" {c} |" for c in ct.columns)
        ct_sep    = "|---|" + "---|" * len(ct.columns)
        ct_rows   = [ct_header, ct_sep]
        for idx, row_data in ct.iterrows():
            ct_rows.append(f"| **{idx}** |" + "".join(f" {v} |" for v in row_data))
        lines += [f"**Prediction cross-tabulation** (`{a}` rows × `{b}` cols)",
                  "", "\n".join(ct_rows), ""]

        # Linguistic feature breakdown
        bd = feature_divergence_breakdown(feat_df, pa, pb, gold,
                                          name_a=a, name_b=b)
        lines += ["**Linguistic feature divergence** (sorted by Δagreement ↑ = hurts most)",
                  "", _md_table(bd), ""]

    # ── Write report ─────────────────────────────────────────────────────────
    report_path = out / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")

    # Per-pair disagreement CSVs
    for a, b in pairs:
        pa, pb = models[a]["preds"], models[b]["preds"]
        mask = pa != pb
        d = df[mask].copy()
        d[f"{a}_pred"]       = pa[mask]
        d[f"{b}_pred"]       = pb[mask]
        d[f"{a}_pred_label"] = [LABEL_NAMES[p] for p in pa[mask]]
        d[f"{b}_pred_label"] = [LABEL_NAMES[p] for p in pb[mask]]
        d.to_csv(out / f"disagreements_{a}_vs_{b}.csv", index=False)

    print(f"Disagreement CSVs saved to {out}/")


if __name__ == "__main__":
    main()
