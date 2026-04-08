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
            "f1_feature_model": round(f1_a, 4),
            "f1_roberta": round(f1_b, 4),
            "f1_delta": round(f1_b - f1_a, 4),
        })

    return pd.DataFrame(rows).sort_values("delta_agreement", ascending=True)


# Main

def main():
    parser = argparse.ArgumentParser(description="Divergence analysis")
    parser.add_argument("--annotations", required=True,
                        help="CSV with columns: text, clean_text, label")
    parser.add_argument("--feature-preds", required=True,
                        help="Pre-computed feature model predictions CSV "
                             "(from scripts.analysis.predict)")
    parser.add_argument("--roberta-preds", required=True,
                        help="Pre-computed RoBERTa predictions CSV "
                             "(from scripts.analysis.predict)")
    parser.add_argument("--feature-model-name", default="NB",
                        choices=["NB", "LR"],
                        help="Name of the feature-based model (for labels)")
    parser.add_argument("--output-dir", default="results/divergence")
    parser.add_argument("--text-col", default="text",
                        help="Raw text column (for linguistic tagging)")
    parser.add_argument("--label-col", default="label",
                        help="Gold label column")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fm = args.feature_model_name

    # Load annotations and predictions
    print("Loading annotations...")
    df = pd.read_csv(args.annotations)
    texts_raw = df[args.text_col]
    gold = df[args.label_col].values

    print(f"  Tweets: {len(df)}")
    print(f"  Label distribution: {dict(Counter(gold))}")

    feature_df_in = pd.read_csv(args.feature_preds)
    roberta_df_in = pd.read_csv(args.roberta_preds)
    
    feature_preds = feature_df_in["pred"].values
    roberta_preds = roberta_df_in["pred"].values
    
    # Load probabilities and confidence if available
    prob_cols = ["prob_negative", "prob_neutral", "prob_positive"]
    has_probs = all(c in feature_df_in.columns for c in prob_cols) and all(c in roberta_df_in.columns for c in prob_cols)
    
    if has_probs:
        feature_probs = feature_df_in[prob_cols].values
        roberta_probs = roberta_df_in[prob_cols].values
        feature_conf = feature_df_in["confidence"].values
        roberta_conf = roberta_df_in["confidence"].values
    else:
        feature_probs = roberta_probs = feature_conf = roberta_conf = None
 
    assert len(feature_preds) == len(gold), (
        f"Feature preds ({len(feature_preds)}) != annotations ({len(gold)})"
    )
    assert len(roberta_preds) == len(gold), (
        f"RoBERTa preds ({len(roberta_preds)}) != annotations ({len(gold)})"
    )

    # Individual model performance
    print(f"\n{'=' * 60}")
    print(f"  {fm} Performance on Elon Tweets")
    print("=" * 60)
    print(classification_report(gold, feature_preds,
                                target_names=LABEL_NAMES, zero_division=0))

    print("=" * 60)
    print("  RoBERTa Performance on Elon Tweets")
    print("=" * 60)
    print(classification_report(gold, roberta_preds,
                                target_names=LABEL_NAMES, zero_division=0))

    # Divergence metrics
    print("=" * 60)
    print("  Divergence Analysis")
    print("=" * 60)

    agree = agreement_rate(feature_preds, roberta_preds)
    kappa = cohen_kappa_score(feature_preds, roberta_preds)
    error_overlap = error_overlap_rate(feature_preds, roberta_preds, gold)
    per_class = per_class_agreement(feature_preds, roberta_preds, gold, LABEL_NAMES)

    print(f"\n  Agreement rate:     {agree:.4f}")
    print(f"  Cohen's kappa:      {kappa:.4f}")
    print(f"\n  Error overlap (Jaccard): {error_overlap['overlap_jaccard']:.4f}")
    print(f"    {fm}-only errors: {error_overlap['errors_a_only']}")
    print(f"    RoBERTa-only errors:  {error_overlap['errors_b_only']}")
    print(f"    Shared errors:        {error_overlap['errors_both']}")

    print("\n  Per-class agreement:")
    for cls, stats in per_class.items():
        print(f"    {cls:>10s}: {stats['agreement']:.4f}  (n={stats['count']})")

    mcnemar_res = mcnemars_test(feature_preds, roberta_preds, gold)
    print(f"\n  McNemar's Test (Statistical Significance of Divergence):")
    print(f"    Statistic: {mcnemar_res['statistic']:.4f}")
    print(f"    p-value:   {mcnemar_res['p_value']:.4e}")
    if mcnemar_res['p_value'] < 0.05:
        print("    -> The difference in error rates is statistically significant (p < 0.05).")
    else:
        print("    -> The difference in error rates is NOT statistically significant.")

    js_val = 0.0
    ece_feat = 0.0
    ece_rob = 0.0
    if has_probs:
        js_val = average_js_divergence(feature_probs, roberta_probs)
        ece_feat = expected_calibration_error(feature_preds, feature_conf, gold)
        ece_rob = expected_calibration_error(roberta_preds, roberta_conf, gold)
        
        print(f"\n  Soft Probability Metrics:")
        print(f"    Average JS Divergence: {js_val:.4f}")
        print(f"\n  Expected Calibration Error (ECE):")
        print(f"    {fm} ECE:      {ece_feat:.4f} (lower is better calibrated)")
        print(f"    RoBERTa ECE: {ece_rob:.4f} (lower is better calibrated)")

    # Cross-tabulation
    cross_tab = disagreement_confusion(feature_preds, roberta_preds, LABEL_NAMES)
    print(f"\n  Prediction cross-tabulation ({fm} rows × RoBERTa cols):")
    print(cross_tab.to_string())

    # Linguistic feature breakdown
    print(f"\n{'=' * 60}")
    print("  Linguistic Feature Divergence Breakdown")
    print("=" * 60)

    feature_df = tag_linguistic_features(texts_raw)
    breakdown = feature_divergence_breakdown(feature_df, feature_preds,
                                            roberta_preds, gold)

    print(breakdown.to_string(index=False))

    # Disagreement examples
    disagree_mask = feature_preds != roberta_preds
    disagree_df = df[disagree_mask].copy()
    disagree_df[f"{fm}_pred"] = feature_preds[disagree_mask]
    disagree_df["roberta_pred"] = roberta_preds[disagree_mask]
    disagree_df[f"{fm}_pred_label"] = [
        LABEL_NAMES[p] for p in feature_preds[disagree_mask]
    ]
    disagree_df["roberta_pred_label"] = [
        LABEL_NAMES[p] for p in roberta_preds[disagree_mask]
    ]

    print(f"\n  Total disagreements: {disagree_mask.sum()} / {len(df)} "
          f"({disagree_mask.sum() / len(df) * 100:.1f}%)")

    # Save outputs
    # Full results with predictions
    results_df = df.copy()
    results_df[f"{fm}_pred"] = feature_preds
    results_df["roberta_pred"] = roberta_preds
    results_df["agree"] = (feature_preds == roberta_preds).astype(int)
    results_df[f"{fm}_correct"] = (feature_preds == gold).astype(int)
    results_df["roberta_correct"] = (roberta_preds == gold).astype(int)

    # Add linguistic features
    for col in feature_df.columns:
        results_df[f"feat_{col}"] = feature_df[col].values

    results_df.to_csv(out / "full_results.csv", index=False)
    disagree_df.to_csv(out / "disagreements.csv", index=False)
    breakdown.to_csv(out / "feature_breakdown.csv", index=False)

    # Summary JSON
    summary = {
        "n_tweets": len(df),
        "feature_model": fm,
        "agreement_rate": round(agree, 4),
        "cohens_kappa": round(kappa, 4),
        "error_overlap": {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in error_overlap.items()},
        "per_class_agreement": per_class,
        "feature_model_metrics": {
            "accuracy": round(accuracy_score(gold, feature_preds), 4),
            "macro_f1": round(f1_score(gold, feature_preds, average="macro",
                                       zero_division=0), 4),
        },
        "roberta_metrics": {
            "accuracy": round(accuracy_score(gold, roberta_preds), 4),
            "macro_f1": round(f1_score(gold, roberta_preds, average="macro",
                                       zero_division=0), 4),
        },
        "mcnemars_test": mcnemar_res,
    }
    
    if has_probs:
        summary["soft_metrics"] = {
            "average_js_divergence": round(js_val, 4),
            f"{fm}_ece": round(ece_feat, 4),
            "roberta_ece": round(ece_rob, 4)
        }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {out}/")
    print(f"  full_results.csv      — all tweets with predictions & features")
    print(f"  disagreements.csv     — tweets where models disagree")
    print(f"  feature_breakdown.csv — per-feature divergence stats")
    print(f"  summary.json          — key metrics")


if __name__ == "__main__":
    main()
