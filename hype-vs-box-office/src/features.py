"""
features.py
-----------
Constructs the composite Hype Score (0–100) from:
  - Reddit engagement signals (volume, velocity, sentiment)
  - Trailer metrics (views, like ratio)
  - Genre/budget/franchise priors

Also builds the Performance Score from actual box office data.
The gap between Hype Score and Performance Score is the project's
core insight: positive gap = overhyped, negative gap = underrated.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Hype Score construction
# ---------------------------------------------------------------------------

def build_hype_score(row: pd.Series) -> float:
    """
    Weighted composite of pre-release signals, normalized to 0–100.

    Weights chosen based on correlation analysis with opening weekend:
        - weighted_sentiment:        30%
        - score_velocity (Reddit):   25%
        - trailer_views_log:         25%
        - comment_to_post_ratio:     10%
        - pct_positive:              10%
    """
    WEIGHTS = {
        "weighted_sentiment_norm": 0.30,
        "score_velocity_norm":     0.25,
        "trailer_views_log_norm":  0.25,
        "comment_to_post_ratio_norm": 0.10,
        "pct_positive_norm":       0.10,
    }

    score = 0.0
    for col, w in WEIGHTS.items():
        score += row.get(col, 0.5) * w  # default to 0.5 if missing

    return round(score * 100, 2)


def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Min-max normalize a set of columns, appending _norm suffix."""
    df = df.copy()
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(df[feature_cols].fillna(0))
    for i, col in enumerate(feature_cols):
        df[f"{col}_norm"] = normed[:, i]
    return df


def build_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Performance Score (0–100) from box office data.

    Formula: weighted blend of
        - opening_roi: opening_weekend / production_budget
        - total_roi:   total_gross / production_budget
        - legs: total_gross / opening_weekend (measures staying power)
    """
    df = df.copy()

    # Avoid division by zero
    budget = df["production_budget"].replace(0, np.nan)

    df["opening_roi"]   = df["opening_weekend"] / budget
    df["total_roi"]     = df["total_gross"] / budget
    df["legs"]          = df["total_gross"] / df["opening_weekend"].replace(0, np.nan)

    perf_features = ["opening_roi", "total_roi", "legs"]
    df = normalize_features(df, perf_features)

    df["performance_score"] = (
        df["opening_roi_norm"]  * 0.40 +
        df["total_roi_norm"]    * 0.40 +
        df["legs_norm"]         * 0.20
    ) * 100

    return df


def compute_hype_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    hype_gap = hype_score - performance_score
    Positive → overhyped (expectations exceeded reality)
    Negative → underrated (exceeded expectations)
    """
    df = df.copy()
    df["hype_gap"] = df["hype_score"] - df["performance_score"]

    df["verdict"] = pd.cut(
        df["hype_gap"],
        bins=[-np.inf, -20, -5, 5, 20, np.inf],
        labels=["Surprise Hit", "Slightly Underrated", "As Expected", "Slightly Overhyped", "Overhyped"]
    )
    return df


def engineer_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final feature set for the XGBoost prediction model.
    Target: log1p(opening_weekend)
    """
    df = df.copy()
    df["log_opening"] = np.log1p(df["opening_weekend"])
    df["log_budget"]  = np.log1p(df["production_budget"].fillna(0))
    df["log_trailer_views"] = np.log1p(df["trailer_views"].fillna(0))
    df["days_since_prev_release"] = df["days_since_prev_release"].fillna(90)

    feature_cols = [
        "log_budget",
        "log_trailer_views",
        "weighted_sentiment",
        "score_velocity",
        "comment_to_post_ratio",
        "pct_positive",
        "sentiment_std",
        "is_franchise",
        "is_sequel",
        "days_since_prev_release",
        "release_month",
        "genre_encoded",
    ]

    existing = [c for c in feature_cols if c in df.columns]
    return df, existing


if __name__ == "__main__":
    # Minimal smoke test
    mock = pd.DataFrame([{
        "weighted_sentiment": 0.45,
        "score_velocity": 1200,
        "trailer_views": 25_000_000,
        "comment_to_post_ratio": 45,
        "pct_positive": 0.72,
        "opening_weekend": 80_000_000,
        "total_gross": 950_000_000,
        "production_budget": 185_000_000,
        "days_since_prev_release": 180,
        "release_month": 7,
        "is_franchise": 1,
        "is_sequel": 0,
        "genre_encoded": 3,
    }])

    raw_features = ["weighted_sentiment", "score_velocity", "trailer_views", "comment_to_post_ratio", "pct_positive"]
    mock = normalize_features(mock, raw_features)
    mock["trailer_views_log"] = np.log1p(mock["trailer_views"])
    mock = normalize_features(mock, ["trailer_views_log"])
    mock["hype_score"] = mock.apply(build_hype_score, axis=1)
    mock = build_performance_score(mock)
    mock = compute_hype_gap(mock)
    print(mock[["hype_score", "performance_score", "hype_gap", "verdict"]])
