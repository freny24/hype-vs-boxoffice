"""
model.py
--------
XGBoost model to predict opening weekend box office from pre-release signals.
Includes SHAP explainability so we can tell recruiters (and users) WHY
the model predicts what it predicts — not just what.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


FEATURE_COLS = [
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

TARGET_COL = "log_opening"


def train(df: pd.DataFrame, save_path: str = "models/xgb_model.pkl") -> dict:
    """
    Train XGBoost on feature-engineered data.

    Returns:
        dict with model, metrics, SHAP explainer, and feature importance
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].fillna(0)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"RMSE (log scale): {rmse:.3f} | R²: {r2:.3f}")

    # SHAP values for explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    feature_importance = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_,
        "shap_mean_abs": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_mean_abs", ascending=False)

    joblib.dump({"model": model, "explainer": explainer}, save_path)
    print(f"Model saved to {save_path}")

    return {
        "model": model,
        "explainer": explainer,
        "metrics": {"rmse": rmse, "r2": r2},
        "feature_importance": feature_importance,
        "X_test": X_test,
        "y_test": y_test,
        "shap_values": shap_values,
    }


def predict_opening(model, features: dict) -> dict:
    """
    Predict opening weekend for a single film.

    Args:
        model: trained XGBoost model
        features: dict of feature values

    Returns:
        dict with predicted_opening (dollars) and confidence interval
    """
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
    log_pred = model.predict(X)[0]
    predicted = np.expm1(log_pred)

    # Rough 80% PI from training RMSE (~0.43 in log space)
    rmse_log = 0.43
    lower = np.expm1(log_pred - 1.28 * rmse_log)
    upper = np.expm1(log_pred + 1.28 * rmse_log)

    return {
        "predicted_opening": round(predicted, 0),
        "ci_lower": round(lower, 0),
        "ci_upper": round(upper, 0),
    }


if __name__ == "__main__":
    # Load processed data and train (run after running the notebooks)
    try:
        df = pd.read_csv("data/processed/films_features.csv")
        results = train(df)
        print(results["feature_importance"].head(8))
    except FileNotFoundError:
        print("Run notebooks 01-03 first to generate processed data.")
