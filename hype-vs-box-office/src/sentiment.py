"""
sentiment.py
-------------
Two-layer sentiment scoring:
  1. VADER — fast, lexicon-based, good for short social text
  2. DistilBERT — transformer-based, captures context and sarcasm better

We score each post title+selftext with both, then compare.
The transformer score is used as the primary signal; VADER as a feature.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm import tqdm

# Lazy-load the transformer (heavy, ~250MB download on first run)
_transformer_pipeline = None


def get_transformer():
    global _transformer_pipeline
    if _transformer_pipeline is None:
        print("Loading DistilBERT sentiment model (first run only)...")
        _transformer_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _transformer_pipeline


def vader_score(text: str) -> dict:
    """Returns VADER compound, pos, neu, neg scores."""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(text))


def transformer_score(text: str) -> float:
    """
    Returns a [-1, 1] sentiment score from DistilBERT.
    POSITIVE maps to +score, NEGATIVE maps to -score.
    """
    model = get_transformer()
    result = model(str(text)[:512])[0]
    score = result["score"]
    return score if result["label"] == "POSITIVE" else -score


def score_posts(df: pd.DataFrame, use_transformer: bool = True) -> pd.DataFrame:
    """
    Adds sentiment columns to a posts DataFrame.

    Columns added:
        vader_compound, vader_pos, vader_neu, vader_neg
        transformer_score (if use_transformer=True)
        combined_text (title + selftext)
    """
    df = df.copy()
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["selftext"].fillna("")
    ).str.strip()

    # VADER scores (fast)
    vader_results = df["combined_text"].apply(vader_score)
    df["vader_compound"] = vader_results.apply(lambda x: x["compound"])
    df["vader_pos"] = vader_results.apply(lambda x: x["pos"])
    df["vader_neu"] = vader_results.apply(lambda x: x["neu"])
    df["vader_neg"] = vader_results.apply(lambda x: x["neg"])

    # Transformer scores (slower, batched)
    if use_transformer:
        tqdm.pandas(desc="Transformer scoring")
        df["transformer_score"] = df["combined_text"].progress_apply(transformer_score)

    return df


def aggregate_film_sentiment(df: pd.DataFrame) -> dict:
    """
    Collapse post-level sentiment to film-level features.

    Weights high-engagement posts (score * num_comments) more heavily,
    since viral posts have outsized influence on public perception.
    """
    if df.empty:
        return {
            "mean_vader": 0.0,
            "mean_transformer": 0.0,
            "weighted_sentiment": 0.0,
            "sentiment_std": 0.0,
            "pct_positive": 0.0,
        }

    weights = np.log1p(df["score"] * df["num_comments"] + 1)
    weights = weights / weights.sum()

    sentiment_col = (
        "transformer_score" if "transformer_score" in df.columns else "vader_compound"
    )

    return {
        "mean_vader": df["vader_compound"].mean(),
        "mean_transformer": df.get("transformer_score", df["vader_compound"]).mean(),
        "weighted_sentiment": (df[sentiment_col] * weights).sum(),
        "sentiment_std": df[sentiment_col].std(),
        "pct_positive": (df[sentiment_col] > 0.1).mean(),
    }


if __name__ == "__main__":
    # Quick smoke test without Reddit API
    test_texts = [
        {"title": "Oppenheimer looks absolutely incredible", "selftext": "Can't wait for this masterpiece", "score": 1200, "num_comments": 340, "upvote_ratio": 0.97},
        {"title": "I'm so tired of superhero movies", "selftext": "This looks like more of the same", "score": 50, "num_comments": 12, "upvote_ratio": 0.61},
    ]
    df = pd.DataFrame(test_texts)
    scored = score_posts(df, use_transformer=False)
    print(scored[["title", "vader_compound"]])
    print(aggregate_film_sentiment(scored))
