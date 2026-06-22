"""
reddit_collector.py
--------------------
Pulls pre-release Reddit posts and comments for a given film title.
Searches r/movies, r/boxoffice, r/flicks in the 30-day window before release.
"""

import os
import praw
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "hype-vs-boxoffice/1.0"),
    )


SUBREDDITS = ["movies", "boxoffice", "flicks", "MovieSuggestions"]


def collect_pre_release_posts(title: str, release_date: str, days_before: int = 30) -> pd.DataFrame:
    """
    Fetch Reddit posts mentioning a film title in the window before its release.

    Args:
        title: Film title to search
        release_date: 'YYYY-MM-DD' string
        days_before: How many days before release to look back

    Returns:
        DataFrame with columns: post_id, subreddit, title, selftext,
                                score, num_comments, created_utc, url
    """
    reddit = get_reddit_client()
    release_dt = datetime.strptime(release_date, "%Y-%m-%d")
    cutoff_dt = release_dt - timedelta(days=days_before)

    records = []
    for sub in SUBREDDITS:
        results = reddit.subreddit(sub).search(
            query=title,
            sort="top",
            time_filter="year",
            limit=100,
        )
        for post in results:
            post_dt = datetime.utcfromtimestamp(post.created_utc)
            if cutoff_dt <= post_dt <= release_dt:
                records.append({
                    "post_id": post.id,
                    "subreddit": sub,
                    "title": post.title,
                    "selftext": post.selftext,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "upvote_ratio": post.upvote_ratio,
                    "created_utc": post_dt.isoformat(),
                    "url": post.url,
                })

    df = pd.DataFrame(records).drop_duplicates("post_id")
    print(f"[{title}] Collected {len(df)} posts across {len(SUBREDDITS)} subreddits")
    return df


def compute_engagement_features(df: pd.DataFrame) -> dict:
    """
    Aggregate post-level data into film-level engagement signals.

    Returns dict of features:
        - total_posts: raw volume of discussion
        - total_comments: total comment count
        - avg_upvote_ratio: how well-received posts were
        - score_velocity: avg daily score over collection window
        - comment_to_post_ratio: engagement depth indicator
    """
    if df.empty:
        return {k: 0 for k in [
            "total_posts", "total_comments", "avg_upvote_ratio",
            "score_velocity", "comment_to_post_ratio"
        ]}

    df["created_utc"] = pd.to_datetime(df["created_utc"])
    days_span = max((df["created_utc"].max() - df["created_utc"].min()).days, 1)

    return {
        "total_posts": len(df),
        "total_comments": df["num_comments"].sum(),
        "avg_upvote_ratio": df["upvote_ratio"].mean(),
        "score_velocity": df["score"].sum() / days_span,
        "comment_to_post_ratio": df["num_comments"].sum() / len(df),
    }


if __name__ == "__main__":
    # Quick test
    df = collect_pre_release_posts("Oppenheimer", "2023-07-21")
    print(df.head())
    print(compute_engagement_features(df))
