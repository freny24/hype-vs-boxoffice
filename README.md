# 🎬 Does Hype Kill Movies?
### Quantifying the Gap Between Pre-Release Buzz and Box Office Reality

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

Every awards season, a hyped blockbuster bombs. Every summer, a "nobody saw this coming" film dominates. Is pre-release hype actually predictive of box office success — or does it actively set films up to disappoint?

This project builds a full data pipeline to **quantify the hype-to-performance gap** across 500+ films using Reddit sentiment, trailer engagement, and box office data.

---

## Who This Is For

- **Studios & distributors** tracking whether marketing spend converts to opening weekend revenue
- **Analysts** benchmarking campaign effectiveness across genres and release windows
- **Anyone** who's ever watched a trailer 20 times only to leave the theater disappointed

---

## What Was Built

```
Reddit API ──► Sentiment Scoring ──► Feature Engineering ──► Hype Score
                                                                    │
TMDB API ──► Metadata + Trailer Views                               ▼
                                                          Hype vs Performance Model
The Numbers API ──► Box Office Data ──► Performance Score           │
                                                                    ▼
                                                         Streamlit Dashboard
```

**Pipeline stages:**
1. **Data ingestion** — Reddit posts/comments, trailer stats, box office figures via public APIs
2. **Sentiment analysis** — VADER baseline + DistilBERT fine-tuned transformer comparison
3. **Feature engineering** — Rolling hype scores, velocity of buzz, comment-to-upvote ratios
4. **Modeling** — XGBoost to predict opening weekend from pre-release signals
5. **Evaluation** — RMSE, feature importance, SHAP explainability
6. **Dashboard** — Live interactive viz with film search, genre filters, and hype leaderboard

---

## Results

| Metric | Value |
|--------|-------|
| Films analyzed | 512 |
| Prediction RMSE (log revenue) | 0.43 |
| R² score | 0.71 |
| Biggest over-hyped film found | *See dashboard* |
| Biggest under-hyped film found | *See dashboard* |

**Key finding:** Films with the highest pre-release hype scores in the 90th percentile **underperformed** their predicted box office by 34% on average — the "hype ceiling" effect.

---

## Project Structure

```
hype-vs-box-office/
│
├── data/
│   ├── raw/                  # Raw API pulls (gitignored)
│   ├── processed/            # Cleaned, feature-engineered CSVs
│   └── sample_data.csv       # 50-film sample for demo (no API key needed)
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling_and_eval.ipynb
│
├── src/
│   ├── reddit_collector.py   # Reddit API ingestion
│   ├── sentiment.py          # VADER + transformer scoring
│   ├── features.py           # Hype score construction
│   ├── model.py              # XGBoost training + SHAP
│   └── utils.py              # Shared helpers
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── requirements.txt
├── .env.example              # API key template
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/hype-vs-box-office.git
cd hype-vs-box-office

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys (or skip — sample data works without any keys)
cp .env.example .env
# Edit .env with your Reddit and TMDB API keys

# 4. Run the dashboard
streamlit run dashboard/app.py
```

> **No API keys?** The dashboard runs on `data/sample_data.csv` out of the box. All 50 sample films load instantly.

---

## Data Sources

| Source | What It Provides | API |
|--------|-----------------|-----|
| Reddit (PRAW) | Pre-release sentiment, post volume, upvote velocity | Free |
| TMDB | Film metadata, genres, budget, trailer data | Free |
| The Numbers | Box office opening/total revenue | Public scrape |

---

## Skills Demonstrated

`Python` · `NLP / Sentiment Analysis` · `Transformer Models (DistilBERT)` · `XGBoost` · `SHAP Explainability` · `Feature Engineering` · `REST APIs` · `Data Pipeline Design` · `Streamlit` · `Plotly`

---

## Future Work

- [ ] Extend to TV series premiere ratings (Netflix, HBO)
- [ ] Add international box office for global hype comparison
- [ ] Fine-tune sentiment model on movie-specific vocabulary
- [ ] Real-time tracking for upcoming releases

---

## Author

**Freny Reji** · [LinkedIn](https://www.linkedin.com/in/frenyreji-2401) · [GitHub](https://github.com/freny24)

*MS Data Science, Indiana University Bloomington*
