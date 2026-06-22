"""
dashboard/app.py
-----------------
Streamlit dashboard for the Hype vs Box Office project.
Runs on sample data out of the box — no API keys needed.

Deploy to Streamlit Community Cloud:
  1. Push this repo to GitHub
  2. Go to share.streamlit.io
  3. Connect repo, set main file to dashboard/app.py
  4. Done — live URL in 2 minutes
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Does Hype Kill Movies?",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load data ───────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "sample_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["hype_gap"] = df["hype_score"] - df["performance_score"]
    df["opening_m"] = df["opening_weekend"] / 1e6
    df["budget_m"] = df["production_budget"] / 1e6
    df["opening_roi"] = df["opening_weekend"] / df["production_budget"]
    return df

df = load_data()

# ─── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.title("Filters")

genres = ["All"] + sorted(df["genre"].unique().tolist())
selected_genre = st.sidebar.selectbox("Genre", genres)

verdicts = ["All", "Overhyped", "Slightly Overhyped", "As Expected", "Slightly Underrated", "Surprise Hit"]
selected_verdict = st.sidebar.selectbox("Verdict", verdicts)

year_range = st.sidebar.slider(
    "Release year", int(df["year"].min()), int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

hype_range = st.sidebar.slider("Hype score range", 0, 100, (0, 100))

# ─── Filter data ──────────────────────────────────────────────────────────────
mask = (
    df["year"].between(*year_range) &
    df["hype_score"].between(*hype_range)
)
if selected_genre != "All":
    mask &= df["genre"] == selected_genre
if selected_verdict != "All":
    mask &= df["verdict"] == selected_verdict

filtered = df[mask].copy()

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🎬 Does Hype Kill Movies?")
st.markdown(
    "Quantifying the gap between pre-release Reddit buzz and box office reality. "
    f"Showing **{len(filtered)}** of {len(df)} films."
)

# ─── Metric row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Films analyzed", len(filtered))
with col2:
    avg_gap = filtered["hype_gap"].mean()
    st.metric("Avg hype gap", f"{avg_gap:+.1f}", delta_color="inverse")
with col3:
    pct_over = (filtered["hype_gap"] > 5).mean() * 100
    st.metric("Overhyped", f"{pct_over:.0f}%")
with col4:
    pct_hit = (filtered["hype_gap"] < -5).mean() * 100
    st.metric("Surprise hits", f"{pct_hit:.0f}%")
with col5:
    biggest_flop = filtered.loc[filtered["hype_gap"].idxmax(), "title"] if len(filtered) else "—"
    st.metric("Biggest hype gap", biggest_flop[:18] + "…" if len(biggest_flop) > 18 else biggest_flop)

st.divider()

# ─── Scatter: hype vs performance ────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

VERDICT_COLORS = {
    "Overhyped": "#D85A30",
    "Slightly Overhyped": "#EF9F27",
    "As Expected": "#888780",
    "Slightly Underrated": "#378ADD",
    "Surprise Hit": "#1D9E75",
}

with col_a:
    st.subheader("Hype score vs performance score")
    fig_scatter = px.scatter(
        filtered,
        x="hype_score", y="performance_score",
        color="verdict",
        color_discrete_map=VERDICT_COLORS,
        hover_name="title",
        hover_data={"hype_score": ":.1f", "performance_score": ":.1f",
                    "hype_gap": ":.1f", "opening_m": ":.0f"},
        size="opening_m",
        size_max=30,
        labels={"hype_score": "Hype Score", "performance_score": "Performance Score"},
    )
    # Diagonal line — perfect calibration
    fig_scatter.add_shape(
        type="line", x0=0, x1=100, y0=0, y1=100,
        line=dict(color="rgba(128,128,128,0.3)", dash="dot", width=1)
    )
    fig_scatter.add_annotation(
        x=85, y=88, text="Perfect calibration", showarrow=False,
        font=dict(size=10, color="rgba(128,128,128,0.6)"), textangle=-40
    )
    fig_scatter.update_layout(height=400, legend_title="Verdict")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    st.subheader("Verdict distribution")
    verdict_counts = filtered["verdict"].value_counts().reset_index()
    verdict_counts.columns = ["verdict", "count"]
    fig_donut = px.pie(
        verdict_counts, names="verdict", values="count",
        color="verdict", color_discrete_map=VERDICT_COLORS, hole=0.55
    )
    fig_donut.update_layout(height=400, legend_title="Verdict")
    st.plotly_chart(fig_donut, use_container_width=True)

# ─── Top overhyped and surprise hits ──────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Most overhyped films")
    top_over = filtered[filtered["hype_gap"] > 0].nlargest(10, "hype_gap")
    fig_over = px.bar(
        top_over, x="hype_gap", y="title", orientation="h",
        color_discrete_sequence=["#D85A30"],
        labels={"hype_gap": "Hype gap", "title": ""},
        hover_data={"opening_m": ":.0f", "budget_m": ":.0f"},
    )
    fig_over.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_over, use_container_width=True)

with col_d:
    st.subheader("Biggest surprise hits")
    top_hits = filtered[filtered["hype_gap"] < 0].nsmallest(10, "hype_gap")
    top_hits = top_hits.copy()
    top_hits["gap_abs"] = top_hits["hype_gap"].abs()
    fig_hit = px.bar(
        top_hits, x="gap_abs", y="title", orientation="h",
        color_discrete_sequence=["#1D9E75"],
        labels={"gap_abs": "Outperformed hype by", "title": ""},
    )
    fig_hit.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_hit, use_container_width=True)

# ─── Franchise analysis ───────────────────────────────────────────────────────
st.divider()
st.subheader("Franchise vs original — who gets more hype? Who delivers?")

franchise_df = filtered.groupby("is_franchise").agg(
    avg_hype=("hype_score", "mean"),
    avg_perf=("performance_score", "mean"),
    avg_gap=("hype_gap", "mean"),
    count=("title", "count"),
).reset_index()
franchise_df["type"] = franchise_df["is_franchise"].map({1: "Franchise", 0: "Original"})

fig_fran = go.Figure()
for _, row in franchise_df.iterrows():
    fig_fran.add_trace(go.Bar(
        name=row["type"], x=["Avg Hype", "Avg Performance", "Avg Gap"],
        y=[row["avg_hype"], row["avg_perf"], row["avg_gap"]],
        text=[f"{row['avg_hype']:.1f}", f"{row['avg_perf']:.1f}", f"{row['avg_gap']:+.1f}"],
        textposition="outside",
    ))

fig_fran.update_layout(
    barmode="group", height=320,
    legend_title="Film type",
    yaxis_title="Score",
)
st.plotly_chart(fig_fran, use_container_width=True)

# ─── Full data table ──────────────────────────────────────────────────────────
st.divider()
st.subheader("All films")

display_df = filtered[[
    "title", "year", "genre", "hype_score", "performance_score",
    "hype_gap", "opening_m", "budget_m", "verdict"
]].copy()
display_df.columns = [
    "Title", "Year", "Genre", "Hype", "Performance",
    "Gap", "Opening ($M)", "Budget ($M)", "Verdict"
]
display_df["Opening ($M)"] = display_df["Opening ($M)"].round(1)
display_df["Budget ($M)"] = display_df["Budget ($M)"].round(0).astype(int)
display_df = display_df.sort_values("Gap", ascending=False)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Hype": st.column_config.ProgressColumn("Hype", min_value=0, max_value=100, format="%.1f"),
        "Performance": st.column_config.ProgressColumn("Perf", min_value=0, max_value=100, format="%.1f"),
        "Gap": st.column_config.NumberColumn("Gap", format="%+.1f"),
    }
)

st.caption(
    "Hype Score: weighted composite of Reddit engagement, sentiment, and trailer views. "
    "Performance Score: opening weekend ROI + total gross ROI + box office legs. "
    "Gap = Hype − Performance. Positive = overhyped, negative = underrated."
)
