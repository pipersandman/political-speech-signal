# dashboard.py
# Trump Truth Social Rhetoric — interactive dashboard

import os, glob, json
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Trump Truth Social Rhetoric", layout="wide")

# ---------- locate results ----------
# Prefer the canonical pipeline output; if missing, fall back to newest results/<stamp>/
PREFERRED = r"pipeline/oneoff/out/post_labels.csv"
candidates = [PREFERRED] + sorted(glob.glob("results/*/post_labels.csv"), reverse=True)
csv_path = next((p for p in candidates if os.path.exists(p)), None)

if not csv_path:
    st.error("Could not find post_labels.csv. Expected at pipeline/oneoff/out/ or results/<stamp>/")
    st.stop()

st.sidebar.success(f"Loaded: {csv_path}")

# ---------- load data ----------
df = pd.read_csv(csv_path)

# Normalize date column (our pipeline writes 'created_at'; sometimes 'date')
date_col = "created_at" if "created_at" in df.columns else ("date" if "date" in df.columns else None)
if not date_col:
    st.error("No 'created_at' or 'date' column in CSV.")
    st.stop()

# Parse timestamps, coerce and drop blanks
df["created_at"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
df = df.dropna(subset=["created_at"]).sort_values("created_at")
df["date"] = df["created_at"].dt.date

# Category columns from the pipeline
CATS = [
    "derogatory_name_calling",
    "violent_rhetoric",
    "angry_rhetoric",
    "positive_rhetoric",
    "dehumanization",
    "hyperbole",
    "divisive_labeling",
    "call_for_unity",
    "call_for_legal_action",
]

# Ensure columns exist and are numeric 0/1
for c in CATS:
    if c not in df.columns:
        df[c] = 0
    # coerce truthy/strings to 0/1
    df[c] = df[c].astype(int)

# Optional target columns produced by adjudicator
for opt in ["who_is_praised", "who_is_criticized", "targets_positive", "targets_negative"]:
    if opt not in df.columns:
        df[opt] = ""

# ---------- sidebar filters ----------
min_date, max_date = df["date"].min(), df["date"].max()
st.sidebar.header("Filters")
start, end = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(start, tuple) or isinstance(end, tuple):  # Streamlit quirk
    start, end = start[0], end[0]
mask = (df["date"] >= start) & (df["date"] <= end)
df = df.loc[mask].copy()

st.title("Truth Social Rhetoric — Trump")

# ---------- top metrics ----------
colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Posts analyzed", f"{len(df):,}")
colB.metric("% Anger", f"{100*df['angry_rhetoric'].mean():.1f}%")
colC.metric("% Violent", f"{100*df['violent_rhetoric'].mean():.1f}%")
colD.metric("% Hyperbole", f"{100*df['hyperbole'].mean():.1f}%")
colE.metric("% Calls for Unity", f"{100*df['call_for_unity'].mean():.1f}%")

# ---------- time series ----------
daily = (
    df.groupby("date")[["angry_rhetoric","violent_rhetoric","positive_rhetoric","call_for_unity"]]
    .mean()
    .reset_index()
)
fig_ts = px.line(
    daily,
    x="date",
    y=["angry_rhetoric","violent_rhetoric","positive_rhetoric","call_for_unity"],
    labels={"value":"Proportion of posts", "variable":"Rhetoric"},
    title="Rhetoric over time",
)
st.plotly_chart(fig_ts, width="stretch")


# ---------- monthly rollups ----------
st.subheader("Monthly Rhetoric Trends")

# Group data by month
monthly = (
    df.groupby(pd.Grouper(key="created_at", freq="ME"))[
        ["angry_rhetoric","violent_rhetoric","positive_rhetoric","call_for_unity"]
    ]
    .mean()
    .reset_index()
)

# Create line chart of monthly averages
fig_month = px.line(
    monthly,
    x="created_at",
    y=["angry_rhetoric","violent_rhetoric","positive_rhetoric","call_for_unity"],
    labels={"value":"Proportion of posts", "variable":"Rhetoric Type"},
    title="Monthly Averages of Rhetoric Types",
)
st.plotly_chart(fig_month, width="stretch")

# ---------- event annotations ----------
st.subheader("Key Events Overlay")

# Define events you want to mark (edit these as you like)
events = [
    {"date": "2025-01-20", "event": "Inauguration Day"},
    {"date": "2025-03-15", "event": "Midterm Announcement"},
    {"date": "2025-04-15", "event": "Tax Day / Economic Rally"},
    {"date": "2025-06-27", "event": "Presidential Debate"},
    {"date": "2025-08-15", "event": "Court Ruling / DOJ Decision"},
]

fig_month_events = px.line(
    monthly,
    x="created_at",
    y=["angry_rhetoric","violent_rhetoric","positive_rhetoric","call_for_unity"],
    labels={"value":"Proportion of posts", "variable":"Rhetoric Type"},
    title="Rhetoric vs. Key Political Events",
)

# Add dotted lines and labels for each event
for e in events:
    fig_month_events.add_vline(x=e["date"], line_dash="dot", line_color="gray")
    fig_month_events.add_annotation(
        x=e["date"],
        y=1,
        text=e["event"],
        showarrow=False,
        yanchor="bottom",
        font=dict(size=10),
        textangle=-90,
    )

st.plotly_chart(fig_month_events, width="stretch")


# ---------- tone balance index ----------
df["tone_index"] = (df["positive_rhetoric"] + df["call_for_unity"]) - (
    df["angry_rhetoric"] + df["violent_rhetoric"] + df["dehumanization"] + df["divisive_labeling"]
)
ti = df.groupby("date")["tone_index"].mean().reset_index()
fig_ti = px.line(ti, x="date", y="tone_index", title="Tone Balance Index (Positive vs Aggressive)")
st.plotly_chart(fig_ti, width="stretch")

# ---------- targets (simple heuristic from text columns if present) ----------
# We'll expand later with entity extraction; for now try 'who_is_praised'/'who_is_criticized' lists if present.
def explode_targets(col):
    # handle JSON-like or comma strings
    vals = []
    for x in df[col].fillna(""):
        if isinstance(x, str) and x.strip():
            try:
                maybe = json.loads(x)
                if isinstance(maybe, list):
                    vals.extend([str(v).strip() for v in maybe if str(v).strip()])
                else:
                    vals.extend([t.strip() for t in x.split(",") if t.strip()])
            except Exception:
                vals.extend([t.strip() for t in x.split(",") if t.strip()])
    return pd.Series(vals)

neg_targets = explode_targets("who_is_criticized")
pos_targets = explode_targets("who_is_praised")

if not neg_targets.empty:
    top_neg = neg_targets.value_counts().head(12).reset_index(names=["target","mentions"])
    fig_neg = px.bar(top_neg, x="mentions", y="target", orientation="h",
                     title="Most criticized targets", labels={"mentions":"Mentions","target":"Target"})
    st.plotly_chart(fig_neg, width="stretch")

if not pos_targets.empty:
    top_pos = pos_targets.value_counts().head(12).reset_index(names=["target","mentions"])
    fig_pos = px.bar(top_pos, x="mentions", y="target", orientation="h",
                     title="Most praised targets", labels={"mentions":"Mentions","target":"Target"})
    st.plotly_chart(fig_pos, width="stretch")

# ---------- drilldown table ----------
st.subheader("Browse posts")
show_cols = ["created_at","text"] + CATS + ["tone_index"]
st.dataframe(df[show_cols].sort_values("created_at", ascending=False), width="stretch")
