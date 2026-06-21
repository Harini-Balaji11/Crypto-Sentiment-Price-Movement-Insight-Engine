# src/dashboard.py
# Subtask F — Crypto Sentiment & Signal Dashboard
# Run with: streamlit run src/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Sentiment Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
FEATURE_MATRIX = Path("data/features/feature_matrix.csv")
PLOTS_DIR      = Path("reports/plots")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
THRESHOLD = 0.01
FEATURE_COLS = [
    "avg_score", "pct_positive", "pct_negative", "tweet_count",
    "vader_mean", "vader_std", "sentiment_7d_avg", "sentiment_7d_std",
    "sentiment_momentum", "volume_7d_avg", "price_return",
    "price_7d_volatility", "price_7d_momentum",
]

COLORS = {
    "BTC": "#F7931A",   # Bitcoin orange
    "ETH": "#627EEA",   # Ethereum purple
    "Up":   "#00C896",
    "Down": "#FF4B4B",
    "Flat": "#AAAAAA",
    "bg":   "#0E1117",
    "card": "#1E2130",
}

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and label the feature matrix. Cached so it only runs once."""
    df = pd.read_csv(FEATURE_MATRIX)
    df["date"] = pd.to_datetime(df["date"])

    result_frames = []
    for coin in ["BTC", "ETH"]:
        coin_df = df[df["coin"] == coin].copy()
        coin_df = coin_df.sort_values("date").reset_index(drop=True)
        coin_df["next_return"] = coin_df["price_return"].shift(-1)

        def label_direction(ret):
            if pd.isna(ret): return None
            elif ret > THRESHOLD: return "Up"
            elif ret < -THRESHOLD: return "Down"
            else: return "Flat"

        coin_df["label"] = coin_df["next_return"].apply(label_direction)
        result_frames.append(coin_df)

    return pd.concat(result_frames, ignore_index=True).dropna(subset=["label"])


@st.cache_data
def train_models(df_serializable):
    """Train models and return predictions. Cached after first run."""
    df = pd.read_json(df_serializable)
    df["date"] = pd.to_datetime(df["date"])

    all_preds = {}

    for coin in ["BTC", "ETH"]:
        coin_df = df[df["coin"] == coin].copy()
        coin_df = coin_df.sort_values("date").reset_index(drop=True)
        coin_df = coin_df.dropna(subset=FEATURE_COLS + ["label"])

        X = coin_df[FEATURE_COLS]
        y = coin_df["label"]
        dates = coin_df["date"]

        split_idx = int(len(coin_df) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        test_dates = dates.iloc[split_idx:]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        if coin == "BTC":
            # Tuned XGBoost (best from Subtask E)
            model = XGBClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.1, subsample=0.8,
                random_state=42, eval_metric="mlogloss", verbosity=0
            )
            model.fit(X_train, y_train_enc)
            y_pred = le.inverse_transform(model.predict(X_test))
        else:
            # Stacked Ensemble (best from Subtask E)
            base = [
                ("lr", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")),
                ("xgb", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss", verbosity=0)),
            ]
            model = StackingClassifier(estimators=base,
                                       final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                                       cv=3, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        all_preds[coin] = pd.DataFrame({
            "date":      test_dates.values,
            "actual":    y_test.values,
            "predicted": y_pred,
            "correct":   y_test.values == y_pred,
        })

    return all_preds


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=40)
    st.sidebar.title("Crypto Sentiment Engine")
    st.sidebar.markdown("---")

    coin = st.sidebar.selectbox("Select Coin", ["BTC", "ETH"])

    coin_df = df[df["coin"] == coin]
    min_date = coin_df["date"].min().date()
    max_date = coin_df["date"].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project:** Crypto Sentiment & Price Movement Insight Engine")
    st.sidebar.markdown("**Author:** Harini Balaji")
    st.sidebar.markdown("**Subtask:** F — Dashboard")

    return coin, date_range


# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
def page_overview(df):
    st.title("📊 Crypto Sentiment & Price Movement Engine")
    st.markdown("*Linking social sentiment signals to cryptocurrency price movements*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Coins Covered", "BTC + ETH")
    with col3:
        st.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")
    with col4:
        st.metric("Features Used", "13")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What this project does")
        st.markdown("""
        This engine connects **Twitter/X sentiment** with **cryptocurrency price data**
        to predict next-day price direction (Up / Down / Flat).

        Built across 8 subtasks:
        - **A** — Scope & data audit
        - **B** — Data ingestion pipeline
        - **C** — NLP feature engineering
        - **D** — Baseline ML models
        - **E** — Advanced modelling & SHAP
        - **F** — This dashboard
        - **G** — Error analysis
        - **H** — Documentation
        """)

    with col2:
        st.subheader("Label distribution")
        label_counts = df.groupby(["coin", "label"]).size().reset_index(name="count")
        fig = px.bar(label_counts, x="label", y="count", color="coin",
                     barmode="group",
                     color_discrete_map={"BTC": COLORS["BTC"], "ETH": COLORS["ETH"]},
                     title="Up / Down / Flat label counts per coin")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key findings from modelling")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BTC Best F1 (macro)", "0.422", delta="+0.016 vs baseline",
                  help="Tuned XGBoost — Subtask E")
    with col2:
        st.metric("ETH Best F1 (macro)", "0.310", delta="+0.046 vs baseline",
                  help="Stacked Ensemble — Subtask E")
    with col3:
        st.metric("Random baseline", "0.333",
                  help="3-class random guessing gives 33.3%")


# ── PAGE 2: PRICE & SENTIMENT TRENDS ─────────────────────────────────────────
def page_trends(df, coin, date_range):
    st.title(f"📈 Price & Sentiment Trends — {coin}")
    st.markdown("---")

    # Filter
    coin_df = df[df["coin"] == coin].copy()
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        coin_df = coin_df[(coin_df["date"] >= start) & (coin_df["date"] <= end)]

    if coin_df.empty:
        st.warning("No data for selected range.")
        return

    # Dual-axis chart: price + sentiment
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Closing Price (USD)", "7-Day Average Sentiment Score"),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=coin_df["date"], y=coin_df["close"],
        name="Close Price",
        line=dict(color=COLORS[coin], width=1.5),
        fill="tozeroy", fillcolor=f"rgba{tuple(list(bytes.fromhex(COLORS[coin][1:])) + [30])}",
    ), row=1, col=1)

    # Sentiment line
    fig.add_trace(go.Scatter(
        x=coin_df["date"], y=coin_df["sentiment_7d_avg"],
        name="7d Avg Sentiment",
        line=dict(color="#00C896", width=1.5),
    ), row=2, col=1)

    # Neutral line at 2.0 (middle of 1-3 scale)
    fig.add_hline(y=2.0, line_dash="dot", line_color="gray",
                  annotation_text="Neutral (2.0)", row=2, col=1)

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Close", f"${coin_df['close'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Avg Sentiment", f"{coin_df['avg_score'].mean():.2f} / 3.0")
    with col3:
        st.metric("Days Shown", f"{len(coin_df):,}")
    with col4:
        pct_pos = coin_df["pct_positive"].mean() * 100
        st.metric("Avg % Positive", f"{pct_pos:.1f}%")


# ── PAGE 3: TWEET VOLUME & VADER ──────────────────────────────────────────────
def page_sentiment(df, coin, date_range):
    st.title(f"💬 Tweet Volume & VADER Sentiment — {coin}")
    st.markdown("---")

    coin_df = df[df["coin"] == coin].copy()
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        coin_df = coin_df[(coin_df["date"] >= start) & (coin_df["date"] <= end)]

    if coin_df.empty:
        st.warning("No data for selected range.")
        return

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Daily Tweet Count", "VADER Compound Score (-1 to +1)"),
        vertical_spacing=0.1,
        row_heights=[0.45, 0.55],
    )

    # Tweet volume bars
    fig.add_trace(go.Bar(
        x=coin_df["date"], y=coin_df["tweet_count"],
        name="Tweet Count",
        marker_color=COLORS[coin], opacity=0.8,
    ), row=1, col=1)

    # VADER compound score
    vader_colors = coin_df["vader_mean"].apply(
        lambda v: COLORS["Up"] if v > 0.05
        else (COLORS["Down"] if v < -0.05 else COLORS["Flat"])
        if pd.notna(v) else COLORS["Flat"]
    )

    fig.add_trace(go.Scatter(
        x=coin_df["date"],
        y=coin_df["vader_mean"],
        name="VADER Score",
        mode="lines",
        line=dict(color="#A78BFA", width=1.5),
    ), row=2, col=1)

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  annotation_text="Neutral", row=2, col=1)
    fig.add_hline(y=0.05, line_dash="dash", line_color=COLORS["Up"],
                  annotation_text="Positive threshold", row=2, col=1)
    fig.add_hline(y=-0.05, line_dash="dash", line_color=COLORS["Down"],
                  annotation_text="Negative threshold", row=2, col=1)

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tweets", f"{coin_df['tweet_count'].sum():,.0f}")
    with col2:
        avg_vader = coin_df["vader_mean"].mean()
        label = "Positive" if avg_vader > 0.05 else ("Negative" if avg_vader < -0.05 else "Neutral")
        st.metric("Avg VADER Score", f"{avg_vader:.3f}", delta=label)
    with col3:
        gap_pct = coin_df["sentiment_gap"].mean() * 100
        st.metric("Gap Days (no tweets)", f"{gap_pct:.1f}%")


# ── PAGE 4: MODEL SIGNALS ─────────────────────────────────────────────────────
def page_signals(df, coin, all_preds):
    st.title(f"🤖 Model Signals — {coin}")
    st.markdown("---")

    if coin not in all_preds:
        st.warning("Model predictions not available.")
        return

    pred_df = all_preds[coin].copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Signal chart
    color_map = {"Up": COLORS["Up"], "Down": COLORS["Down"], "Flat": COLORS["Flat"]}
    pred_df["color"] = pred_df["predicted"].map(color_map)

    fig = go.Figure()

    for signal in ["Up", "Down", "Flat"]:
        mask = pred_df["predicted"] == signal
        fig.add_trace(go.Scatter(
            x=pred_df[mask]["date"],
            y=[signal] * mask.sum(),
            mode="markers",
            name=f"Predicted {signal}",
            marker=dict(
                color=color_map[signal], size=8,
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            customdata=np.stack([
                pred_df[mask]["actual"],
                pred_df[mask]["correct"].astype(str)
            ], axis=1),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Predicted:</b> " + signal + "<br>"
                "<b>Actual:</b> %{customdata[0]}<br>"
                "<b>Correct:</b> %{customdata[1]}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"{coin} — Model Predictions on Test Set (2022–2023)",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(categoryorder="array", categoryarray=["Down", "Flat", "Up"]),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy metrics
    acc = pred_df["correct"].mean()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", f"{acc*100:.1f}%")
    with col2:
        st.metric("Test Rows", len(pred_df))
    with col3:
        correct = pred_df["correct"].sum()
        st.metric("Correct Predictions", f"{correct} / {len(pred_df)}")

    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    labels = ["Down", "Flat", "Up"]
    cm = confusion_matrix(pred_df["actual"], pred_df["predicted"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig_cm = px.imshow(
        cm_df, text_auto=True,
        color_continuous_scale="Blues",
        title=f"{coin} — Actual (rows) vs Predicted (columns)",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig_cm.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ── PAGE 5: FEATURE IMPORTANCE ────────────────────────────────────────────────
def page_importance(coin):
    st.title(f"🔍 Feature Importance — {coin}")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        shap_path = PLOTS_DIR / f"{coin}_shap_importance.png"
        if shap_path.exists():
            st.subheader("SHAP Feature Importance")
            st.markdown("*Mean absolute SHAP value — how much each feature moves the prediction*")
            st.image(str(shap_path), use_column_width=True)
        else:
            st.warning(f"SHAP plot not found at {shap_path}. Run src/advanced_models.py first.")

    with col2:
        perm_path = PLOTS_DIR / f"{coin}_permutation_importance.png"
        if perm_path.exists():
            st.subheader("Permutation Importance")
            st.markdown("*Drop in F1 when each feature is randomly shuffled*")
            st.image(str(perm_path), use_column_width=True)
        else:
            st.warning(f"Permutation plot not found at {perm_path}.")

    st.markdown("---")
    st.markdown("""
    **How to read these charts:**
    - **Blue bars** = sentiment features (from Twitter/X)
    - **Orange bars** = price features (from Yahoo Finance)
    - **Longer bar** = feature had more influence on predictions

    **Key finding:** Sentiment features appear in the top 5 for both coins.
    For ETH, `vader_mean` is the single most important feature — above all price features.
    """)


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    # Sidebar
    coin, date_range = render_sidebar(df)

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Price & Sentiment Trends",
         "Tweet Volume & VADER", "Model Signals", "Feature Importance"],
        index=0,
    )

    # Train models (cached)
    if page == "Model Signals":
        with st.spinner("Training models (first run takes ~1 min)..."):
            all_preds = train_models(df.to_json())
    else:
        all_preds = {}

    # Render selected page
    if page == "Overview":
        page_overview(df)
    elif page == "Price & Sentiment Trends":
        page_trends(df, coin, date_range)
    elif page == "Tweet Volume & VADER":
        page_sentiment(df, coin, date_range)
    elif page == "Model Signals":
        page_signals(df, coin, all_preds)
    elif page == "Feature Importance":
        page_importance(coin)


if __name__ == "__main__":
    main()