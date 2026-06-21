# src/features.py
# Subtask C — Text Cleaning & Sentiment Feature Engineering
#
# What this script does:
#   1. Re-cleans tweet text more thoroughly
#   2. Runs VADER sentiment scorer on every tweet
#   3. Aggregates VADER scores to daily level per coin
#   4. Engineers rolling features (volatility, momentum, averages)
#   5. Adds price-based features
#   6. Saves the final feature matrix ready for ML in Subtask D

import pandas as pd
import re
import ftfy
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── PATHS ─────────────────────────────────────────────────────────────────────
RAW_SENTIMENT  = Path("data/raw/sentiment_raw.csv")
ALIGNED_DATA   = Path("data/processed/btc_eth_aligned.csv")
OUTPUT_FILE    = Path("data/features/feature_matrix.csv")

# ── COIN KEYWORDS ─────────────────────────────────────────────────────────────
COIN_KEYWORDS = {
    "BTC": ["#bitcoin", "#btc", "$btc"],
    "ETH": ["#ethereum", "#eth", "$eth"]
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DEEP TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_tweet(text):
    """
    Cleans a single tweet for NLP processing.
    We remove things that add noise but no meaning.

    Why each step matters:
    - URLs: "https://t.co/abc123" tells us nothing about sentiment
    - Mentions: "@Bitcoin" is a handle, not a sentiment word
    - Hashtag symbol: "#Bitcoin" → "Bitcoin" (keep the word, drop the #)
    - Extra spaces: cleaning leaves gaps, we tidy them up
    - Lowercase: so "BULLISH" and "bullish" are treated the same
    """
    # Fix encoding first (in case any slipped through)
    text = ftfy.fix_text(str(text))

    # Remove URLs (http://... or https://...)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbol but keep the word (#Bitcoin → Bitcoin)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove numbers on their own (like "1️⃣" remnants or standalone digits)
    text = re.sub(r'\b\d+\b', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Lowercase
    text = text.lower()

    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VADER SENTIMENT SCORING
# ══════════════════════════════════════════════════════════════════════════════
def score_with_vader(tweets_df):
    """
    Runs VADER on every tweet and returns a compound score.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) was built
    specifically for social media text. It understands:
    - Capitalisation: "GREAT" scores higher than "great"
    - Punctuation: "great!!!" scores higher than "great"
    - Emojis: 🚀 is positive, 📉 is negative
    - Slang: "lol", "wtf", crypto terms

    compound score ranges from -1.0 (most negative) to +1.0 (most positive)
    A score > 0.05  = Positive
    A score < -0.05 = Negative
    In between      = Neutral
    """
    print("  Running VADER on all tweets (takes ~30 seconds)...")
    analyzer = SentimentIntensityAnalyzer()

    # Apply VADER to each cleaned tweet
    # The compound score is the most useful single number
    tweets_df["vader_compound"] = tweets_df["clean_tweet"].apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )

    # Also get the individual positive/negative/neutral scores
    tweets_df["vader_pos"] = tweets_df["clean_tweet"].apply(
        lambda text: analyzer.polarity_scores(text)["pos"]
    )
    tweets_df["vader_neg"] = tweets_df["clean_tweet"].apply(
        lambda text: analyzer.polarity_scores(text)["neg"]
    )

    # Derive a VADER-based label (for comparison with original labels)
    def vader_label(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    tweets_df["vader_label"] = tweets_df["vader_compound"].apply(vader_label)

    return tweets_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — AGGREGATE VADER TO DAILY LEVEL
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_vader_daily(tweets_df, coin):
    """
    Collapses VADER scores to one row per day per coin.
    """
    coin_tweets = tweets_df[tweets_df[f"is_{coin}"]].copy()

    daily = coin_tweets.groupby("date").agg(
        vader_mean     = ("vader_compound", "mean"),
        vader_std      = ("vader_compound", "std"),   # spread of opinions that day
        vader_pos_mean = ("vader_pos", "mean"),
        vader_neg_mean = ("vader_neg", "mean"),
        tweet_count    = ("vader_compound", "count"),
    ).reset_index()

    # Fill NaN std (happens when only 1 tweet that day — no spread)
    daily["vader_std"] = daily["vader_std"].fillna(0)
    daily["coin"] = coin

    return daily


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ENGINEER ROLLING FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    """
    Takes the aligned dataset (price + basic sentiment) and adds
    rolling and momentum features for each coin separately.

    We MUST process each coin separately because BTC and ETH rows
    are interleaved in the dataset. If we calculated a 7-day rolling
    average across both, the last BTC row would bleed into the first
    ETH row — completely wrong.
    """
    print("  Engineering rolling features...")
    result_frames = []

    for coin in ["BTC", "ETH"]:
        coin_df = df[df["coin"] == coin].copy()
        coin_df = coin_df.sort_values("date").reset_index(drop=True)

        # ── SENTIMENT FEATURES ────────────────────────────────────────────

        # 7-day rolling average sentiment
        # min_periods=1 means: even if we only have 1 day of data,
        # still compute the average (don't return NaN)
        coin_df["sentiment_7d_avg"] = (
            coin_df["avg_score"]
            .rolling(window=7, min_periods=1)
            .mean()
        )

        # 7-day sentiment volatility (standard deviation)
        # High std = crowd is divided / uncertain
        # Low std  = crowd agrees
        coin_df["sentiment_7d_std"] = (
            coin_df["avg_score"]
            .rolling(window=7, min_periods=2)
            .std()
            .fillna(0)
        )

        # Sentiment momentum = today's score minus yesterday's score
        # Positive momentum = crowd getting more bullish
        # Negative momentum = crowd getting more bearish
        coin_df["sentiment_momentum"] = coin_df["avg_score"].diff(1)

        # 7-day tweet volume trend
        # Are people talking more or less about this coin vs last week?
        coin_df["volume_7d_avg"] = (
            coin_df["tweet_count"]
            .rolling(window=7, min_periods=1)
            .mean()
        )

        # ── PRICE FEATURES ────────────────────────────────────────────────

        # Daily return = % change in closing price from yesterday
        # (today_close - yesterday_close) / yesterday_close
        # This is more useful than raw price — it's scale-independent
        coin_df["price_return"] = coin_df["close"].pct_change(1)

        # 7-day price volatility
        # How much has price been swinging over the past week?
        coin_df["price_7d_volatility"] = (
            coin_df["price_return"]
            .rolling(window=7, min_periods=2)
            .std()
            .fillna(0)
        )

        # Price momentum = 7-day return (close today vs close 7 days ago)
        coin_df["price_7d_momentum"] = coin_df["close"].pct_change(7)

        result_frames.append(coin_df)

    return pd.concat(result_frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_features():
    print("=" * 55)
    print("SUBTASK C - FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    # ── Load aligned dataset from Subtask B ───────────────────────────
    print("\nLoading aligned dataset from Subtask B...")
    aligned_df = pd.read_csv(ALIGNED_DATA)
    print(f"  Loaded {len(aligned_df):,} rows")

    # ── Load raw tweets for VADER scoring ─────────────────────────────
    print("\nLoading raw tweets for VADER scoring...")
    tweets_df = pd.read_csv(RAW_SENTIMENT)
    tweets_df = tweets_df.drop_duplicates(subset=["Tweet"])

    # Parse dates
    tweets_df["Date"] = pd.to_datetime(tweets_df["Date"], utc=True, errors="coerce")
    tweets_df["date"] = tweets_df["Date"].dt.date.astype(str)

    # Add coin flags
    tweet_lower = tweets_df["Tweet"].str.lower().fillna("")
    for coin, keywords in COIN_KEYWORDS.items():
        tweets_df[f"is_{coin}"] = tweet_lower.str.contains(
            "|".join(keywords), regex=True
        )

    # Step 1: Clean tweet text
    print("\nStep 1 — Cleaning tweet text...")
    tweets_df["clean_tweet"] = tweets_df["Tweet"].apply(clean_tweet)
    print(f"  Cleaned {len(tweets_df):,} tweets")

    # Step 2: VADER scoring
    print("\nStep 2 — VADER sentiment scoring...")
    tweets_df = score_with_vader(tweets_df)

    # Compare VADER labels vs original labels
    agreement = (tweets_df["vader_label"] == tweets_df["sentiment_type"]).mean()
    print(f"  VADER agrees with original labels: {agreement*100:.1f}%")

    # Step 3: Aggregate VADER to daily level
    print("\nStep 3 — Aggregating VADER scores by coin + day...")
    vader_frames = []
    for coin in ["BTC", "ETH"]:
        daily_vader = aggregate_vader_daily(tweets_df, coin)
        print(f"  {coin}: {len(daily_vader)} days with VADER scores")
        vader_frames.append(daily_vader)

    vader_daily = pd.concat(vader_frames, ignore_index=True)

    # Step 4: Merge VADER features into aligned dataset
    print("\nStep 4 — Merging VADER features into main dataset...")
    aligned_df["date"] = aligned_df["date"].astype(str)
    vader_daily["date"] = vader_daily["date"].astype(str)

    merged_df = pd.merge(
        aligned_df,
        vader_daily[["date", "coin", "vader_mean", "vader_std",
                     "vader_pos_mean", "vader_neg_mean"]],
        on=["date", "coin"],
        how="left"
    )
    print(f"  Merged dataset: {len(merged_df):,} rows")

    # Step 5: Engineer rolling features
    print("\nStep 5 — Engineering rolling + momentum features...")
    feature_df = engineer_features(merged_df)

    # Step 6: Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_FILE, index=False)

    # ── Validation report ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 55)
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Total rows  : {len(feature_df):,}")
    print(f"Total cols  : {len(feature_df.columns)}")
    print(f"\nAll columns:")
    for col in feature_df.columns:
        null_count = feature_df[col].isna().sum()
        print(f"  {col:<30} nulls: {null_count}")

    print(f"\nPer coin sample stats:")
    for coin in ["BTC", "ETH"]:
        coin_rows = feature_df[feature_df["coin"] == coin]
        print(f"\n  {coin}:")
        print(f"    avg VADER score : {coin_rows['vader_mean'].mean():.3f}")
        print(f"    avg sentiment   : {coin_rows['avg_score'].mean():.3f}")
        print(f"    avg daily tweets: {coin_rows['tweet_count'].mean():.1f}")
        print(f"    avg price return: {coin_rows['price_return'].mean()*100:.2f}%")

    print("\nSubtask C complete! File saved to data/features/feature_matrix.csv")


if __name__ == "__main__":
    run_features()