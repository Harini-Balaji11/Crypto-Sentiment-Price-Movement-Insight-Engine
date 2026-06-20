# src/ingestion.py
# Subtask B — Sentiment & Price Data Ingestion Pipeline
# Uses yfinance (Yahoo Finance) for price data — free, no API key needed

import pandas as pd
import yfinance as yf
import ftfy
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
RAW_SENTIMENT = Path("data/raw/sentiment_raw.csv")
OUTPUT_FILE   = Path("data/processed/btc_eth_aligned.csv")

# ── COIN CONFIG ───────────────────────────────────────────────────────────────
# ticker = the Yahoo Finance symbol for this coin
COINS = {
    "BTC": {
        "ticker":   "BTC-USD",
        "keywords": ["#bitcoin", "#btc", "$btc"]
    },
    "ETH": {
        "ticker":   "ETH-USD",
        "keywords": ["#ethereum", "#eth", "$eth"]
    }
}

# ── STEP 1: CLEAN SENTIMENT DATA ──────────────────────────────────────────────
def load_and_clean_sentiment(filepath):
    print("Loading sentiment CSV...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} rows")

    # Remove duplicate tweets
    before = len(df)
    df = df.drop_duplicates(subset=["Tweet"])
    print(f"  Removed {before - len(df)} duplicate tweets")

    # Fix broken encoding
    print("  Fixing broken encoding (takes ~10 seconds)...")
    df["Tweet"] = df["Tweet"].apply(lambda x: ftfy.fix_text(str(x)))

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    bad = df["Date"].isna().sum()
    if bad > 0:
        print(f"  Dropped {bad} rows with bad dates")
        df = df.dropna(subset=["Date"])

    # Extract date only (no time)
    df["date"] = df["Date"].dt.date

    # Add coin columns by searching tweet text
    tweet_lower = df["Tweet"].str.lower().fillna("")
    for coin, config in COINS.items():
        df[f"is_{coin}"] = tweet_lower.str.contains(
            "|".join(config["keywords"]), regex=True
        )

    df = df.rename(columns={"sentiment_score": "score"})

    print(f"  Clean dataset: {len(df):,} tweets")
    print(f"  BTC tweets: {df['is_BTC'].sum():,}")
    print(f"  ETH tweets: {df['is_ETH'].sum():,}")
    return df


# ── STEP 2: FETCH PRICE DATA FROM YAHOO FINANCE ───────────────────────────────
def fetch_price_data(coin_name, ticker, start_date, end_date):
    """
    Downloads daily OHLCV data from Yahoo Finance.
    yfinance is free and needs no API key at all.
    start_date / end_date come from our sentiment data range.
    """
    print(f"  Fetching {coin_name} price from Yahoo Finance ({ticker})...")

    # Download the data
    # auto_adjust=True means prices are already adjusted for splits
    df = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date),
        interval="1d",        # daily candles
        auto_adjust=True,
        progress=False         # hides the yfinance progress bar
    )

    if df.empty:
        print(f"  ERROR: No data returned for {ticker}")
        return None

    # yfinance returns a multi-level column index — flatten it
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower()
                  for col in df.columns]

    # Reset index so Date becomes a regular column
    df = df.reset_index()

    # Rename Date column and extract just the date part
    df = df.rename(columns={"Date": "date", "Price": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["coin"] = coin_name

    # Keep only the columns we need
    keep = ["date", "coin", "open", "high", "low", "close", "volume"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing]

    print(f"  Got {len(df):,} days of price data")
    print(f"  Range: {df['date'].min()} to {df['date'].max()}")
    return df


# ── STEP 3: AGGREGATE SENTIMENT TO DAILY LEVEL ────────────────────────────────
def aggregate_sentiment_daily(sentiment_df, coin):
    """
    Collapses all tweets for a coin into one row per day.
    Each row summarises what Twitter was saying that day.
    """
    coin_df = sentiment_df[sentiment_df[f"is_{coin}"]].copy()

    daily = coin_df.groupby("date").agg(
        tweet_count  = ("score", "count"),
        avg_score    = ("score", "mean"),
        pct_positive = ("score", lambda x: (x == 3).mean()),
        pct_negative = ("score", lambda x: (x == 1).mean()),
        pct_neutral  = ("score", lambda x: (x == 2).mean()),
    ).reset_index()

    daily["coin"] = coin
    print(f"  {coin}: {len(daily)} days with sentiment data")
    return daily


# ── STEP 4: JOIN SENTIMENT + PRICE ────────────────────────────────────────────
def join_sentiment_and_price(sentiment_daily, price_df, coin):
    """
    Merges daily sentiment with price data.
    Left join from price — keeps every trading day.
    Gap days (no tweets) get forward-filled sentiment.
    """
    # Join on date + coin
    joined = pd.merge(price_df, sentiment_daily, on=["date", "coin"], how="left")

    # Flag days with no tweets
    joined["sentiment_gap"] = joined["tweet_count"].isna()
    joined["tweet_count"] = joined["tweet_count"].fillna(0).astype(int)

    # Forward fill sentiment for gap days (max 3 days)
    joined = joined.sort_values("date")
    for col in ["avg_score", "pct_positive", "pct_negative", "pct_neutral"]:
        joined[col] = joined[col].ffill(limit=3)

    print(f"  {coin}: {len(joined)} total rows")
    print(f"  {coin}: {int(joined['sentiment_gap'].sum())} gap days")
    return joined


# ── STEP 5: MAIN PIPELINE ─────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 55)
    print("SUBTASK B - INGESTION PIPELINE")
    print("=" * 55)

    # Step 1: Clean sentiment
    sentiment_df = load_and_clean_sentiment(RAW_SENTIMENT)

    # Get our date range from the sentiment data
    min_date = sentiment_df["date"].min()
    max_date = sentiment_df["date"].max()
    print(f"\nDate range from sentiment: {min_date} to {max_date}")

    all_frames = []

    for coin, config in COINS.items():
        print(f"\n--- Processing {coin} ---")

        # Step 2: Fetch price
        price_df = fetch_price_data(
            coin,
            config["ticker"],
            start_date=min_date,
            end_date=max_date
        )

        if price_df is None:
            print(f"  SKIPPING {coin} - price fetch failed")
            continue

        # Step 3: Aggregate sentiment
        sentiment_daily = aggregate_sentiment_daily(sentiment_df, coin)

        # Step 4: Join
        joined = join_sentiment_and_price(sentiment_daily, price_df, coin)
        all_frames.append(joined)

    if not all_frames:
        print("\nERROR: No data was fetched. Check your internet connection.")
        return

    # Combine BTC and ETH into one file
    final_df = pd.concat(all_frames, ignore_index=True)
    final_df = final_df.sort_values(["coin", "date"]).reset_index(drop=True)
    final_df["date"] = final_df["date"].astype(str)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    # Validation report
    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Total rows  : {len(final_df):,}")
    print(f"Columns     : {list(final_df.columns)}")
    print(f"\nPer coin breakdown:")
    for coin in COINS:
        rows = final_df[final_df["coin"] == coin]
        if len(rows) > 0:
            gaps = int(rows["sentiment_gap"].sum())
            print(f"  {coin}: {len(rows)} days | {gaps} gap days")
    print(f"\nDate range  : {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"\nFirst 3 rows of output:")
    print(final_df[["date","coin","close","tweet_count","avg_score","sentiment_gap"]].head(3).to_string())
    print("\nSubtask B complete! File saved to data/processed/btc_eth_aligned.csv")


if __name__ == "__main__":
    run_pipeline()