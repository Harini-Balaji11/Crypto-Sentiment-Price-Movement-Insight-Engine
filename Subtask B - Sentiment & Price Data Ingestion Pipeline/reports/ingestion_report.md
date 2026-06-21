# Subtask B — Ingestion Pipeline Report
**Project:** Crypto Sentiment & Price Movement Insight Engine
**Author:** Harini Balaji
**Depends on:** Subtask A (Scope & Data Audit)

---

## 1. What This Pipeline Does

Subtask B builds the data foundation for every subtask that follows.
It takes two separate raw data sources — a sentiment CSV and Yahoo
Finance price data — cleans them individually, and joins them into one
single aligned dataset keyed by coin + date.


## 2. Pipeline Steps

### Step 1 — Clean the sentiment CSV
- Loaded 4,267 raw tweets
- Removed 85 duplicate tweets (same tweet scraped by multiple volunteers)
- Fixed encoding corruption on ~61% of tweets using the `ftfy` library
  (garbled characters like `ð¡` and `Ã©` restored to correct UTF-8)
- Parsed all timestamps to UTC datetime format
- Added a `coin` column by matching tweet text against coin keywords:
  - BTC: `#bitcoin`, `#btc`, `$btc` → 975 tweets matched
  - ETH: `#ethereum`, `#eth`, `$eth` → 754 tweets matched
- Final clean dataset: **4,182 unique tweets**

### Step 2 — Fetch price data from Yahoo Finance
- Source: Yahoo Finance via the `yfinance` Python library
- No API key required — completely free
- Downloaded daily OHLCV (Open, High, Low, Close, Volume) for:
  - BTC-USD: 1,141 days (2019-12-12 → 2023-01-25)
  - ETH-USD: 1,141 days (2019-12-12 → 2023-01-25)
- Date range matches the sentiment data window exactly

### Step 3 — Aggregate sentiment to daily level
- Grouped tweets by coin + date
- Computed per-day features:
  - `tweet_count` — number of tweets about that coin that day
  - `avg_score` — average sentiment score (1=Negative, 2=Neutral, 3=Positive)
  - `pct_positive` — fraction of tweets that were positive
  - `pct_negative` — fraction of tweets that were negative
  - `pct_neutral` — fraction of tweets that were neutral
- Result: BTC had **581 days** with sentiment data, ETH had **507 days**

### Step 4 — Join sentiment + price
- Left join from price data (price is the anchor — exists for every trading day)
- Days with no tweets are flagged with `sentiment_gap = True`
- Gap days have `tweet_count = 0`
- Sentiment features on gap days are forward-filled for up to 3 consecutive
  days (last observation carried forward — standard time-series practice)
- Gaps longer than 3 days remain as NaN — too uncertain to impute

---

## 3. Output Dataset

**File:** `data/processed/btc_eth_aligned.csv`

| Property | Value |
|----------|-------|
| Total rows | 2,282 |
| Coins covered | BTC, ETH |
| Days per coin | 1,141 |
| Date range | 2019-12-12 → 2023-01-25 |
| BTC gap days | 560 (49.1% of days) |
| ETH gap days | 634 (55.6% of days) |

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Trading date (YYYY-MM-DD) |
| `coin` | string | BTC or ETH |
| `open` | float | Opening price (USD) |
| `high` | float | Highest price that day (USD) |
| `low` | float | Lowest price that day (USD) |
| `close` | float | Closing price (USD) |
| `volume` | float | Trading volume (USD) |
| `tweet_count` | int | Number of tweets about this coin that day |
| `avg_score` | float | Average sentiment score (1–3 scale) |
| `pct_positive` | float | Fraction of tweets that were positive (0–1) |
| `pct_negative` | float | Fraction of tweets that were negative (0–1) |
| `pct_neutral` | float | Fraction of tweets that were neutral (0–1) |
| `sentiment_gap` | bool | True = no tweets found that day (gap-filled) |

**Sample rows:**

| date | coin | close | tweet_count | avg_score | sentiment_gap |
|------|------|-------|-------------|-----------|---------------|
| 2019-12-12 | BTC | 7243.13 | 2 | 2.00 | False |
| 2019-12-13 | BTC | 7269.68 | 0 | 2.00 | True |
| 2019-12-14 | BTC | 7124.67 | 0 | 2.00 | True |

---

## 4. Design Decisions

**Why Yahoo Finance instead of CoinGecko?**
CoinGecko's free API tier now restricts historical data beyond 365 days.
Yahoo Finance (via `yfinance`) provides full historical OHLCV data for
crypto assets at no cost and with no API key requirement.

**Why daily granularity?**
As identified in Subtask A, the median tweet count per coin per day is 1.
Hourly aggregation would produce features based on 0 tweets for most
time buckets, which is statistically meaningless. Daily aggregation
pools enough signal to compute reliable averages.

**Why left join from price data?**
Price data has an entry for every trading day. Sentiment data has gaps.
Joining from price as the left anchor ensures no trading day is lost —
we simply flag missing sentiment rather than dropping the row.

**Why forward-fill sentiment for gaps?**
A gap day does not mean sentiment was neutral — it means we have no
data. Carrying the most recent known sentiment forward (up to 3 days)
is the standard "last observation carried forward" (LOCF) technique for
sparse time series. The `sentiment_gap` flag preserves transparency —
the model in Subtask D can treat gap rows differently if needed.

---

## 5. Known Limitations

- **High gap rate:** 49–56% of days have no tweets. This is a data
  collection limitation, not a pipeline error. Documented in Subtask A.
- **Forward-fill ceiling:** Gaps longer than 3 days remain as NaN and
  will need to be handled by the model (e.g. drop rows or use a
  separate "no data" feature value).
- **Coin detection by keyword:** Tweets mentioning both BTC and ETH
  are counted in both coins' datasets. This is intentional — a tweet
  comparing BTC and ETH is relevant to both.

---

## 6. Files Produced

| File | Description |
|------|-------------|
| `src/ingestion.py` | The full pipeline script |
| `data/processed/btc_eth_aligned.csv` | The joined dataset |
| `reports/ingestion_report.md` | This document |

---

## 7. Next Step — Subtask C

The output file `btc_eth_aligned.csv` is the input for Subtask C
(Text Cleaning & Sentiment Feature Engineering), where we will:
- Re-run deeper text cleaning on the tweet text
- Apply VADER sentiment scoring to verify/extend the existing labels
- Engineer additional features (sentiment volatility, rolling averages)
- Prepare the final feature matrix for the ML model in Subtask D
