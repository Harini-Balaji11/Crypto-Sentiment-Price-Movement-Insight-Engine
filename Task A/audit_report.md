# Subtask A — Data Audit Report
**Project:** Crypto Sentiment & Price Movement Insight Engine  
**File audited:** `CryptoCurrencySentiment_Sabarish_Nivetha.csv`  
**Audit run by:** Harini Balaji 

---

## 1. File Overview

| Field | Value |
|-------|-------|
| Total rows | 4,267 tweets |
| Total columns | 5 |
| File size | ~3 MB |
| Date range | 2019-12-12 → 2023-01-26 |
| Timezone | UTC (consistent throughout) |
| Bad / unparseable timestamps | 0 |

**Columns present:**

| Column | Type | Description |
|--------|------|-------------|
| `sentiment_type` | string | `Positive`, `Neutral`, or `Negative` |
| `Date` | datetime (UTC) | When the tweet was posted |
| `User` | string | Twitter username |
| `Tweet` | string | Full tweet text |
| `sentiment_score` | integer | 1 = Negative, 2 = Neutral, 3 = Positive |

---

## 2. Coin Coverage

No coin column exists in the file. Coin labels were inferred by searching tweet text for BTC/ETH keywords.

| Coin | Keywords matched | Tweet count | % of total | Avg tweets/day | Max tweets/day |
|------|-----------------|-------------|------------|----------------|----------------|
| BTC | #bitcoin, #btc, $btc | 977 | 22.9% | 1.7 | 13 |
| ETH | #ethereum, #eth, $eth | 757 | 17.7% | 1.5 | 8 |
| Both coins | — | 321 | 7.5% | — | — |
| Neither (general crypto) | — | 2,854 | 66.9% | — | — |

**Sentiment breakdown — BTC tweets:**
- Neutral: 519 (53.1%)
- Positive: 483 (49.4%)  
- Negative: 63 (6.4%)

**Sentiment breakdown — ETH tweets:**
- Neutral: 435 (57.5%)
- Positive: 425 (56.1%)
- Negative: 33 (4.4%)

---

## 3. Issues Found

### Issue 1 — Encoding corruption (HIGH impact, fixable)
**What:** 2,603 tweets (61% of the dataset) contain garbled characters caused by a UTF-8 to Latin-1 encoding mismatch during data collection. Emojis appear as sequences like `ð¡` and accented characters appear as `Ã©`.  
**Impact:** Adds noise to NLP features if not fixed. The garbled characters are not meaningful words.  
**Fix:** Apply `ftfy.fix_text()` from the `ftfy` Python library in Subtask C text cleaning. One-line fix.  
**Example of broken tweet:**
```
ð¡DÃ©couvrez le Top 5 des #cryptomonnaies
```
**Should read:**
```
🔡Découvrez le Top 5 des #cryptomonnaies
```

---

### Issue 2 — Sparse daily signal (MEDIUM impact, documented limitation)
**What:** Most days have very few tweets for any given coin.
- BTC: 84% of days with data have fewer than 3 tweets
- ETH: 89% of days with data have fewer than 3 tweets
- Median tweets per day: 1 for both coins

**Impact:** Hourly aggregation is not viable. Daily aggregation is required.  
**Decision:** Aggregate all features at the daily level. Days with zero tweets will be forward-filled (carry yesterday's sentiment) or flagged as NaN depending on the gap length.  
**This is a known limitation** and will be reported honestly in the Subtask G error analysis.

---

### Issue 3 — 124 missing days (LOW impact, not a blocker)
**What:** Out of 1,142 total calendar days in the Dec 2019–Jan 2023 range, 124 days have no tweets at all.  
**Impact:** Creates gaps in the time series. Does not break the pipeline.  
**Fix:** In Subtask B, these are handled by forward-filling sentiment values for gaps of 1–2 days, or flagging as NaN for longer gaps. Standard time-series practice.

---

### Issue 4 — 85 duplicate tweets (LOW impact, easy fix)
**What:** 85 tweet texts appear more than once in the file. Likely caused by multiple volunteers scraping the same tweets.  
**Fix:** `df.drop_duplicates(subset=['Tweet'])` in the Subtask B cleaning step. Reduces dataset to 4,182 unique tweets.

---

## 4. What Is Missing

| Missing Data | Why It's Needed | Action |
|-------------|----------------|--------|
| BTC daily price (OHLCV) | Required to define Up/Down/Flat labels for the ML model | Fetch from CoinGecko in Subtask B |
| ETH daily price (OHLCV) | Same reason | Fetch from CoinGecko in Subtask B |
| Coin column | Needed for clean filtering per coin | Infer from tweet text (solution above) |

---

## 5. Minimal Scrape Plan

Based on this audit, the following is the minimum additional data needed:

| Data needed | Source | API key required? | Priority |
|------------|--------|-------------------|----------|
| BTC OHLCV daily, Dec 2019 → Jan 2023 | CoinGecko free API | No | HIGH |
| ETH OHLCV daily, Dec 2019 → Jan 2023 | CoinGecko free API | No | HIGH |
| Additional sentiment tweets | Twitter/X API | Yes (paid) | LOW — skip for now |

**Why CoinGecko?**  
- Free tier covers unlimited historical daily data
- No API key required for basic requests
- Well-documented, reliable, widely used in academic crypto research
- Returns data in clean JSON format

**Why skip Twitter/X API?**  
The existing CSV provides 3+ years of labelled sentiment data. The Twitter/X Basic API now costs $100/month minimum and is rate-limited. Given the existing data is sufficient for Subtask B, scraping more tweets is deferred unless model performance in Subtask D is below acceptable thresholds.

---

## 6. Audit Summary

| Check | Result |
|-------|--------|
| File readable | PASS |
| Timestamps parseable | PASS — 0 errors |
| BTC data present | PASS — 977 tweets |
| ETH data present | PASS — 757 tweets |
| Price data present | FAIL — not in this file |
| Encoding clean | FAIL — 61% garbled (fixable) |
| Duplicates | 85 found (fixable) |
| Date gaps | 124 missing days (manageable) |

**Overall verdict:** Data is usable. Two fixable issues (encoding + duplicates) will be resolved in Subtask B/C. One structural gap (no price data) will be filled by fetching from CoinGecko in Subtask B.

---

## 7. Next Steps (Subtask B)

1. Remove 85 duplicate tweets
2. Fix encoding on 2,603 tweets using `ftfy`
3. Add `coin` column by matching tweet text to BTC/ETH keywords
4. Fetch BTC + ETH daily OHLCV from CoinGecko (Dec 2019 → Jan 2023)
5. Join sentiment data with price data on `coin` + `date`
6. Handle the 124 missing days with forward-fill strategy
7. Save joined dataset as `data/processed/btc_eth_aligned.csv`
