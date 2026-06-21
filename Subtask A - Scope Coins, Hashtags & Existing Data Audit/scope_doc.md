# Subtask A — Scope Document
**Project:** Crypto Sentiment & Price Movement Insight Engine  
**Author:** Harini Balaji  
**Date:** 2026  

---

## 1. Coins Selected

| Coin | Full Name | Reason for Selection |
|------|-----------|----------------------|
| BTC | Bitcoin | Highest market cap, largest Twitter/X community, most benchmarked coin in academic sentiment research — results are comparable to published work |
| ETH | Ethereum | Second largest coin with a distinctly different community (developer-heavy vs BTC's retail-heavy audience), which means its sentiment profile is genuinely different — comparing the two makes the project richer |

**Why only two coins?**  
More coins = more complexity for diminishing returns at this stage. BTC and ETH together represent ~60% of total crypto market cap and the vast majority of social media discussion. Starting with two well-understood coins produces cleaner, more explainable results.

---

## 2. Hashtags & Keywords Per Coin

These are the keywords we search for inside tweet text to assign each tweet to a coin. A tweet can belong to both coins if it mentions both.

| Coin | Keywords tracked |
|------|-----------------|
| BTC | `#Bitcoin`, `#BTC`, `$BTC` |
| ETH | `#Ethereum`, `#ETH`, `$ETH` |

**Why not just one hashtag per coin?**  
Different hashtags attract different audiences. `#Bitcoin` is used by mainstream media and retail investors. `#BTC` is preferred by traders and more technical users. Using all three gives broader, more representative coverage. The same logic applies to ETH vs Ethereum.

**Note on coin detection:**  
The source CSV has no coin column — coin labels are *inferred* by searching tweet text for the above keywords. This is standard practice when working with general crypto sentiment data.

---

## 3. Time Range

**Chosen range:** December 2019 → January 2023 (driven by available data)

This range is actually very valuable for a portfolio project because it captures three distinct market regimes:

| Period | Market Condition | Why It Matters |
|--------|-----------------|----------------|
| Dec 2019 – Feb 2020 | Pre-COVID bull market | Baseline normal conditions |
| Mar 2020 | COVID crash | Extreme fear event — tests model under shock |
| Apr 2020 – Nov 2021 | Bull run to all-time highs | Strong upward sentiment trend |
| Dec 2021 – Jan 2023 | Bear market and crash | Prolonged negative sentiment |

A model that performs reasonably across all three regimes is far more credible than one trained on a single market condition.

---

## 4. Aggregation Granularity

**Decision: Daily aggregation (not hourly)**

**Reason:** The dataset has a median of 1 tweet per coin per day, with 84% of BTC days and 89% of ETH days having fewer than 3 tweets. Hourly aggregation would produce feature vectors based on 0 or 1 tweets for most time buckets, which is statistically meaningless. Daily aggregation pools enough signal to compute meaningful averages.

This is an honest limitation of the data and will be documented in the error analysis (Subtask G).

---

## 5. Data Sources

| Source | What It Provides | Status |
|--------|-----------------|--------|
| Volunteer CSV (Sabarish/Nivetha) | 4,267 sentiment-labelled tweets, Dec 2019–Jan 2023 | Available — see audit report |
| CoinGecko API (free, no key) | BTC + ETH daily OHLCV price data | Needs fetch in Subtask B |
| Twitter/X API | Additional tweets | Not needed — existing data sufficient |

**Why skip the Twitter/X API?**  
The existing volunteer CSV covers 3+ years of data. Fetching more tweets would require managing strict rate limits and costs with minimal benefit at this stage. If the model underperforms in Subtask D, adding more tweet data is listed as a first improvement to try.

---

## 6. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Coins | BTC, ETH | High liquidity, distinct communities, comparable to literature |
| Hashtags | 3 per coin | Broader audience coverage than single hashtag |
| Time range | Dec 2019 – Jan 2023 | Driven by existing data; covers 3 market regimes |
| Granularity | Daily | Too sparse for hourly; daily gives enough signal |
| Extra scraping | CoinGecko only | Minimize API usage; existing sentiment data sufficient |
