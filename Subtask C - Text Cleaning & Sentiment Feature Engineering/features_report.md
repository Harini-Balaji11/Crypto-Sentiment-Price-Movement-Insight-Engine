# Subtask C — Text Cleaning & Sentiment Feature Engineering Report
**Project:** Crypto Sentiment & Price Movement Insight Engine
**Author:** Harini Balaji
**Depends on:** Subtask B (Ingestion Pipeline)

---

## 1. What This Script Does

Subtask C takes the aligned dataset from Subtask B and makes it
ML-ready by doing two things:

1. **Deeper text cleaning + VADER scoring** — re-processes every tweet
   with a more powerful sentiment scorer built for social media
2. **Feature engineering** — creates 11 new features capturing
   sentiment trends, momentum, and price behaviour over time

**Input → Output:**
```
data/processed/btc_eth_aligned.csv  ──┐
                                       ├── features.py ──► data/features/feature_matrix.csv
data/raw/sentiment_raw.csv          ──┘
```

---

## 2. Step-by-Step Process

### Step 1 — Deep text cleaning
Each tweet was cleaned with the following operations in order:

| Operation | Example before | Example after |
|-----------|---------------|---------------|
| Fix encoding | `ð¡DÃ©couvrez` | `🔡Découvrez` |
| Remove URLs | `check https://t.co/abc123` | `check` |
| Remove @mentions | `@Bitcoin is rising` | `is rising` |
| Remove # symbol | `#Bitcoin hits ATH` | `Bitcoin hits ATH` |
| Lowercase | `BULLISH on BTC` | `bullish on btc` |
| Remove extra spaces | `bullish  on  btc` | `bullish on btc` |

**Result:** 4,182 tweets cleaned and ready for VADER scoring.

### Step 2 — VADER sentiment scoring
VADER (Valence Aware Dictionary and sEntiment Reasoner) was applied
to every cleaned tweet. VADER was specifically designed for social
media text and understands:
- Capitalisation: "GREAT" scores higher than "great"
- Punctuation: "great!!!" scores higher than "great"
- Crypto slang and emoji context

**VADER compound score range:** -1.0 (very negative) to +1.0 (very positive)

**Agreement with original volunteer labels:** 62.0%
The 38% disagreement is expected — VADER is general-purpose while
original labels were applied by humans with crypto context. Both
signals are retained as separate features for the model to use.

### Step 3 — Daily VADER aggregation
VADER scores were aggregated per coin per day:
- BTC: 581 days with VADER scores
- ETH: 507 days with VADER scores

### Step 4 — Rolling feature engineering
11 new features were engineered. All rolling calculations were
performed per coin separately to prevent BTC data bleeding into ETH.

---

## 3. All Features in the Output

### Sentiment features (from original labels)

| Feature | Description |
|---------|-------------|
| `avg_score` | Average sentiment score (1=Neg, 2=Neu, 3=Pos) |
| `pct_positive` | Fraction of tweets that were positive (0–1) |
| `pct_negative` | Fraction of tweets that were negative (0–1) |
| `pct_neutral` | Fraction of tweets that were neutral (0–1) |
| `tweet_count` | Number of tweets about this coin that day |
| `sentiment_gap` | True = no tweets that day (gap-filled) |

### VADER features (new in Subtask C)

| Feature | Description |
|---------|-------------|
| `vader_mean` | Average VADER compound score (-1 to +1) |
| `vader_std` | Spread of VADER scores that day (crowd disagreement) |
| `vader_pos_mean` | Average positive component score |
| `vader_neg_mean` | Average negative component score |

### Engineered rolling features (new in Subtask C)

| Feature | Window | Description |
|---------|--------|-------------|
| `sentiment_7d_avg` | 7 days | Rolling average sentiment — captures the trend |
| `sentiment_7d_std` | 7 days | Rolling sentiment volatility — captures uncertainty |
| `sentiment_momentum` | 1 day | Today minus yesterday — detects sharp mood shifts |
| `volume_7d_avg` | 7 days | Rolling tweet volume — is interest growing or fading? |
| `price_return` | 1 day | Daily % price change |
| `price_7d_volatility` | 7 days | Rolling price volatility |
| `price_7d_momentum` | 7 days | Price change over past 7 days |

---

## 4. Output Dataset

**File:** `data/features/feature_matrix.csv`

| Property | Value |
|----------|-------|
| Total rows | 2,282 |
| Total columns | 24 |
| Coins | BTC, ETH |
| Days per coin | 1,141 |
| Date range | 2019-12-12 → 2023-01-25 |

### Per-coin statistics

| Metric | BTC | ETH |
|--------|-----|-----|
| Avg VADER score | 0.245 | 0.241 |
| Avg original score | 2.414 | 2.443 |
| Avg daily tweets | 0.9 | 0.7 |
| Avg daily price return | +0.17% | +0.34% |

### Null values explanation

| Column | Nulls | Reason |
|--------|-------|--------|
| `avg_score`, `pct_*` | 276 | Gap days with no tweets + forward-fill limit exceeded |
| `vader_mean`, `vader_*` | 1,194 | Days with no tweet text to score |
| `sentiment_momentum` | 381 | First row per coin + rows after long gaps |
| `price_return` | 2 | First row per coin (no previous day) |

These nulls will be handled in Subtask D during model preparation
(rows with NaN in key feature columns will be dropped or imputed).

---

## 5. Key Findings

**Finding 1 — Both coins show mild overall positivity**
Average VADER scores of 0.245 (BTC) and 0.241 (ETH) indicate the
crypto Twitter crowd was slightly optimistic across the full period,
even through the 2022 bear market.

**Finding 2 — VADER and original labels capture different signals**
62% agreement means both features carry independent information.
Using both in the model gives it two complementary sentiment views.

**Finding 3 — Sparse tweet volume is confirmed**
Average of less than 1 tweet per coin per day confirms the Subtask A
finding that daily aggregation is the correct granularity.

**Finding 4 — ETH outperformed BTC on returns**
Average daily return of +0.34% (ETH) vs +0.17% (BTC) over this
period. This is a real pattern — ETH grew faster in 2020-2021.

---

## 6. Files Produced

| File | Description |
|------|-------------|
| `src/features.py` | The feature engineering script |
| `data/features/feature_matrix.csv` | Final feature matrix (Subtask D input) |
| `reports/features_report.md` | This document |

---

## 7. Next Step — Subtask D

`feature_matrix.csv` is the direct input for Subtask D (Baseline
Price Movement Classification Model), where we will:
- Define the target label: next-day price direction (Up/Down/Flat)
- Split data into train/test sets using walk-forward validation
- Train Logistic Regression, Random Forest, and XGBoost classifiers
- Evaluate with accuracy, F1 score, and confusion matrix
