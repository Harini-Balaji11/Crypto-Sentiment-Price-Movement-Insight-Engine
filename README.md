# Crypto Sentiment & Price Movement Insight Engine

Linking Twitter/X social sentiment signals to cryptocurrency price movements to predict next-day price direction (Up / Down / Flat) for Bitcoin and Ethereum.

## Project Overview

This end-to-end data science project was built across 8 subtasks covering the full ML lifecycle — from raw data audit through feature engineering, model training, SHAP analysis, an interactive dashboard, and honest error analysis.

**Core question:** Does what people say about crypto on Twitter predict what the price will do tomorrow?

**Answer:** Yes, partially. Sentiment features appear in the top 5 most important predictors for BTC. For ETH, VADER sentiment score is the single most important feature — above all price features.

## Results

| Coin | Best Model | Accuracy | F1 Macro |
|------|-----------|----------|----------|
| BTC | Tuned XGBoost | 42.6% | 0.422 |
| ETH | Stacked Ensemble | 41.6% | 0.310 |
| Random baseline | — | 33.3% | 0.333 |

## Tech Stack

- **Data:** Twitter/X sentiment CSV + Yahoo Finance (yfinance)
- **NLP:** VADER sentiment scorer, ftfy for encoding repair
- **ML:** Logistic Regression, Random Forest, XGBoost, Stacked Ensemble
- **Explainability:** SHAP, Permutation Importance
- **Dashboard:** Streamlit + Plotly
- **Language:** Python 3.12

## How to Run

```bash
pip install pandas numpy requests ftfy yfinance vaderSentiment
pip install scikit-learn xgboost shap joblib matplotlib seaborn plotly streamlit

python src/ingestion.py        # Subtask B — clean + fetch data
python src/features.py         # Subtask C — VADER + feature engineering
python src/models.py           # Subtask D — baseline models
python src/advanced_models.py  # Subtask E — tuning + SHAP
python src/error_analysis.py   # Subtask G — error analysis
streamlit run src/dashboard.py # Subtask F — launch dashboard
```

## Project Structure

| Folder | Contents |
|--------|----------|
| Subtask A | Scope document + data audit report |
| Subtask B | Ingestion pipeline + aligned dataset |
| Subtask C | Feature engineering + VADER scoring |
| Subtask D | Baseline ML models + confusion matrices |
| Subtask E | Tuned XGBoost + SHAP + ensemble |
| Subtask F | Streamlit dashboard (5 pages) |
| Subtask G | Error analysis + 3 case studies |
| Subtask H | Full documentation & usage guide |

## Author

**Harini Balaji** — TaskVerse Volunteer
