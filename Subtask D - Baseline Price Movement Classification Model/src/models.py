# src/models.py
# Subtask D — Baseline Price Movement Classification Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
FEATURE_MATRIX = Path("data/features/feature_matrix.csv")
MODELS_DIR     = Path("models")
REPORTS_DIR    = Path("reports")
PLOTS_DIR      = Path("reports/plots")

# ── LABEL DEFINITION ──────────────────────────────────────────────────────────
# If tomorrow's price is more than THRESHOLD% higher → Up
# If tomorrow's price is more than THRESHOLD% lower  → Down
# In between                                         → Flat
THRESHOLD = 0.01   # 1% — standard in crypto prediction literature


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CREATE THE LABEL
# ══════════════════════════════════════════════════════════════════════════════
def create_labels(df):
    """
    Creates the target variable: next-day price direction.

    Key concept — we shift the price return by -1:
    This means: for row on Day 1, the label = what happens on Day 2.
    We're predicting TOMORROW using TODAY's features.

    Without the shift, we'd be predicting today using today's price —
    that's cheating (the model would already know the answer).
    """
    result_frames = []

    for coin in ["BTC", "ETH"]:
        coin_df = df[df["coin"] == coin].copy()
        coin_df = coin_df.sort_values("date").reset_index(drop=True)

        # next_return = tomorrow's price return
        # shift(-1) moves the column up by 1 row
        # so row 0 gets row 1's value, row 1 gets row 2's, etc.
        coin_df["next_return"] = coin_df["price_return"].shift(-1)

        # Apply threshold to create Up/Down/Flat label
        def label_direction(ret):
            if pd.isna(ret):
                return None
            elif ret > THRESHOLD:
                return "Up"
            elif ret < -THRESHOLD:
                return "Down"
            else:
                return "Flat"

        coin_df["label"] = coin_df["next_return"].apply(label_direction)
        result_frames.append(coin_df)

    labelled = pd.concat(result_frames, ignore_index=True)

    # Drop rows where we couldn't create a label
    # (last row of each coin — no "tomorrow" to predict)
    labelled = labelled.dropna(subset=["label"])

    print(f"  Total labelled rows: {len(labelled):,}")
    for coin in ["BTC", "ETH"]:
        coin_rows = labelled[labelled["coin"] == coin]
        counts = coin_rows["label"].value_counts()
        print(f"  {coin} labels: {dict(counts)}")

    return labelled


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPARE FEATURES + WALK-FORWARD SPLIT
# ══════════════════════════════════════════════════════════════════════════════
# These are the features we give to the model
# We deliberately exclude: date, coin, open, high, low (raw price levels)
# Why exclude raw prices? They are non-stationary (always trending up/down)
# Models trained on raw prices don't generalise to new price ranges
FEATURE_COLS = [
    "avg_score",          # original sentiment score (1-3)
    "pct_positive",       # fraction positive tweets
    "pct_negative",       # fraction negative tweets
    "tweet_count",        # how many tweets that day
    "vader_mean",         # VADER compound score
    "vader_std",          # disagreement in sentiment
    "sentiment_7d_avg",   # 7-day sentiment trend
    "sentiment_7d_std",   # 7-day sentiment volatility
    "sentiment_momentum", # sentiment change vs yesterday
    "volume_7d_avg",      # 7-day tweet volume trend
    "price_return",       # today's price return
    "price_7d_volatility",# 7-day price volatility
    "price_7d_momentum",  # 7-day price momentum
]


def prepare_data(df, coin):
    """
    Prepares features and labels for one coin.
    Uses walk-forward split: first 80% = train, last 20% = test.

    Why walk-forward and not random split?
    In time series, the future cannot inform the past.
    If we randomly split, a training row from Nov 2021 could
    sit next to a test row from Oct 2021 — the model learns
    patterns from the future. That's data leakage.
    Walk-forward ensures train always precedes test in time.
    """
    coin_df = df[df["coin"] == coin].copy()
    coin_df = coin_df.sort_values("date").reset_index(drop=True)

    # Drop rows where any feature is NaN
    # Most models can't handle NaN — better to drop than impute badly
    coin_df = coin_df.dropna(subset=FEATURE_COLS + ["label"])

    X = coin_df[FEATURE_COLS]
    y = coin_df["label"]
    dates = coin_df["date"]

    # Walk-forward split: 80% train, 20% test
    split_idx = int(len(coin_df) * 0.80)

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    train_dates = dates.iloc[:split_idx]
    test_dates  = dates.iloc[split_idx:]

    print(f"  {coin}: {len(X_train)} train rows | {len(X_test)} test rows")
    print(f"  Train: {train_dates.min()} → {train_dates.max()}")
    print(f"  Test:  {test_dates.min()} → {test_dates.max()}")

    # Scale features for Logistic Regression
    # LR is sensitive to feature scale — a feature ranging 0-10000
    # will dominate one ranging 0-1 unless we normalise
    # RF and XGBoost don't need scaling but it doesn't hurt them
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, scaler, coin_df)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN + EVALUATE MODELS
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(X_train, X_test, y_train, y_test,
                       X_train_scaled, X_test_scaled, coin):
    """
    Trains 3 models and returns their results.
    Uses macro F1 as the main metric — better than accuracy
    for imbalanced classes (more Up days than Down days typically).
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,        # enough iterations to converge
            random_state=42,      # reproducibility
            class_weight="balanced"  # handles class imbalance
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,     # 200 trees — good balance of speed vs accuracy
            max_depth=6,          # prevents overfitting
            random_state=42,
            class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,   # slow learning = less overfitting
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0
        ),
    }

    # Encode string labels to numbers for XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif name == "XGBoost":
            model.fit(X_train, y_train_enc)
            y_pred_enc = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Calculate metrics
        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro",
                          zero_division=0)
        f1_wtd = f1_score(y_test, y_pred, average="weighted",
                          zero_division=0)

        print(f"    Accuracy     : {acc*100:.1f}%")
        print(f"    F1 (macro)   : {f1_mac:.3f}")
        print(f"    F1 (weighted): {f1_wtd:.3f}")
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred,
                                    zero_division=0))

        results[name] = {
            "model":    model,
            "y_pred":   y_pred,
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_wtd":   f1_wtd,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PLOT CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(results, y_test, coin):
    """
    Plots a confusion matrix for each model side by side.
    A confusion matrix shows: for each true label (Up/Down/Flat),
    how often did the model predict each class?
    Perfect model = all values on the diagonal.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Down", "Flat", "Up"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{coin} — Confusion Matrices (Test Set)", fontsize=14)

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"], labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{name}\nAcc={res['accuracy']*100:.1f}% "
                     f"F1={res['f1_macro']:.3f}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    path = PLOTS_DIR / f"{coin}_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix plot: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_models():
    print("=" * 55)
    print("SUBTASK D - BASELINE ML MODELS")
    print("=" * 55)

    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature matrix
    print("\nLoading feature matrix...")
    df = pd.read_csv(FEATURE_MATRIX)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Step 1: Create labels
    print("\nStep 1 — Creating labels (Up/Down/Flat)...")
    df = create_labels(df)

    # Store all results for summary
    all_results = {}

    for coin in ["BTC", "ETH"]:
        print(f"\n{'='*55}")
        print(f"PROCESSING {coin}")
        print(f"{'='*55}")

        # Step 2: Prepare data
        print(f"\nStep 2 — Preparing data for {coin}...")
        (X_train, X_test, y_train, y_test,
         X_train_scaled, X_test_scaled,
         scaler, coin_df) = prepare_data(df, coin)

        if len(X_train) < 10 or len(X_test) < 5:
            print(f"  Not enough data for {coin}, skipping")
            continue

        # Step 3: Train models
        print(f"\nStep 3 — Training models for {coin}...")
        results = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, coin
        )

        # Step 4: Plot confusion matrices
        print(f"\nStep 4 — Plotting confusion matrices for {coin}...")
        plot_confusion_matrices(results, y_test, coin)

        # Save best model (highest macro F1)
        best_name = max(results, key=lambda k: results[k]["f1_macro"])
        best_model = results[best_name]["model"]
        model_path = MODELS_DIR / f"{coin}_best_model.pkl"
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, MODELS_DIR / f"{coin}_scaler.pkl")
        print(f"\n  Best model: {best_name} "
              f"(F1={results[best_name]['f1_macro']:.3f})")
        print(f"  Saved to: {model_path}")

        all_results[coin] = results

    # Final summary
    print("\n" + "=" * 55)
    print("SUBTASK D COMPLETE — RESULTS SUMMARY")
    print("=" * 55)
    for coin, results in all_results.items():
        print(f"\n{coin}:")
        for name, res in results.items():
            print(f"  {name:<22} "
                  f"Acc={res['accuracy']*100:.1f}%  "
                  f"F1={res['f1_macro']:.3f}")

    print("\nFiles saved:")
    print("  models/BTC_best_model.pkl")
    print("  models/ETH_best_model.pkl")
    print("  reports/plots/BTC_confusion_matrices.png")
    print("  reports/plots/ETH_confusion_matrices.png")
    print("\nSubtask D complete!")


if __name__ == "__main__":
    run_models()