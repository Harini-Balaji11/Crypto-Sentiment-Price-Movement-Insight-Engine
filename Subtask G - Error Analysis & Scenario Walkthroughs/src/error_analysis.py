# src/error_analysis.py
# Subtask G — Error Analysis & Scenario Walkthroughs
# Finds the most interesting misclassified periods for case study writing

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

FEATURE_MATRIX = Path("data/features/feature_matrix.csv")
PLOTS_DIR      = Path("reports/plots")
THRESHOLD      = 0.01

FEATURE_COLS = [
    "avg_score", "pct_positive", "pct_negative", "tweet_count",
    "vader_mean", "vader_std", "sentiment_7d_avg", "sentiment_7d_std",
    "sentiment_momentum", "volume_7d_avg", "price_return",
    "price_7d_volatility", "price_7d_momentum",
]


def create_labels(df):
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


def get_predictions(df, coin):
    """Reproduce predictions from Subtask E best models."""
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
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
            eval_metric="mlogloss", verbosity=0
        )
        model.fit(X_train, y_train_enc)
        y_pred = le.inverse_transform(model.predict(X_test))
    else:
        base = [
            ("lr", LogisticRegression(max_iter=1000, random_state=42,
                                       class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=6,
                                          random_state=42,
                                          class_weight="balanced")),
            ("xgb", XGBClassifier(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42,
                                   eval_metric="mlogloss", verbosity=0)),
        ]
        model = StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=3, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Build full test dataframe with all context columns
    test_df = coin_df.iloc[split_idx:].copy()
    test_df["predicted"] = y_pred
    test_df["actual"]    = y_test.values
    test_df["correct"]   = test_df["predicted"] == test_df["actual"]
    test_df["date"]      = test_dates.values

    return test_df


def find_interesting_failures(test_df, coin):
    """
    Find the most explainable failure cases:
    1. Sentiment positive but price went Down (false optimism)
    2. Sentiment negative but price went Up (false pessimism)
    3. High confidence period with many wrong calls in a row
    """
    wrong = test_df[~test_df["correct"]].copy()

    print(f"\n{'='*55}")
    print(f"{coin} — ERROR ANALYSIS")
    print(f"{'='*55}")
    print(f"Total test rows  : {len(test_df)}")
    print(f"Correct          : {test_df['correct'].sum()} ({test_df['correct'].mean()*100:.1f}%)")
    print(f"Wrong            : {len(wrong)} ({len(wrong)/len(test_df)*100:.1f}%)")

    # Misclassification breakdown
    print(f"\nMisclassification breakdown:")
    for actual in ["Up", "Down", "Flat"]:
        for pred in ["Up", "Down", "Flat"]:
            if actual != pred:
                count = len(test_df[(test_df["actual"]==actual) &
                                    (test_df["predicted"]==pred)])
                if count > 0:
                    print(f"  Actual={actual}, Predicted={pred}: {count} cases")

    # Case Study 1: Positive sentiment → price went Down
    # (model predicted Up but actual was Down)
    false_optimism = test_df[
        (test_df["actual"] == "Down") &
        (test_df["predicted"] == "Up") &
        (test_df["vader_mean"] > 0.05)
    ].copy()

    print(f"\nCase Study type 1 — Positive sentiment but price fell:")
    print(f"  Found {len(false_optimism)} cases")
    if len(false_optimism) > 0:
        worst = false_optimism.nlargest(3, "vader_mean")
        for _, row in worst.iterrows():
            print(f"  Date: {str(row['date'])[:10]} | "
                  f"VADER: {row['vader_mean']:.3f} | "
                  f"next_return: {row['next_return']*100:.2f}% | "
                  f"close: ${row['close']:,.0f}")

    # Case Study 2: Negative sentiment → price went Up
    false_pessimism = test_df[
        (test_df["actual"] == "Up") &
        (test_df["predicted"] == "Down") &
        (test_df["vader_mean"] < -0.05)
    ].copy()

    print(f"\nCase Study type 2 — Negative sentiment but price rose:")
    print(f"  Found {len(false_pessimism)} cases")
    if len(false_pessimism) > 0:
        for _, row in false_pessimism.iterrows():
            print(f"  Date: {str(row['date'])[:10]} | "
                  f"VADER: {row['vader_mean']:.3f} | "
                  f"next_return: {row['next_return']*100:.2f}% | "
                  f"close: ${row['close']:,.0f}")

    # Case Study 3: Consecutive wrong predictions (model confusion period)
    test_df_sorted = test_df.sort_values("date").reset_index(drop=True)
    test_df_sorted["wrong_streak"] = (
        (~test_df_sorted["correct"])
        .groupby((test_df_sorted["correct"]).cumsum())
        .cumsum()
    )
    max_streak = test_df_sorted["wrong_streak"].max()
    streak_end = test_df_sorted[
        test_df_sorted["wrong_streak"] == max_streak
    ]["date"].iloc[0]

    print(f"\nCase Study type 3 — Longest wrong streak:")
    print(f"  {int(max_streak)} consecutive wrong predictions ending around {str(streak_end)[:10]}")

    # Worst single predictions (largest price move model got wrong)
    test_df["abs_return"] = test_df["next_return"].abs()
    big_misses = test_df[~test_df["correct"]].nlargest(5, "abs_return")
    print(f"\nTop 5 biggest price moves the model got wrong:")
    for _, row in big_misses.iterrows():
        print(f"  {str(row['date'])[:10]} | actual={row['actual']} "
              f"predicted={row['predicted']} | "
              f"move={row['next_return']*100:.1f}% | "
              f"VADER={row['vader_mean']:.3f}")

    # Overall sentiment accuracy
    print(f"\nAccuracy by actual label:")
    for label in ["Up", "Down", "Flat"]:
        subset = test_df[test_df["actual"] == label]
        if len(subset) > 0:
            acc = subset["correct"].mean()
            print(f"  {label}: {acc*100:.1f}% ({subset['correct'].sum()}/{len(subset)})")

    return test_df


def plot_error_timeline(test_df, coin):
    """Plot correct vs wrong predictions over time."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{coin} — Error Analysis Timeline", fontsize=14, fontweight="bold")

    dates = pd.to_datetime(test_df["date"])

    # Panel 1: Price
    axes[0].plot(dates, test_df["close"], color="#F7931A", linewidth=1)
    axes[0].set_ylabel("Price (USD)", fontsize=10)
    axes[0].set_title("Closing Price", fontsize=10)

    # Panel 2: VADER sentiment
    axes[1].bar(dates, test_df["vader_mean"].fillna(0),
                color=test_df["vader_mean"].apply(
                    lambda v: "#00C896" if v > 0.05
                    else ("#FF4B4B" if v < -0.05 else "#AAAAAA")
                    if pd.notna(v) else "#AAAAAA"
                ), alpha=0.7, width=2)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("VADER Score", fontsize=10)
    axes[1].set_title("Daily VADER Sentiment", fontsize=10)

    # Panel 3: Correct vs Wrong
    correct_mask = test_df["correct"]
    wrong_mask   = ~test_df["correct"]

    axes[2].scatter(dates[correct_mask], [1]*correct_mask.sum(),
                    color="#00C896", s=30, label="Correct", alpha=0.8, zorder=3)
    axes[2].scatter(dates[wrong_mask], [0]*wrong_mask.sum(),
                    color="#FF4B4B", s=30, label="Wrong", alpha=0.8, zorder=3)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Wrong", "Correct"])
    axes[2].set_title("Prediction Correctness", fontsize=10)
    axes[2].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOTS_DIR / f"{coin}_error_timeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def run_error_analysis():
    print("=" * 55)
    print("SUBTASK G — ERROR ANALYSIS")
    print("=" * 55)

    df = pd.read_csv(FEATURE_MATRIX)
    df["date"] = pd.to_datetime(df["date"])
    df = create_labels(df)

    for coin in ["BTC", "ETH"]:
        print(f"\nGenerating predictions for {coin}...")
        test_df = get_predictions(df, coin)
        test_df = find_interesting_failures(test_df, coin)
        plot_error_timeline(test_df, coin)

    print("\n" + "=" * 55)
    print("ERROR ANALYSIS COMPLETE")
    print("Plots saved to reports/plots/")
    print("Use the output above to write your case studies.")
    print("=" * 55)


if __name__ == "__main__":
    run_error_analysis()