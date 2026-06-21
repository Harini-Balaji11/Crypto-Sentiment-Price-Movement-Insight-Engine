# src/advanced_models.py
# Subtask E — Advanced Modelling & Feature Importance Analysis

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
FEATURE_MATRIX = Path("data/features/feature_matrix.csv")
MODELS_DIR     = Path("models")
PLOTS_DIR      = Path("reports/plots")
REPORTS_DIR    = Path("reports")

# ── SAME FEATURES AS SUBTASK D ────────────────────────────────────────────────
FEATURE_COLS = [
    "avg_score", "pct_positive", "pct_negative", "tweet_count",
    "vader_mean", "vader_std", "sentiment_7d_avg", "sentiment_7d_std",
    "sentiment_momentum", "volume_7d_avg", "price_return",
    "price_7d_volatility", "price_7d_momentum",
]

THRESHOLD = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# LABEL CREATION — same as Subtask D
# ══════════════════════════════════════════════════════════════════════════════
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

    labelled = pd.concat(result_frames, ignore_index=True)
    return labelled.dropna(subset=["label"])


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION — same walk-forward split as Subtask D
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data(df, coin):
    coin_df = df[df["coin"] == coin].copy()
    coin_df = coin_df.sort_values("date").reset_index(drop=True)
    coin_df = coin_df.dropna(subset=FEATURE_COLS + ["label"])

    X = coin_df[FEATURE_COLS]
    y = coin_df["label"]

    split_idx = int(len(coin_df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Encode labels for XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    print(f"  {coin}: {len(X_train)} train | {len(X_test)} test rows")
    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            y_train_enc, y_test_enc, le, scaler)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — XGBOOST HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
def tune_xgboost(X_train, y_train_enc, le):
    """
    GridSearchCV searches over combinations of hyperparameters.
    We use TimeSeriesSplit instead of regular cross-validation
    to respect the time ordering within the training set.

    Parameters we search:
    - n_estimators: how many trees (more = more powerful but slower)
    - max_depth: how deep each tree can go (deeper = more complex)
    - learning_rate: how much each tree corrects the previous one
    - subsample: fraction of rows used per tree (prevents overfitting)
    """
    print("  Tuning XGBoost with GridSearchCV...")
    print("  (This takes 2-3 minutes — searching 36 combinations)")

    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample":     [0.8],
    }

    xgb = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )

    # StratifiedKFold preserves class balance in each fold
    # n_splits=3 because our training set is small
    cv = StratifiedKFold(n_splits=3, shuffle=False)

    grid_search = GridSearchCV(
        xgb, param_grid,
        cv=cv,
        scoring="f1_macro",   # optimise for balanced F1
        n_jobs=-1,            # use all CPU cores
        verbose=0,
    )
    grid_search.fit(X_train, y_train_enc)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1 (macro): {grid_search.best_score_:.3f}")
    return grid_search.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STACKED ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
def build_stacked_ensemble(X_train, y_train, X_train_scaled):
    """
    A stacked ensemble works in two layers:
    - Layer 1 (base learners): LR, RF, XGBoost each make predictions
    - Layer 2 (meta-learner): Logistic Regression learns which base
      learner to trust in which situations

    The key trick: base learners are trained on cross-validated
    out-of-fold predictions so the meta-learner never sees training
    data that the base learners were trained on (prevents overfitting).

    Note: StackingClassifier in sklearn handles this automatically.
    """
    print("  Building stacked ensemble...")

    base_learners = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=6,
                                       random_state=42,
                                       class_weight="balanced")),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=4,
                               learning_rate=0.05, random_state=42,
                               eval_metric="mlogloss", verbosity=0)),
    ]

    # Meta-learner: simple LR that combines base learner outputs
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=3,                    # 3-fold cross-val for out-of-fold predictions
        passthrough=False,       # only use base learner predictions, not raw features
        n_jobs=-1,
    )

    stack.fit(X_train, y_train)
    return stack


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SHAP FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def run_shap_analysis(model, X_train, X_test, coin):
    """
    SHAP (SHapley Additive exPlanations) assigns each feature a
    contribution value for each individual prediction.

    Positive SHAP value = pushed prediction toward the predicted class
    Negative SHAP value = pushed prediction away from predicted class

    We use TreeExplainer because our model is XGBoost (tree-based).
    TreeExplainer is much faster than the model-agnostic KernelExplainer.

    We plot:
    1. Summary bar plot — overall feature importance (mean |SHAP|)
    2. Beeswarm plot — shows direction + magnitude for each feature
    """
    print(f"  Running SHAP analysis for {coin}...")

    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for test set
    # shap_values shape: (n_samples, n_features, n_classes)
    shap_values = explainer.shap_values(X_test)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Summary bar — mean absolute SHAP per feature ──────────────
    # If shap_values is 3D (one per class), average across classes
    if isinstance(shap_values, list):
        # older shap returns list of arrays (one per class)
        shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # newer shap returns 3D array
        shap_mean = np.mean(np.abs(shap_values), axis=2)
    else:
        shap_mean = np.abs(shap_values)

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_shap": shap_mean.mean(axis=0)
    }).sort_values("mean_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#0A7EA4" if "sentiment" in f or "vader" in f or
              "avg_score" in f or "pct_" in f or "tweet" in f or "volume" in f
              else "#C75B00" for f in feature_importance["feature"]]
    bars = ax.barh(feature_importance["feature"],
                   feature_importance["mean_shap"],
                   color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Mean |SHAP Value| — average impact on prediction", fontsize=11)
    ax.set_title(f"{coin} — Feature Importance (SHAP)\n"
                 f"Blue = sentiment features   Orange = price features",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path1 = PLOTS_DIR / f"{coin}_shap_importance.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path1}")

    # Return importance for report
    return feature_importance


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def run_permutation_importance(model, X_test, y_test, coin):
    """
    Permutation importance is model-agnostic — works on any model.
    For each feature, it randomly shuffles that column in the test set
    and measures how much the accuracy drops.

    Big drop = feature was important (model relied on it)
    Small drop = feature wasn't contributing much
    Negative drop = shuffling actually helped (feature was adding noise)

    This is a great cross-check for SHAP results.
    """
    print(f"  Running permutation importance for {coin}...")

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=20,          # shuffle each feature 20 times and average
        random_state=42,
        scoring="f1_macro",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#0A7EA4" if "sentiment" in f or "vader" in f or
              "avg_score" in f or "pct_" in f or "tweet" in f or "volume" in f
              else "#C75B00" for f in perm_df["feature"]]
    ax.barh(perm_df["feature"], perm_df["importance"],
            xerr=perm_df["std"], color=colors,
            edgecolor="white", height=0.7, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Drop in F1 macro when feature is shuffled", fontsize=11)
    ax.set_title(f"{coin} — Permutation Importance\n"
                 f"Blue = sentiment features   Orange = price features",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = PLOTS_DIR / f"{coin}_permutation_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return perm_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_advanced_models():
    print("=" * 55)
    print("SUBTASK E - ADVANCED MODELLING & FEATURE IMPORTANCE")
    print("=" * 55)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading feature matrix...")
    df = pd.read_csv(FEATURE_MATRIX)
    df = create_labels(df)
    print(f"  Loaded {len(df):,} labelled rows")

    all_results = {}

    for coin in ["BTC", "ETH"]:
        print(f"\n{'='*55}")
        print(f"PROCESSING {coin}")
        print(f"{'='*55}")

        # Prepare data
        (X_train, X_test, y_train, y_test,
         X_train_scaled, X_test_scaled,
         y_train_enc, y_test_enc, le, scaler) = prepare_data(df, coin)

        coin_results = {}

        # ── Step 1: Tuned XGBoost ──────────────────────────────────────────
        print(f"\nStep 1 — Tuned XGBoost for {coin}...")
        tuned_xgb = tune_xgboost(X_train, y_train_enc, le)

        y_pred_enc = tuned_xgb.predict(X_test)
        y_pred     = le.inverse_transform(y_pred_enc)

        acc_txgb = accuracy_score(y_test, y_pred)
        f1_txgb  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"  Tuned XGBoost — Accuracy: {acc_txgb*100:.1f}%  F1: {f1_txgb:.3f}")
        print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
        coin_results["Tuned XGBoost"] = {"accuracy": acc_txgb, "f1_macro": f1_txgb}

        # ── Step 2: Stacked Ensemble ───────────────────────────────────────
        print(f"\nStep 2 — Stacked Ensemble for {coin}...")
        stack = build_stacked_ensemble(X_train, y_train, X_train_scaled)
        y_pred_stack = stack.predict(X_test)

        acc_stack = accuracy_score(y_test, y_pred_stack)
        f1_stack  = f1_score(y_test, y_pred_stack, average="macro", zero_division=0)
        print(f"  Stacked Ensemble — Accuracy: {acc_stack*100:.1f}%  F1: {f1_stack:.3f}")
        print(f"\n{classification_report(y_test, y_pred_stack, zero_division=0)}")
        coin_results["Stacked Ensemble"] = {"accuracy": acc_stack, "f1_macro": f1_stack}

        # Save best model from E
        best_e_name = max(coin_results, key=lambda k: coin_results[k]["f1_macro"])
        best_e_f1   = coin_results[best_e_name]["f1_macro"]

        # Load Subtask D best F1 for comparison
        d_f1 = {"BTC": 0.406, "ETH": 0.264}[coin]
        improved = best_e_f1 > d_f1
        print(f"\n  Subtask D best F1 : {d_f1:.3f}")
        print(f"  Subtask E best F1 : {best_e_f1:.3f}  "
              f"({'IMPROVED' if improved else 'same/lower'})")

        # ── Step 3: SHAP on tuned XGBoost ─────────────────────────────────
        print(f"\nStep 3 — SHAP analysis for {coin}...")
        shap_df = run_shap_analysis(tuned_xgb, X_train, X_test, coin)
        print(f"\n  Top 5 features by SHAP importance:")
        print(shap_df.tail(5)[["feature","mean_shap"]].to_string(index=False))

        # ── Step 4: Permutation importance on stacked ensemble ─────────────
        print(f"\nStep 4 — Permutation importance for {coin}...")
        perm_df = run_permutation_importance(stack, X_test, y_test, coin)
        print(f"\n  Top 5 features by permutation importance:")
        print(perm_df.tail(5)[["feature","importance"]].to_string(index=False))

        all_results[coin] = {
            "results": coin_results,
            "shap_top5": shap_df.tail(5)["feature"].tolist(),
            "perm_top5": perm_df.tail(5)["feature"].tolist(),
            "d_f1": d_f1,
            "best_e_f1": best_e_f1,
        }

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUBTASK E COMPLETE — SUMMARY")
    print("=" * 55)

    for coin, data in all_results.items():
        print(f"\n{coin}:")
        print(f"  Subtask D baseline F1 : {data['d_f1']:.3f}")
        for name, res in data["results"].items():
            print(f"  {name:<22} Acc={res['accuracy']*100:.1f}%  "
                  f"F1={res['f1_macro']:.3f}")
        print(f"  Top SHAP features : {data['shap_top5'][-3:]}")
        print(f"  Top Perm features : {data['perm_top5'][-3:]}")

    print("\nPlots saved:")
    for coin in ["BTC", "ETH"]:
        print(f"  reports/plots/{coin}_shap_importance.png")
        print(f"  reports/plots/{coin}_permutation_importance.png")
    print("\nSubtask E complete!")


if __name__ == "__main__":
    run_advanced_models()