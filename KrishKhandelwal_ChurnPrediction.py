# -*- coding: utf-8 -*-
"""
train_model.py — Standalone training script for Telco Customer Churn Prediction
================================================================================
Usage:
    python train_model.py                      # train all models, save artefacts
    python train_model.py --model rf           # train only Random Forest
    python train_model.py --cv                 # include 5-fold cross-validation
    python train_model.py --model gb --cv      # GBT with cross-validation

Saved artefacts
---------------
  models/logistic_regression.pkl
  models/random_forest.pkl
  models/gradient_boosting.pkl
  models/best_model.pkl          <- best by ROC-AUC
  models/feature_columns.pkl     <- feature name list for inference alignment
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.preprocessing import load_and_preprocess

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ── ANSI colour helpers ──────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def hdr(text):  print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}  {text}{RESET}\n{BOLD}{CYAN}{'─'*60}{RESET}")
def ok(text):   print(f"  {GREEN}✔  {RESET}{text}")
def info(text): print(f"  {YELLOW}»  {RESET}{text}")
def err(text):  print(f"  {RED}✘  {RESET}{text}")


# ── Model registry ────────────────────────────────────────────────────────────
ALL_MODELS = {
    "lr": ("Logistic Regression", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ])),
    "rf": ("Random Forest", Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])),
    "gb": ("Gradient Boosting", Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4,
            random_state=RANDOM_STATE,
        )),
    ])),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(obj, filename):
    path = MODEL_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def _evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    return {
        "name":      name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1 Score":  round(f1_score(y_test, y_pred),        4),
        "ROC-AUC":   round(auc, 4),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "cm":        confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred,
                         target_names=["Retained", "Churned"]),
    }


def _print_metrics(result):
    m = result
    print(f"\n  {'Metric':<14} {'Value':>8}")
    print(f"  {'─'*24}")
    for key in ("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"):
        val = m[key]
        colour = GREEN if val >= 0.80 else (YELLOW if val >= 0.65 else RED)
        print(f"  {key:<14} {colour}{val:>8.4f}{RESET}")


def _print_confusion(cm, name):
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix — {name}")
    print(f"  {'':>14} Pred Retained  Pred Churned")
    print(f"  {'Act Retained':<14} {tn:>13,}  {fp:>12,}")
    print(f"  {'Act Churned':<14} {fn:>13,}  {tp:>12,}")
    print(f"\n  True Positives  : {tp:,}   (correctly identified churners)")
    print(f"  False Negatives : {fn:,}   (missed churners)")
    print(f"  False Positives : {fp:,}   (false alarms)")
    print(f"  True Negatives  : {tn:,}   (correctly identified retained)")


def _print_feature_importance(pipeline, feature_names, top_n=15):
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\n  Top {top_n} Feature Importances:")
    print(f"  {'Feature':<42} {'Importance':>10}")
    print(f"  {'─'*54}")
    for feat, val in imp.head(top_n).items():
        bar = "█" * int(val * 200)
        print(f"  {feat:<42} {val:>10.4f}  {bar}")


def _cross_validate(name, pipeline, X, y):
    hdr(f"Cross-Validation: {name}")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    info(f"Running {CV_FOLDS}-fold stratified CV …")
    t0 = time.time()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed = time.time() - t0
    info(f"Completed in {elapsed:.1f}s")
    print(f"\n  {'Metric':<14} {'Mean':>8}  {'Std':>7}  {'Min':>8}  {'Max':>8}")
    print(f"  {'─'*52}")
    mapping = {
        "Accuracy":  "test_accuracy",
        "Precision": "test_precision",
        "Recall":    "test_recall",
        "F1 Score":  "test_f1",
        "ROC-AUC":   "test_roc_auc",
    }
    for label, key in mapping.items():
        vals = scores[key]
        colour = GREEN if vals.mean() >= 0.80 else (YELLOW if vals.mean() >= 0.65 else RED)
        print(
            f"  {label:<14} {colour}{vals.mean():>8.4f}{RESET}  "
            f"±{vals.std():>6.4f}  {vals.min():>8.4f}  {vals.max():>8.4f}"
        )


# ── Main training routine ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train churn prediction models on the Telco dataset."
    )
    parser.add_argument(
        "--model", choices=["lr", "rf", "gb", "all"], default="all",
        help="Which model(s) to train: lr=Logistic Regression, rf=Random Forest, gb=Gradient Boosting, all=all three (default: all)",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold stratified cross-validation in addition to hold-out evaluation",
    )
    parser.add_argument(
        "--test-size", type=float, default=TEST_SIZE,
        help=f"Fraction of data to use as test set (default: {TEST_SIZE})",
    )
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}")
    print("  Telco Customer Churn -- Model Training Script")
    print(f"{'='*60}{RESET}")

    # ── Load data ─────────────────────────────────────────────────────────────
    hdr("1. Loading & Preprocessing Data")
    t0 = time.time()
    raw, processed, X, y = load_and_preprocess()
    elapsed = time.time() - t0

    ok(f"Raw dataset loaded          : {raw.shape[0]:,} rows × {raw.shape[1]} columns")
    ok(f"After feature engineering   : {X.shape[1]} features")
    ok(f"Target distribution         : Retained={int((y==0).sum()):,}  Churned={int((y==1).sum()):,}  (churn rate={y.mean():.1%})")
    ok(f"Load + preprocess time      : {elapsed:.2f}s")

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    ok(f"Train size: {len(X_train):,}   Test size: {len(X_test):,}   (stratified split {int((1-args.test_size)*100)}/{int(args.test_size*100)})")

    # Save feature column list for inference alignment
    _save(list(X.columns), "feature_columns.pkl")
    ok(f"Feature column list saved   → models/feature_columns.pkl")

    # ── Select models to train ────────────────────────────────────────────────
    if args.model == "all":
        selected = list(ALL_MODELS.items())
    else:
        selected = [(args.model, ALL_MODELS[args.model])]

    # ── Train & evaluate ──────────────────────────────────────────────────────
    all_results = {}

    for key, (name, pipeline) in selected:
        hdr(f"2. Training: {name}")
        info(f"Fitting on {len(X_train):,} samples …")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        ok(f"Training complete in {elapsed:.1f}s")
        ok(f"Pipeline: {' -> '.join(pipeline.named_steps.keys())}")

        # Hold-out evaluation
        hdr(f"3. Hold-Out Evaluation: {name}")
        result = _evaluate(name, pipeline, X_test, y_test)
        all_results[name] = result
        _print_metrics(result)

        # Confusion matrix
        _print_confusion(result["cm"], name)

        # Classification report
        print(f"\n  Classification Report — {name}")
        for line in result["report"].split("\n"):
            print(f"  {line}")

        # Feature importances (tree models)
        _print_feature_importance(pipeline, list(X.columns))

        # Cross-validation
        if args.cv:
            _cross_validate(name, pipeline, X, y)

        # Save individual model
        fname = f"{key}.pkl" if key != "all" else f"{name.lower().replace(' ', '_')}.pkl"
        short_name = name.lower().replace(" ", "_")
        saved_path = _save(pipeline, f"{short_name}.pkl")
        ok(f"Model saved → {saved_path.relative_to(ROOT)}")

    # ── Save all models when training all ────────────────────────────────────
    if args.model == "all":
        # Determine best by ROC-AUC
        best_name = max(all_results, key=lambda n: all_results[n]["ROC-AUC"])
        best_key  = [k for k, (n, _) in ALL_MODELS.items() if n == best_name][0]
        best_pipe = ALL_MODELS[best_key][1]
        _save(best_pipe, "best_model.pkl")
        ok(f"Best model ({best_name}) saved → models/best_model.pkl")

        # ── Summary table ─────────────────────────────────────────────────────
        hdr("4. Model Comparison Summary")
        metrics_order = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        col_w = 16
        print(f"  {'Model':<28} " + "  ".join(f"{m:>{col_w}}" for m in metrics_order))
        print(f"  {'─'*28} " + "  ".join("─"*col_w for _ in metrics_order))

        for mname, res in all_results.items():
            is_best = mname == best_name
            prefix = f"{GREEN}★ {mname:<26}{RESET}" if is_best else f"  {mname:<28}"
            vals = "  ".join(
                f"{YELLOW if is_best else ''}{res[m]:>{col_w}.4f}{RESET}"
                for m in metrics_order
            )
            print(f"  {prefix} {vals}")

        print(f"\n  {GREEN}{BOLD}Best model: {best_name} (ROC-AUC = {all_results[best_name]['ROC-AUC']:.4f}){RESET}")

    # ── Done ──────────────────────────────────────────────────────────────────
    hdr("Done")
    ok("All models trained and saved to models/")
    ok("Run the app:  streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
