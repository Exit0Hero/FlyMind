"""Traditional ML baselines."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def run_baselines(X_train, y_train, X_val, y_val, X_test, y_test, class_names):
    """Run baseline models and return results dict."""
    results = {}

    # Baseline 1: Majority class
    print("  Running Baseline 1: Majority class...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    results["majority_class"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # Baseline 2: Logistic Regression
    print("  Running Baseline 2: Logistic Regression...")
    lr = LogisticRegression(
        max_iter=500, random_state=42,
        solver="lbfgs", class_weight="balanced",
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # Baseline 3: Random Forest
    print("  Running Baseline 3: Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42,
        n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["random_forest"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    return results


def run_graph_feature_baselines(X_bio_train, X_graph_train, y_train,
                                 X_bio_test, X_graph_test, y_test):
    """Run baselines comparing bio-only vs graph-only vs combined."""
    results = {}

    # Graph features only
    print("  Running Random Forest: graph features only...")
    rf_graph = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced",
    )
    rf_graph.fit(X_graph_train, y_train)
    y_pred = rf_graph.predict(X_graph_test)
    results["rf_graph_only"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # Biological features only
    print("  Running Random Forest: biological features only...")
    rf_bio = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced",
    )
    rf_bio.fit(X_bio_train, y_train)
    y_pred = rf_bio.predict(X_bio_test)
    results["rf_bio_only"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # Combined
    print("  Running Random Forest: biological + graph features...")
    X_combined_train = np.hstack([X_bio_train, X_graph_train])
    X_combined_test = np.hstack([X_bio_test, X_graph_test])
    rf_combined = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced",
    )
    rf_combined.fit(X_combined_train, y_train)
    y_pred = rf_combined.predict(X_combined_test)
    results["rf_bio_plus_graph"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    return results
