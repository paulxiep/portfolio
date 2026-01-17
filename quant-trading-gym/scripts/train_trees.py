#!/usr/bin/env python3
"""
V5.4: Train tree-based models on simulation data and export to JSON for Rust inference.

Models: Decision Tree, Random Forest, Gradient Boosted Trees
Output: JSON files with tree structure for V5.5 Rust inference

Usage:
    python scripts/train_trees.py                           # Use default config
    python scripts/train_trees.py --config my_config.yaml   # Custom config
    python scripts/train_trees.py --input data/other.parquet  # Override input
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DEFAULT_CONFIG = Path(__file__).parent / "train_config.yaml"


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(path: str, agent_filter: str | None = None) -> pl.DataFrame:
    """Load training data from Parquet file."""
    df = pl.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")

    if agent_filter:
        df = df.filter(pl.col("agent_name").str.contains(agent_filter))
        print(f"Filtered to {len(df):,} rows matching '{agent_filter}'")

    return df


def prepare_features(df: pl.DataFrame) -> tuple:
    """Extract features and labels from dataframe."""
    import numpy as np

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"Using {len(feature_cols)} features")

    X = df.select(feature_cols).to_numpy()
    y = df["action"].to_numpy()

    # Check for NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        nan_cols = np.isnan(X).any(axis=0)
        nan_features = [feature_cols[i] for i, has_nan in enumerate(nan_cols) if has_nan]
        print(f"Warning: {nan_count:,} NaN values in {len(nan_features)} features: {nan_features}")
        print("Imputing NaN with 0 (neutral/no history)...")
        X = np.nan_to_num(X, nan=0.0)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Class distribution: sell={dist.get(-1, 0)}, hold={dist.get(0, 0)}, buy={dist.get(1, 0)}")

    return X, y, feature_cols


def tree_to_dict(tree, feature_names: list[str]) -> dict:
    """Convert sklearn tree to JSON-serializable dict with class probabilities."""
    tree_ = tree.tree_
    n_nodes = tree_.node_count

    nodes = []
    for i in range(n_nodes):
        is_leaf = tree_.children_left[i] == -1

        if is_leaf:
            values = tree_.value[i][0]
            total = values.sum()
            probs = (values / total).tolist() if total > 0 else [0.33, 0.34, 0.33]

            nodes.append({
                "feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "value": probs
            })
        else:
            nodes.append({
                "feature": int(tree_.feature[i]),
                "threshold": float(tree_.threshold[i]),
                "left": int(tree_.children_left[i]),
                "right": int(tree_.children_right[i]),
                "value": None
            })

    return {"n_nodes": n_nodes, "nodes": nodes}


def train_decision_tree(
    X_train, y_train, X_test, y_test,
    feature_names: list[str],
    config: dict,
    name: str
) -> tuple[DecisionTreeClassifier, dict]:
    """Train a single decision tree."""
    print(f"\n=== Training Decision Tree: {name} ===")

    clf = DecisionTreeClassifier(
        max_depth=config.get("max_depth", 10),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tree depth: {clf.get_depth()}, nodes: {clf.tree_.node_count}")

    result = {
        "model_type": "decision_tree",
        "model_name": name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": 3,
        "classes": [-1, 0, 1],
        "tree": tree_to_dict(clf, feature_names),
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "actual_depth": clf.get_depth(),
            "accuracy": float(accuracy)
        }
    }

    return clf, result


def train_random_forest(
    X_train, y_train, X_test, y_test,
    feature_names: list[str],
    config: dict,
    name: str
) -> tuple[RandomForestClassifier, dict]:
    """Train a random forest classifier."""
    print(f"\n=== Training Random Forest: {name} ===")

    n_estimators = config.get("n_estimators", 100)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=config.get("max_depth", 10),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        max_features=config.get("max_features", "sqrt"),
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Trees: {n_estimators}, max_depth: {config.get('max_depth', 10)}")

    importances = clf.feature_importances_.tolist()
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 features:")
    for feat_name, imp in top_features:
        print(f"  {feat_name}: {imp:.4f}")

    trees = [tree_to_dict(est, feature_names) for est in clf.estimators_]

    result = {
        "model_type": "random_forest",
        "model_name": name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": 3,
        "classes": [-1, 0, 1],
        "n_estimators": n_estimators,
        "trees": trees,
        "feature_importances": importances,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "accuracy": float(accuracy)
        }
    }

    return clf, result


def train_gradient_boosted(
    X_train, y_train, X_test, y_test,
    feature_names: list[str],
    config: dict,
    name: str
) -> tuple[HistGradientBoostingClassifier, dict]:
    """Train a histogram-based gradient boosted classifier (handles NaN natively)."""
    print(f"\n=== Training Gradient Boosted: {name} ===")

    n_estimators = config.get("n_estimators", 100)
    learning_rate = config.get("learning_rate", 0.1)
    max_depth = config.get("max_depth", 5)

    clf = HistGradientBoostingClassifier(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=config.get("min_samples_leaf", 20),
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Iterations: {n_estimators}, max_depth: {max_depth}, lr: {learning_rate}")

    # HistGradientBoosting doesn't expose feature_importances_ directly in older sklearn
    # but does in newer versions
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_.tolist()
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 features:")
        for feat_name, imp in top_features:
            print(f"  {feat_name}: {imp:.4f}")
    else:
        importances = None
        print("Feature importances not available")

    # HistGradientBoostingClassifier uses a different internal structure
    # We'll export metadata but note that tree extraction is not directly available
    n_classes = len(clf.classes_)

    result = {
        "model_type": "hist_gradient_boosted",
        "model_name": name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "classes": clf.classes_.tolist(),
        "n_iterations": n_estimators,
        "learning_rate": learning_rate,
        "feature_importances": importances,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "accuracy": float(accuracy),
            "note": "HistGradientBoosting uses histogram-based trees, export format differs from standard GB"
        }
    }

    return clf, result


def compute_shap(model, X_test, feature_names: list[str], model_name: str, max_samples: int = 1000) -> dict | None:
    """Compute SHAP values for model explainability."""
    try:
        import shap
    except ImportError:
        print("SHAP not installed, skipping SHAP analysis")
        return None

    print(f"\n=== Computing SHAP values for {model_name} ===")

    if len(X_test) > max_samples:
        import numpy as np
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_test), size=max_samples, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_abs = sum(abs(sv).mean(axis=0) for sv in shap_values) / len(shap_values)
    else:
        mean_abs = abs(shap_values).mean(axis=0)

    mean_abs_list = mean_abs.tolist()
    top_features = sorted(zip(feature_names, mean_abs_list), key=lambda x: x[1], reverse=True)[:15]

    print("Top 15 features by SHAP importance:")
    for feat_name, imp in top_features:
        print(f"  {feat_name}: {imp:.4f}")

    return {
        "model_name": model_name,
        "shap_values": {
            "feature_names": feature_names,
            "mean_abs_shap": mean_abs_list,
            "top_features": [{"name": n, "importance": float(i)} for n, i in top_features]
        }
    }


def print_classification_report(y_test, y_pred, model_name: str):
    """Print detailed classification metrics."""
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["sell (-1)", "hold (0)", "buy (+1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    parser = argparse.ArgumentParser(description="Train tree-based models for V5.5 Rust inference")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--input", help="Override input Parquet file")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    data_config = config.get("data", {})
    if args.input:
        data_config["input"] = args.input
    if args.output_dir:
        data_config["output_dir"] = args.output_dir

    output_dir = Path(data_config.get("output_dir", "models"))
    output_dir.mkdir(exist_ok=True)

    df = load_data(data_config.get("input", "data/training.parquet"), data_config.get("agent_filter"))
    X, y, feature_names = prepare_features(df)

    test_size = data_config.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Model type mappings
    model_groups = [
        ("decision_trees", "decision_tree", train_decision_tree),
        ("random_forests", "random_forest", train_random_forest),
        ("gradient_boosted", "gradient_boosted", train_gradient_boosted),
    ]

    trained_models = {}

    for config_key, model_type, trainer in model_groups:
        models_list = config.get(config_key, [])
        for model_config in models_list:
            name = model_config.get("name", "unnamed")

            clf, result = trainer(X_train, y_train, X_test, y_test, feature_names, model_config, name)

            filename = f"{name}_{model_type}.json"
            output_path = output_dir / filename
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {output_path}")

            trained_models[f"{name}_{model_type}"] = clf
            print_classification_report(y_test, clf.predict(X_test), f"{name} {model_type}")

    # SHAP analysis
    shap_config = config.get("shap", {})
    if shap_config.get("enabled", False) and trained_models:
        max_samples = shap_config.get("max_samples", 1000)
        shap_results = {}
        for model_key, model in trained_models.items():
            shap_result = compute_shap(model, X_test, feature_names, model_key, max_samples)
            if shap_result:
                shap_results[model_key] = shap_result

        if shap_results:
            output_path = output_dir / "shap_analysis.json"
            with open(output_path, "w") as f:
                json.dump(shap_results, f, indent=2)
            print(f"\nSHAP analysis saved to {output_path}")

    print("\n=== Training Complete ===")
    print(f"Models saved to {output_dir}/")


if __name__ == "__main__":
    main()
