#!/usr/bin/env python3
"""
V5.4: Train tree-based models on simulation data and export to JSON for Rust inference.

Models: Decision Tree, Random Forest, Gradient Boosted Trees
Output: JSON files with tree structure for V5.5 Rust inference

Input: Parquet files with market features (42 columns)
  - Supports multiple numbered files: {base}_001_market.parquet, {base}_002_market.parquet, ...
  - Falls back to single file: {base}_market.parquet
  - Legacy support: {base}.parquet

Usage:
    python scripts/train_trees.py                           # Use default config
    python scripts/train_trees.py --config my_config.yaml   # Custom config
    python scripts/train_trees.py --input data/training     # Base path (loads all matching files)
"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DEFAULT_CONFIG = Path(__file__).parent / "train_config.yaml"


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(base_path: str, label_config: dict) -> pl.DataFrame:
    """Load market data for training.
    
    Loads all numbered parquet files matching {stem}_NNN_market.parquet pattern
    and concatenates them into a single DataFrame. Falls back to single file
    if no numbered files exist.
    
    Args:
        base_path: Base path for parquet files (e.g., "data/training" loads 
                   all "data/training_NNN_market.parquet" files)
        
    Returns:
        DataFrame with market features for training (concatenated from all files)
    """
    import re
    
    base = Path(base_path)
    parent = base.parent
    stem = base.stem
    
    # Find all numbered market parquet files: {stem}_NNN_market.parquet
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)_market\.parquet$")
    numbered_files = []
    
    if parent.exists():
        for f in parent.iterdir():
            match = pattern.match(f.name)
            if match:
                num = int(match.group(1))
                numbered_files.append((num, f))
    
    # Sort by number and load all
    numbered_files.sort(key=lambda x: x[0])
    print(f"Found {len(numbered_files)} recording files:")

    Xs = []
    ys = []
    feature_cols = None
    total_rows = 0
    for num, filepath in numbered_files:
        df = pl.read_parquet(filepath)
        # Add run_id column to distinguish runs
        df = df.with_columns(pl.lit(num).alias("run_id").cast(pl.UInt32))
        X, y, feature_cols = prepare_features(compute_lookahead_labels(df, label_config), label_config)
        Xs.append(X)
        ys.append(y)
        total_rows += len(df)
        print(f"  #{num:03d}: {len(df):,} rows from {filepath.name}")
    
    print(f"Combined: {total_rows:,} total rows, {len(feature_cols)} features")
    print(np.concatenate(Xs).shape, np.concatenate(ys).shape)
    return np.concatenate(Xs), np.concatenate(ys), feature_cols
    

def compute_lookahead_labels(df: pl.DataFrame, label_config: dict) -> pl.DataFrame:
    """Compute lookahead price return rates for labeling.

    Args:
        df: DataFrame with tick, symbol, f_mid_price columns
        label_config: Config dict with horizons, rolling_window, buy_threshold, sell_threshold

    Returns:
        DataFrame with price_return_rate columns
    """
    price_horizons = label_config.get("horizons", [4, 8, 16, 32])
    rolling_window = label_config.get("rolling_window", 1)

    print(f"Computing lookahead labels: price_horizons={price_horizons}, rolling_window={rolling_window}")

    # Compute percentage return to rolling average of future prices
    # For horizon n and window w: avg(price[t+n], price[t+n+1], ..., price[t+n+w-1])
    for n in price_horizons:
        if rolling_window <= 1:
            # Original single-tick behavior
            future_price = pl.col("f_mid_price").shift(-n).over("symbol")
        else:
            # Rolling window average of future prices centered on horizon
            half = rolling_window // 2
            future_prices = [
                pl.col("f_mid_price").shift(-i).over("symbol")
                for i in range(n - half, n - half + rolling_window)
            ]
            future_price = sum(future_prices) / rolling_window

        df = df.with_columns([
            ((future_price - pl.col("f_mid_price")) / pl.col("f_mid_price"))
            .alias(f"price_return_rate_{n}")
        ])

    # Discard rows without future data (last max_horizon + window - 1 ticks)
    max_horizon = max(price_horizons)
    lookahead_needed = max_horizon + rolling_window - 1
    max_tick = df["tick"].max()
    rows_before = len(df)
    df = df.filter(pl.col("tick") <= max_tick - lookahead_needed)
    rows_after = len(df)
    print(f"Discarded {rows_before - rows_after:,} rows without lookahead data (last {lookahead_needed} ticks)")
    
    return df


def prepare_features(df: pl.DataFrame, label_config: dict | None = None) -> tuple:
    """Extract features and labels from dataframe.
    
    Label = avg price return rate across horizons [4, 8, 16, 32]
    buy if > buy_threshold, sell if < sell_threshold, else hold
    """
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    print(f"Using {len(feature_cols)} features")

    X = df.select(feature_cols).to_numpy()
    
    # Average price return rate across horizons
    price_horizons = (label_config or {}).get("horizons", [4, 8, 16, 32])
    buy_threshold = (label_config or {}).get("buy_threshold", 0.005)
    sell_threshold = (label_config or {}).get("sell_threshold", -0.005)
    
    return_rates = []
    for n in price_horizons:
        col = f"price_return_rate_{n}"
        if col in df.columns:
            return_rates.append(df[col].to_numpy())
    
    avg_return = np.nanmean(return_rates, axis=0)
    y = np.where(avg_return > buy_threshold, 1,
                np.where(avg_return < sell_threshold, -1, 0))
    print(f"Label: avg price return rate, buy>{buy_threshold:.4%}/tick, sell<{sell_threshold:.4%}/tick")

    # Check for NaN values in features
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        nan_cols = np.isnan(X).any(axis=0)
        nan_features = [feature_cols[i] for i, has_nan in enumerate(nan_cols) if has_nan]
        print(f"Warning: {nan_count:,} NaN values in {len(nan_features)} features: {nan_features}")
        print("Imputing NaN with -1 (no history)...")
        X = np.nan_to_num(X, nan=-1)
    
    # Handle NaN in labels (from lookahead at boundaries)
    nan_labels = np.isnan(y) if np.issubdtype(y.dtype, np.floating) else np.zeros(len(y), dtype=bool)
    if nan_labels.any():
        print(f"Dropping {nan_labels.sum():,} rows with NaN labels")
        X = X[~nan_labels]
        y = y[~nan_labels]

    # Class distribution
    y = y.astype(int)
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Class distribution: sell={dist.get(-1, 0)}, hold={dist.get(0, 0)}, buy={dist.get(1, 0)}")

    return X, y, feature_cols


def compute_balanced_sample_weights(y):
    """Compute balanced sample weights for classes (for GradientBoosting)."""
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight('balanced', y)


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
        class_weight='balanced',
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
        class_weight='balanced',
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
) -> tuple[GradientBoostingClassifier, dict]:
    """Train a gradient boosted classifier with exportable tree structure."""
    print(f"\n=== Training Gradient Boosted: {name} ===")

    n_estimators = config.get("n_estimators", 100)
    learning_rate = config.get("learning_rate", 0.1)
    max_depth = config.get("max_depth", 5)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=config.get("subsample", 1.0),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        random_state=42
    )
    sample_weights = compute_balanced_sample_weights(y_train)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Estimators: {n_estimators}, max_depth: {max_depth}, lr: {learning_rate}")

    # Feature importances
    importances = clf.feature_importances_.tolist()
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 features:")
    for feat_name, imp in top_features:
        print(f"  {feat_name}: {imp:.4f}")

    # Export trees - GradientBoostingClassifier stores trees in estimators_
    # Shape: (n_estimators, n_classes) for multi-class, each element is a DecisionTreeRegressor
    # For 3-class classification, there are 3 trees per stage (one per class)
    n_classes = len(clf.classes_)

    # Export all trees organized by stage and class
    stages = []
    for stage_idx in range(n_estimators):
        stage_trees = []
        for class_idx in range(n_classes):
            tree = clf.estimators_[stage_idx, class_idx]
            stage_trees.append(tree_to_dict_regressor(tree, feature_names))
        stages.append(stage_trees)

    result = {
        "model_type": "gradient_boosted",
        "model_name": name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "classes": clf.classes_.tolist(),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "init_value": clf.init_.class_prior_.tolist() if hasattr(clf.init_, 'class_prior_') else None,
        "stages": stages,  # List of stages, each stage has n_classes trees
        "feature_importances": importances,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_rows": len(X_train),
            "config": config,
            "accuracy": float(accuracy)
        }
    }

    return clf, result


def tree_to_dict_regressor(tree_regressor, feature_names: list[str]) -> dict:
    """Convert sklearn regression tree (used in GradientBoosting) to JSON-serializable dict.

    Unlike classification trees, regression trees store raw prediction values in leaves.
    """
    tree_ = tree_regressor.tree_
    n_nodes = tree_.node_count

    nodes = []
    for i in range(n_nodes):
        is_leaf = tree_.children_left[i] == -1

        if is_leaf:
            # For regression trees, value is the raw prediction
            # Shape is (1, 1) - just a single value
            leaf_value = float(tree_.value[i][0, 0])
            nodes.append({
                "feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "value": leaf_value
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
    parser.add_argument("--input", help="Base path for Parquet files (loads {input}_market.parquet + {input}_agents.parquet)")
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

    # Load and join market + agent data
    input_base = data_config.get("input", "data/training")
    # Strip .parquet suffix if user provided it (for backwards compatibility)
    if input_base.endswith(".parquet"):
        input_base = input_base[:-8]  # Remove ".parquet"
        
    label_config = config.get("labels", {})
    
    X, y, feature_names = load_data(input_base, label_config)

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
            
            # Also save pickle for SHAP analysis
            pkl_path = output_dir / f"{name}_{model_type}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(clf, f)

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
