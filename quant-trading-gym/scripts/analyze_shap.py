#!/usr/bin/env python3
"""
Analyze SHAP values for trained tree-based models.

Loads trained models and training data, computes SHAP feature importance.

Usage:
    python scripts/analyze_shap.py                          # All models in models/
    python scripts/analyze_shap.py --model models/shallow_decision_tree.json
    python scripts/analyze_shap.py --data data/training --max-samples 2000
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_training_data(base_path: str, max_samples: int = 5000) -> tuple[np.ndarray, list[str]]:
    """Load training data and return features matrix + names."""
    base = Path(base_path)
    market_path = base.parent / f"{base.stem}_market.parquet"
    agents_path = base.parent / f"{base.stem}_agents.parquet"
    
    if market_path.exists() and agents_path.exists():
        df_market = pl.read_parquet(market_path)
        df_agents = pl.read_parquet(agents_path)
        
        market_feature_cols = [c for c in df_market.columns if c.startswith("f_")]
        rename_map = {c: f"m_{c}" for c in market_feature_cols}
        df_market = df_market.rename(rename_map)
        
        df = df_agents.join(
            df_market.select(["tick", "symbol"] + [f"m_{c}" for c in market_feature_cols]),
            on=["tick", "symbol"],
            how="left"
        )
        unrename_map = {f"m_{c}": c for c in market_feature_cols}
        df = df.rename(unrename_map)
    else:
        legacy_path = base.with_suffix(".parquet")
        df = pl.read_parquet(legacy_path)
    
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    
    # Sample if too large
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), size=max_samples, replace=False)
        X = X[indices]
    
    print(f"Loaded {len(X):,} samples, {len(feature_cols)} features")
    return X, feature_cols


def rebuild_model_from_json(model_path: Path):
    """Rebuild sklearn model from JSON export (for tree/RF only)."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    with open(model_path) as f:
        data = json.load(f)
    
    model_type = data.get("model_type")
    n_features = data.get("n_features")
    
    if model_type == "decision_tree":
        # Can't fully rebuild from JSON, need pickle
        print(f"Warning: Cannot rebuild {model_type} from JSON for SHAP. Need .pkl file.")
        return None
    elif model_type == "random_forest":
        print(f"Warning: Cannot rebuild {model_type} from JSON for SHAP. Need .pkl file.")
        return None
    elif model_type == "hist_gradient_boosted":
        print(f"Warning: Cannot rebuild {model_type} from JSON for SHAP. Need .pkl file.")
        return None
    
    return None


def load_model(model_path: Path):
    """Load model from pickle or JSON."""
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    elif model_path.suffix == ".json":
        return rebuild_model_from_json(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")


def analyze_shap(model, X: np.ndarray, feature_names: list[str], model_name: str):
    """Compute and display SHAP analysis."""
    print(f"\n{'='*60}")
    print(f"SHAP Analysis: {model_name}")
    print(f"{'='*60}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Handle multi-class output
    if isinstance(shap_values, list):
        # Average absolute SHAP across classes
        mean_abs = sum(np.abs(sv).mean(axis=0) for sv in shap_values) / len(shap_values)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    importance = list(zip(feature_names, mean_abs))
    importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Features by Mean |SHAP|:")
    print("-" * 40)
    for i, (name, val) in enumerate(importance[:20], 1):
        bar = "█" * int(val / importance[0][1] * 20)
        print(f"{i:2}. {name:30} {val:8.4f} {bar}")
    
    # Summary stats
    print(f"\nTotal features: {len(feature_names)}")
    print(f"Top 10 explain {sum(v for _, v in importance[:10]) / sum(v for _, v in importance) * 100:.1f}% of importance")
    
    return importance


def main():
    parser = argparse.ArgumentParser(description="Analyze SHAP values for trained models")
    parser.add_argument("--model", type=Path, help="Path to specific model (.pkl)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory with model files")
    parser.add_argument("--data", type=str, default="data/training", help="Base path for training data")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples for SHAP computation")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    args = parser.parse_args()
    
    if not SHAP_AVAILABLE:
        print("Error: shap package not installed. Run: pip install shap")
        return 1
    
    # Load training data
    print("Loading training data...")
    X, feature_names = load_training_data(args.data, args.max_samples)
    
    # Find models
    if args.model:
        model_paths = [args.model]
    else:
        model_paths = list(args.models_dir.glob("*.pkl"))
        if not model_paths:
            print(f"No .pkl models found in {args.models_dir}")
            print("Note: SHAP analysis requires pickle files. Add this to train_trees.py:")
            print("  import pickle")
            print("  with open(f'{name}.pkl', 'wb') as f: pickle.dump(clf, f)")
            return 1
    
    results = {}
    
    for model_path in model_paths:
        print(f"\nLoading {model_path}...")
        model = load_model(model_path)
        if model is None:
            continue
        
        importance = analyze_shap(model, X, feature_names, model_path.stem)
        results[model_path.stem] = {
            "feature_importance": [{"name": n, "shap": float(v)} for n, v in importance]
        }
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
