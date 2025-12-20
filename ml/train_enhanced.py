"""Enhanced CML Elimination Model Training Pipeline.

Supports multiple model types: Random Forest, Gradient Boosting, XGBoost.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
import joblib
from pathlib import Path
from datetime import datetime
import json
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.model_registry import (
    get_model_class,
    get_param_grid,
    get_default_params,
    list_available_models,
)


class EnhancedCMLModelTrainer:
    """Train and evaluate CML elimination prediction model with advanced features.

    Supports multiple model types for comparison and selection.
    """

    def __init__(
        self,
        data_path: str,
        model_output_dir: str = "models",
        model_type: str = "random_forest",
    ):
        self.data_path = Path(data_path)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(exist_ok=True)
        self.model = None
        self.feature_names = None
        self.model_type = model_type
        self.comparison_results: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """Load and validate CML data."""
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        required_cols = [
            "average_corrosion_rate",
            "thickness_mm",
            "commodity",
            "feature_type",
            "cml_shape",
            "elimination_flag",
        ]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"Loaded {len(df)} records")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for the model."""
        print("Engineering features...")

        # Corrosion-thickness ratio
        df["corrosion_thickness_ratio"] = (
            df["average_corrosion_rate"] / df["thickness_mm"]
        )

        # Risk categorization
        df["high_corrosion"] = (df["average_corrosion_rate"] > 0.15).astype(int)
        df["thin_wall"] = (df["thickness_mm"] < 8.0).astype(int)

        # Interaction features
        df["risk_interaction"] = df["high_corrosion"] * df["thin_wall"]

        # Time-based features if available
        if "last_inspection_date" in df.columns:
            df["last_inspection_date"] = pd.to_datetime(df["last_inspection_date"])
            df["days_since_inspection"] = (
                datetime.now() - df["last_inspection_date"]
            ).dt.days

        # Remaining life features if available
        if "remaining_life_years" in df.columns:
            df["low_remaining_life"] = (df["remaining_life_years"] < 10).astype(int)

        return df

    def prepare_features(self, df: pd.DataFrame):
        """Prepare feature matrix and target variable."""
        numerical_features = [
            "average_corrosion_rate",
            "thickness_mm",
            "corrosion_thickness_ratio",
        ]

        categorical_features = ["commodity", "feature_type", "cml_shape"]

        # Add optional features if they exist
        if "days_since_inspection" in df.columns:
            numerical_features.append("days_since_inspection")
        if "risk_score" in df.columns:
            numerical_features.append("risk_score")
        if "remaining_life_years" in df.columns:
            numerical_features.append("remaining_life_years")

        # Create preprocessing pipelines
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        X = df[numerical_features + categorical_features]
        y = df["elimination_flag"]

        self.feature_names = numerical_features + categorical_features

        return X, y, preprocessor

    def train_model(self, X, y, preprocessor):
        """Train classifier with hyperparameter tuning using model registry."""
        print(f"Training {self.model_type} model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Get model class and parameters from registry
        model_class = get_model_class(self.model_type)
        default_params = get_default_params(self.model_type)
        param_grid = get_param_grid(self.model_type)

        # Create pipeline
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", model_class(**default_params)),
            ]
        )

        # Grid search with cross-validation
        print("Running hyperparameter tuning...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Keep", "Eliminate"]))
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print(f"\nBest Parameters: {grid_search.best_params_}")

        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="f1")
        print(f"\nCross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Feature importance (if available)
        classifier = self.model.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            feature_names_out = self.model.named_steps[
                "preprocessor"
            ].get_feature_names_out()
            feature_importance_df = pd.DataFrame(
                {"feature": feature_names_out, "importance": importances}
            ).sort_values("importance", ascending=False)

            print("\nTop 10 Feature Importances:")
            print(feature_importance_df.head(10).to_string(index=False))

        return {
            "model_type": self.model_type,
            "test_accuracy": float(self.model.score(X_test, y_test)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "best_params": grid_search.best_params_,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def save_model(self, metrics: dict):
        """Save trained model and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_output_dir / f"cml_elimination_model_{timestamp}.joblib"
        latest_path = self.model_output_dir / "cml_elimination_model.joblib"

        # Save model
        joblib.dump(self.model, model_path)
        joblib.dump(self.model, latest_path)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "feature_names": self.feature_names,
            "metrics": metrics,
            "model_path": str(model_path),
        }

        metadata_path = self.model_output_dir / f"model_metadata_{timestamp}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel saved to {model_path}")
        print(f"Latest model: {latest_path}")

        return model_path


def train_enhanced_cml_model(data_path: str = "data/sample_cml_data.csv") -> Path:
    """Main training function."""
    trainer = EnhancedCMLModelTrainer(data_path)

    # Load and prepare data
    df = trainer.load_data()
    df = trainer.engineer_features(df)
    X, y, preprocessor = trainer.prepare_features(df)

    # Train model
    metrics = trainer.train_model(X, y, preprocessor)

    # Save model
    model_path = trainer.save_model(metrics)

    return model_path


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_cml_data.csv"
    train_enhanced_cml_model(data_path)
