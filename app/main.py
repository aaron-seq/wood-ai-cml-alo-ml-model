from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from datetime import datetime
from app.schemas import HealthResponse, UploadResponse, ScoreResponse, ForecastOutput
from app.utils import validate_cml_dataframe, calculate_inspection_schedule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wood AI CML ALO API",
    version="0.3.0",
    description="ML API for CML optimization",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "cml_elimination_model.joblib"

model = None
try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["corrosion_thickness_ratio"] = df["average_corrosion_rate"] / df["thickness_mm"]
    min_thickness = 3.0
    df["remaining_life_years"] = (df["thickness_mm"] - min_thickness) / df[
        "average_corrosion_rate"
    ]
    df["remaining_life_years"] = df["remaining_life_years"].clip(lower=0)
    if "last_inspection_date" in df.columns:
        df["last_inspection_date"] = pd.to_datetime(
            df["last_inspection_date"], errors="coerce"
        )
        df["days_since_inspection"] = (
            pd.Timestamp.now() - df["last_inspection_date"]
        ).dt.days
        df["days_since_inspection"] = df["days_since_inspection"].fillna(365)
    else:
        df["days_since_inspection"] = 365
    if "risk_score" not in df.columns:
        df["risk_score"] = (
            df["average_corrosion_rate"] * 20 + (10 - df["thickness_mm"]) * 5
        ).clip(0, 100)
    return df


@app.get("/")
async def root():
    return {
        "message": "Wood AI CML Optimization API",
        "version": "0.3.0",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "version": "0.3.0",
    }


@app.get("/model-info")
async def model_info():
    """Return information about the loaded ML model."""
    if model is None:
        return {
            "model_loaded": False,
            "message": "No model loaded",
            "available_models": [
                "random_forest",
                "gradient_boosting",
                "xgboost",
            ],
        }

    model_type = type(model).__name__
    classifier_name = "Unknown"
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier_name = type(model.named_steps["classifier"]).__name__

    # Try to load metadata if available
    metadata_files = list((BASE_DIR / "models").glob("model_metadata_*.json"))
    metrics = {}
    if metadata_files:
        latest_metadata = sorted(metadata_files)[-1]
        try:
            import json

            with open(latest_metadata) as f:
                metadata = json.load(f)
                metrics = metadata.get("metrics", {})
        except Exception:
            pass

    return {
        "model_loaded": True,
        "model_type": model_type,
        "classifier": classifier_name,
        "model_path": str(MODEL_PATH),
        "metrics": metrics,
        "available_models": [
            "random_forest",
            "gradient_boosting",
            "xgboost",
        ],
    }


@app.post("/upload-cml-data", response_model=UploadResponse)
async def upload_cml_data(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(400, "Unsupported file format")
        result = validate_cml_dataframe(df)
        logger.info(
            f"Successfully parsed {file.filename}: {len(df)} rows, {len(df.columns)} columns"
        )
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict("records"),
            "message": f"Successfully uploaded {len(df)} records",
            "validation": result,
        }
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(500, f"Error processing file: {str(e)}")


@app.post("/score-cml-data", response_model=ScoreResponse)
async def score_cml_data(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "ML model not loaded")
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(400, "Unsupported file format")
        logger.info(f"Scoring {len(df)} CMLs from {file.filename}")
        df = engineer_features(df)
        feature_cols = [
            "average_corrosion_rate",
            "thickness_mm",
            "commodity",
            "feature_type",
            "cml_shape",
            "remaining_life_years",
            "corrosion_thickness_ratio",
            "risk_score",
            "days_since_inspection",
        ]
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X = df[feature_cols]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        results = []
        for idx, row in df.iterrows():
            results.append(
                {
                    "id_number": str(row["id_number"]),
                    "predicted_elimination_flag": int(predictions[idx]),
                    "elimination_probability": float(probabilities[idx]),
                    "recommendation": "ELIMINATE" if predictions[idx] == 1 else "KEEP",
                    "confidence": "HIGH"
                    if abs(probabilities[idx] - 0.5) > 0.3
                    else "MODERATE",
                }
            )
        logger.info(f"Successfully scored {len(df)} CMLs")
        return {
            "rows_scored": len(df),
            "results": results[:100],
            "total_results": len(results),
            "model_info": {
                "model_type": type(model).__name__,
                "features_used": feature_cols,
            },
            "message": f"Successfully scored {len(df)} CML records",
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Error during prediction: {str(e)}")


@app.post("/forecast-remaining-life", response_model=List[ForecastOutput])
async def forecast_remaining_life(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(400, "Unsupported file format")

        logger.info(f"Forecasting for {len(df)} CMLs from {file.filename}")

        # Validate data first
        validation = validate_cml_dataframe(df)
        if not validation["valid"]:
            logger.warning(
                f"Validation failed for {file.filename}: {validation['errors']}"
            )
            # We might want to still proceed or raise error. For now, let's proceed but log.

        results = []
        for _, row in df.iterrows():
            try:
                # Map CSV columns to function arguments
                # Assuming CSV has 'average_corrosion_rate' and 'thickness_mm'
                schedule = calculate_inspection_schedule(
                    corrosion_rate=float(row["average_corrosion_rate"]),
                    thickness=float(row["thickness_mm"]),
                    min_thickness=3.0,  # Default or could be from request
                    safety_factor=1.5,
                )

                results.append(
                    ForecastOutput(
                        id_number=str(row["id_number"]),
                        remaining_life_years=schedule["remaining_life_years"],
                        next_inspection_date=schedule["next_inspection_date"],
                        estimated_thickness_at_next_inspection=schedule[
                            "estimated_thickness_at_next_inspection"
                        ],
                        recommended_inspection_frequency_months=schedule[
                            "inspection_interval_months"
                        ],
                        risk_level=schedule["risk_level"],
                    )
                )
            except Exception as row_e:
                logger.warning(f"Skipping row due to error: {row_e}")
                continue

        logger.info(f"Successfully generated forecasts for {len(results)} CMLs")
        return results

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(500, f"Error during forecasting: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
