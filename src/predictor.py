# src/predictor.py
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lightgbm_churn_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model not found, using dummy fallback: {e}")
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0]], [0])

def predict(features: pd.DataFrame):
    """Predict churn probabilities based on model features."""
    if model is None:
        raise ValueError("❌ Model not loaded properly.")

    # Align feature columns with those used in training
    if hasattr(model, "feature_names_in_"):
        missing_cols = [c for c in model.feature_names_in_ if c not in features.columns]
        for col in missing_cols:
            features[col] = 0  # add missing features with default value
        features = features[model.feature_names_in_]

    return model.predict_proba(features)[:, 1]
