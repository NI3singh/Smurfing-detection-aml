import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import logging

# Setup Logging
logger = logging.getLogger("smurf_predictor")
logger.setLevel(logging.INFO)

class PredictionService:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Load the trained AI models from disk.
        We do this ONCE on startup to keep the API fast.
        """
        try:
            # Construct paths relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "../../models")
            
            logger.info(f"📥 Loading AI artifacts from {model_dir}...")
            
            self.model = joblib.load(os.path.join(model_dir, "isolation_forest.pkl"))
            self.explainer = joblib.load(os.path.join(model_dir, "shap_explainer.pkl"))
            
            with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
                self.metadata = json.load(f)
                
            logger.info("✅ AI Models loaded successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"❌ CRITICAL: Could not load models. Did you run Phase 3? Error: {e}")
            raise e

    def predict(self, user_features: pd.DataFrame):
        """
        Run the full inference pipeline on a single user's features.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        # 1. Run Anomaly Detection
        # Sklearn returns -1 for Anomaly, 1 for Normal
        prediction_code = self.model.predict(user_features)[0]
        score = self.model.decision_function(user_features)[0]
        
        is_anomaly = True if prediction_code == -1 else False
        
        # 2. Run SHAP Explainability
        # This tells us WHICH features pushed the score towards anomaly
        shap_values = self.explainer.shap_values(user_features)
        
        # SHAP returns a matrix, we want the first (and only) row
        # Note: Depending on SHAP version, it might return a list or array. 
        # For TreeExplainer with IF, it usually returns raw values.
        if isinstance(shap_values, list):
            shap_values = shap_values[0] # Handle rare edge case
            
        # 3. Interpret Reasons
        # We zip the feature names with their SHAP impact
        feature_names = self.metadata["feature_names"]
        
        # Calculate contribution (Magnitude of impact)
        contributions = zip(feature_names, shap_values[0])
        
        # Sort by impact (Absolute value - biggest drivers first)
        # We only care about reasons if it IS an anomaly
        reasons = []
        if is_anomaly:
            # Sort by negative impact (driving score lower towards anomaly) or just magnitude
            sorted_impacts = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
            
            # Get Top 3 Reasons
            for feature, impact in sorted_impacts[:3]:
                reasons.append({
                    "feature": feature,
                    "impact_score": float(f"{impact:.4f}"),
                    "value": float(user_features[feature].values[0]) # The actual data value
                })

        # 4. Construct Final Verdict
        return {
            "is_anomaly": is_anomaly,
            "risk_score": float(f"{-score:.4f}"), # Invert score so higher = riskier
            "confidence": "High" if is_anomaly else "Low", # Simplified for now
            "top_reasons": reasons
        }

# Global Instance
predictor = PredictionService()