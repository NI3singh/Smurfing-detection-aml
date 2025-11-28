import xgboost as xgb
import os
import logging
from app.core.config import settings

# Setup logging to see errors in Docker console
logger = logging.getLogger("uvicorn")

class SmurfDetector:
    def __init__(self):
        """
        Initialize the XGBoost Classifier.
        Tries to load the pre-trained model file defined in config.
        """
        self.model = xgb.XGBClassifier()
        self.model_loaded = False
        
        if os.path.exists(settings.MODEL_PATH):
            try:
                self.model.load_model(settings.MODEL_PATH)
                self.model_loaded = True
                logger.info(f"✅ Model loaded successfully from {settings.MODEL_PATH}")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
        else:
            logger.warning(f"⚠️  Model file not found at {settings.MODEL_PATH}. System running in 'Blind' mode.")

    def predict(self, features) -> float:
        """
        Performs inference.
        Input: Numpy array of features (1, 4)
        Output: Probability of Smurfing (0.0 to 1.0)
        """
        if not self.model_loaded:
            # Fallback if training hasn't happened yet
            return 0.0
            
        try:
            # predict_proba returns a list of probabilities for each class:
            # [[Prob_Normal, Prob_Smurf]] -> [[0.05, 0.95]]
            # We want the second number (index 1), which is probability of Smurf.
            probs = self.model.predict_proba(features)
            return float(probs[0][1])
            
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return 0.0