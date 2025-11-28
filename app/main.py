from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
import time

# Import our custom modules
from app.core.config import settings
from app.core.security import get_api_key
from app.schemas import TransactionRequest, PredictionResponse
from app.services.feature_engine import FeatureEngine
from app.services.detector import SmurfDetector

# Global variables to hold our services
# We don't initialize them yet; we do that on startup
feature_engine = None
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    LIFESPAN EVENTS:
    Code here runs ONCE when the server starts.
    This is where we load the Heavy AI Model into RAM.
    """
    global feature_engine, detector
    
    print("🚀 System Starting... Initializing Services...")
    feature_engine = FeatureEngine() # Connects to Redis
    detector = SmurfDetector()       # Loads XGBoost Model
    
    yield # The application runs here
    
    print("🛑 System Shutting Down...")
    # Clean up connections if needed

# Initialize FastAPI with the lifespan logic
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# ==========================================
# ROUTES
# ==========================================

@app.get("/", tags=["Health"])
def health_check():
    """
    Simple ping to check if server is alive.
    Used by Docker/Kubernetes to restart container if it freezes.
    """
    return {"status": "active", "model_loaded": detector.model_loaded}

@app.post(
    "/scan_transaction", 
    response_model=PredictionResponse,
    dependencies=[Depends(get_api_key)], # 🔒 Security Lock
    tags=["Detection"]
)
def scan_transaction(tx: TransactionRequest):
    """
    The Main Endpoint.
    1. Receives Transaction
    2. Calculates Features (Redis)
    3. Predicts Smurfing Probability (XGBoost)
    4. Returns Decision
    """
    start_time = time.time()
    
    # 1. Feature Engineering
    # This talks to Redis and gets the sliding window stats
    features = feature_engine.engineer_features(tx)
    
    # 2. Inference
    # Pass the vector to the loaded model
    risk_score = detector.predict(features)
    
    # 3. Business Logic (Decision Matrix)
    action = "ALLOW"
    reason = "Normal Behavior"
    
    if risk_score >= settings.BLOCK_THRESHOLD:
        action = "BLOCK"
        reason = "High Probability of Smurfing/Structuring"
    elif risk_score >= settings.REVIEW_THRESHOLD:
        action = "REVIEW"
        reason = "Suspicious Velocity or Amount Pattern"
        
    # 4. Latency Calculation
    process_time = (time.time() - start_time) * 1000 # in ms
    
    return PredictionResponse(
        transaction_id=tx.transaction_id,
        risk_score=round(risk_score, 4),
        action=action,
        reason=reason,
        process_time_ms=round(process_time, 2)
    )