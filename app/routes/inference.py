from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Import our services
# We need to hack sys.path slightly to reach the simulation folder for feature logic reuse
# In a real monorepo, these would be installed packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../simulation")))
from feature_engineer import FeatureEngineer
from app.services.prediction import predictor

router = APIRouter()

# Initialize Feature Engineer Helper
# (We reuse the class we wrote in Phase 2!)
feat_engineer = FeatureEngineer()

# --- Request Model ---
class AnalysisRequest(BaseModel):
    user_id: str
    transaction_id: str = None # Optional, for logging context

# --- Mock Database Loader ---
# In production, this would query SQL/MongoDB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "../../data/raw/transactions.csv")

def get_user_history_mock(user_id: str):
    """
    Simulates fetching user history from a database.
    Reads from the raw CSV for demo purposes.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("Raw data not found. Please run Phase 1 generator.")
        
    # Read CSV (Optimized: we might want to cache this in memory for the demo to be fast)
    # For now, we read full CSV. In prod, SELECT * FROM tx WHERE user_id = ...
    df = pd.read_csv(RAW_DATA_PATH)
    
    user_df = df[df['user_id'] == user_id].copy()
    
    if user_df.empty:
        return None
        
    return user_df

@router.post("/analyze_user", tags=["Smurfing Detection"])
async def analyze_user(request: AnalysisRequest):
    """
    Real-time Smurfing Detection Endpoint.
    1. Fetches user history.
    2. Calculates behavioral features (12-dim).
    3. Runs AI Model (Isolation Forest).
    4. Explains decision (SHAP).
    """
    try:
        # 1. Fetch History
        history_df = get_user_history_mock(request.user_id)
        if history_df is None:
            raise HTTPException(status_code=404, detail="User not found in transaction history.")
            
        # 2. Feature Engineering (On-the-fly)
        # We need to preprocess timestamps first as per FeatureEngineer requirements
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['hour'] = history_df['timestamp'].dt.hour
        history_df['date'] = history_df['timestamp'].dt.date
        history_df['is_round_number'] = (history_df['amount'] % 1 == 0).astype(int)
        
        # Calculate features using the class methods
        # Note: The engineer returns a DF with all users. We pass just one.
        features = feat_engineer.engineer_features(history_df)
        
        # Drop non-feature columns (labels) that appear in training but not inference
        features = features.drop(columns=['is_smurf', 'smurf_type'], errors='ignore')
        
        # 3. Predict & Explain
        result = predictor.predict(features)
        
        return {
            "status": "success",
            "user_id": request.user_id,
            "analysis": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



# -----------------------------------(Production Version Logic)-------------------------------------------

# from sqlalchemy import create_engine, text
# import pandas as pd

# # 1. Setup DB Connection (Do this in a config file in reality)
# DB_URL = "postgresql://user:pass@localhost:5432/betting_db"
# engine = create_engine(DB_URL)

# def get_user_history_production(user_id: str):
#     """
#     Production Loader: Queries the live database for the user's history.
#     """
#     # 2. The SQL Query
#     # We fetch ALL transactions for this user.
#     # In high-scale systems, we might limit this to "Last 90 Days" to stay fast.
#     query = text("""
#         SELECT 
#             transaction_id,
#             user_id,
#             timestamp,
#             amount,
#             type,
#             device_id,
#             ip_address,
#             related_user_id
#         FROM transactions
#         WHERE user_id = :uid
#         ORDER BY timestamp ASC
#     """)
    
#     # 3. Execute & Convert to Pandas
#     with engine.connect() as conn:
#         df = pd.read_sql(query, conn, params={"uid": user_id})
    
#     if df.empty:
#         return None
        
#     # 4. Type Enforcement (Critical for Feature Engineer)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['amount'] = df['amount'].astype(float)
    
#     return df