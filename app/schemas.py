from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Literal

# ==========================================
# INPUT SCHEMA (What the Bank sends us)
# ==========================================
class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique ID of the transaction from the Bank")
    user_id: str = Field(..., description="The ID of the user performing the transaction")
    
    # Validation: Amount must be greater than 0
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    
    # Validation: Must be ISO format (e.g., "2024-01-01T12:00:00")
    timestamp: datetime = Field(..., description="Exact time of transaction")
    
    # Critical for SMoTeF (Network) logic
    counterparty_id: str = Field(..., description="The ID of the person receiving/sending money")
    
    # Optional field (default to DEPOSIT if not provided)
    type: Literal['DEPOSIT', 'TRANSFER', 'WITHDRAWAL'] = "DEPOSIT"

    # Example for documentation (Swagger UI)
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_123456789",
                "user_id": "USER_99",
                "amount": 9500.00,
                "timestamp": "2024-11-28T10:30:00",
                "counterparty_id": "USER_55",
                "type": "DEPOSIT"
            }
        }

# ==========================================
# OUTPUT SCHEMA (What we send back)
# ==========================================
class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., description="Probability of Smurfing (0.0 to 1.0)")
    
    # The decision: BLOCK, REVIEW, or ALLOW
    action: Literal['BLOCK', 'REVIEW', 'ALLOW']
    
    # Explanation for the analyst
    reason: str
    
    # Latency tracking (how long we took to process)
    process_time_ms: float