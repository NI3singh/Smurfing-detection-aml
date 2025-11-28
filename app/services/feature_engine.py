import redis
import json
import numpy as np
import statistics
from datetime import datetime
from app.core.config import settings
from app.schemas import TransactionRequest

class FeatureEngine:
    def __init__(self):
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True # Returns strings instead of bytes
        )
        
    def _get_user_key(self, user_id: str) -> str:
        """Create a namespaced key like 'user:USER_123:history'"""
        return f"user:{user_id}:history"

    def _update_and_fetch_window(self, tx: TransactionRequest) -> list:
        """
        Updates the sliding window in Redis and returns the active history.
        Uses Redis Sorted Sets (ZSET) for efficient time-based storage.
        """
        key = self._get_user_key(tx.user_id)
        current_ts = tx.timestamp.timestamp() # Unix timestamp (float)
        
        # 1. Add current transaction to the Set
        # We store the Amount as the "Member" and Timestamp as the "Score".
        # Note: In a real system, we'd store a JSON string of the whole tx. 
        # For this logic, we just need the amount. To handle duplicate amounts, 
        # we append the tx_id: "9000.00:TXN_123"
        member_value = f"{tx.amount}:{tx.transaction_id}"
        self.redis_client.zadd(key, {member_value: current_ts})
        
        # 2. Prune History (Sliding Window)
        # Remove anything older than 24 hours (86400 seconds)
        cutoff_time = current_ts - 86400
        self.redis_client.zremrangebyscore(key, min="-inf", max=cutoff_time)
        
        # 3. Fetch valid history (All remaining items)
        # Returns list like ["100.0:TX1", "9500.0:TX2", ...]
        active_history = self.redis_client.zrange(key, 0, -1)
        
        # 4. Set Key Expiration (Safety net)
        # If user stops transacting, delete their key after 48 hours to save RAM
        self.redis_client.expire(key, 172800) 
        
        return active_history

    def engineer_features(self, tx: TransactionRequest) -> np.ndarray:
        """
        Main function called by the API.
        Returns a 1D array ready for XGBoost.
        """
        # 1. Get History (State)
        raw_history = self._update_and_fetch_window(tx)
        
        # 2. Extract Amounts (Parse "Amount:ID" strings)
        amounts = [float(x.split(':')[0]) for x in raw_history]
        
        # 3. Calculate Real-Time Features
        
        # A. Velocity (How many txs in last 24h?)
        velocity_24h = len(amounts)
        
        # B. Volume (Total money moved)
        total_amount_24h = sum(amounts)
        
        # C. Average Amount
        avg_amount_24h = total_amount_24h / velocity_24h if velocity_24h > 0 else 0
        
        # D. Standard Deviation (Robotic precision check)
        # If only 1 transaction, std_dev is 0
        std_dev_amount_24h = statistics.stdev(amounts) if velocity_24h > 1 else 0
        
        # E. Structuring Count (The Smurf Trap)
        # Count deposits between $9,000 and $9,999
        structuring_count_24h = sum(1 for x in amounts if 9000 <= x < 10000)
        
        # 4. Create Feature Vector
        # MUST match the exact order your model was trained on!
        # [velocity, structuring, std_dev, current_amount]
        features = np.array([
            velocity_24h, 
            structuring_count_24h, 
            std_dev_amount_24h, 
            tx.amount
        ]).reshape(1, -1)
        
        return features