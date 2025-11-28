import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
import statistics

# Configuration
NUM_NORMAL_USERS = 5000
NUM_SMURF_USERS = 200 # Imbalanced dataset (Real life scenario)
OUTPUT_FILE = r"aml-detection-engine/modules/Smurfing_detection_V2/data/training_data.csv"

class DataGenerator:
    def __init__(self):
        self.data = []

    def _generate_normal_behavior(self, user_id):
        """
        Normal users: Random amounts, random times, low frequency.
        """
        num_tx = random.randint(5, 50)
        # Start date: 30 days ago
        base_time = datetime.now() - timedelta(days=30)
        
        history = []
        for _ in range(num_tx):
            # Advance time randomly (1 hour to 3 days gaps)
            base_time += timedelta(hours=random.randint(1, 72))
            
            # Normal amounts ($10 to $5000) - messy numbers
            amt = round(random.uniform(10, 5000), 2)
            
            history.append({
                "timestamp": base_time,
                "amount": amt
            })
        return history

    def _generate_smurf_behavior(self, user_id):
        """
        Smurfs: Bursts of activity, structuring amounts ($9000-$9900).
        """
        num_tx = random.randint(10, 30)
        base_time = datetime.now() - timedelta(days=random.randint(1, 5))
        
        history = []
        for _ in range(num_tx):
            # Fast! (1 minute to 10 minutes gaps)
            base_time += timedelta(minutes=random.randint(1, 10))
            
            # Structuring Amounts (Just below $10k)
            # 80% chance of being high risk, 20% mixed in to hide
            if random.random() > 0.2:
                amt = round(random.uniform(9000, 9900), 2)
            else:
                amt = round(random.uniform(100, 1000), 2)
                
            history.append({
                "timestamp": base_time,
                "amount": amt
            })
        return history

    def _calculate_rolling_features(self, history, current_index):
        """
        SIMULATES THE REDIS LOGIC IN PYTHON.
        Looks at the 'past' relative to the current index.
        """
        current_tx = history[current_index]
        current_time = current_tx['timestamp']
        
        # 1. Filter: Get transactions from the last 24h relative to THIS transaction
        past_24h = []
        for i in range(current_index - 1, -1, -1):
            prev_tx = history[i]
            time_diff = (current_time - prev_tx['timestamp']).total_seconds()
            
            if time_diff <= 86400: # 24 hours
                past_24h.append(prev_tx['amount'])
            else:
                break # Stop looking further back
        
        # 2. Engineer Features (Must match feature_engine.py EXACTLY)
        # Note: We add +1 to velocity to include the current transaction effectively
        # or we treat the window as "history before this". 
        # Let's match the Redis logic: The window contains the current tx.
        
        window = past_24h + [current_tx['amount']]
        
        velocity_24h = len(window)
        structuring_count = sum(1 for x in window if 9000 <= x < 10000)
        std_dev = statistics.stdev(window) if len(window) > 1 else 0
        
        return [velocity_24h, structuring_count, std_dev, current_tx['amount']]

    def run(self):
        print("🧪 Generating Synthetic Transactions...")
        
        # 1. Generate Normal Users
        for i in range(NUM_NORMAL_USERS):
            uid = f"NORM_{i}"
            history = self._generate_normal_behavior(uid)
            
            # Process sequence
            for idx in range(len(history)):
                feats = self._calculate_rolling_features(history, idx)
                feats.append(0) # Label: Normal
                self.data.append(feats)

        # 2. Generate Smurfs
        for i in range(NUM_SMURF_USERS):
            uid = f"SMURF_{i}"
            history = self._generate_smurf_behavior(uid)
            
            # Process sequence
            for idx in range(len(history)):
                feats = self._calculate_rolling_features(history, idx)
                feats.append(1) # Label: Smurf
                self.data.append(feats)

        # 3. Save
        columns = ["velocity_24h", "structuring_count_24h", "std_dev_24h", "amount", "is_smurf"]
        df = pd.DataFrame(self.data, columns=columns)
        
        # Create dir if not exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Generated {len(df)} transactions. Saved to {OUTPUT_FILE}")
        print(df.head())

if __name__ == "__main__":
    gen = DataGenerator()
    gen.run()