"""
Phase 3, Step 1: Data Preprocessing & Splitting Strategy.
Prepares the 12-dimensional feature set for Anomaly Detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Ensure we can run from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/features.csv")

def load_and_sanitize_data(filepath):
    """
    Step 1.1: Load Data & Sanitize
    Removes non-mathematical columns (IDs) but keeps labels for validation.
    """
    print(f"📥 Loading features from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    
    # Separate Metadata (IDs) from Features (Math)
    # We need metadata later to identify WHICH user was flagged
    metadata_cols = ['user_id', 'smurf_type']
    # Check if user_id is the index or a column
    if 'user_id' not in df.columns and df.index.name == 'user_id':
        df = df.reset_index()
        
    # Drop non-feature columns if they exist
    drop_cols = [c for c in metadata_cols if c in df.columns]
    metadata = df[drop_cols + ['is_smurf']].copy()
    
    # The "X" (Features) - Drop Labels and Metadata
    # We keep only the 12 features we engineered
    features = df.drop(columns=drop_cols + ['is_smurf'], errors='ignore')
    
    # Ensure all data is numeric
    features = features.select_dtypes(include=[np.number])
    
    print(f"   ✓ Loaded {len(df)} rows.")
    print(f"   ✓ Feature Matrix Shape: {features.shape} (Should be 12 features)")
    
    return df, features, metadata

def calculate_contamination(metadata):
    """
    Step 1.2: Calculate Contamination Ratio
    This is CRITICAL for Isolation Forest. It tells the model 
    "Roughly 3% of this data is garbage, find it."
    """
    total_count = len(metadata)
    smurf_count = metadata['is_smurf'].sum()
    contamination = smurf_count / total_count
    
    print(f"\n📊 Contamination Analysis:")
    print(f"   • Total Users: {total_count}")
    print(f"   • Smurfs: {smurf_count}")
    print(f"   • Contamination Ratio: {contamination:.4f} ({contamination*100:.2f}%)")
    
    return contamination

def perform_stratified_split(df, features, test_size=0.2):
    """
    Step 1.3: The "Normal-Only" Split Strategy
    
    STRATEGY:
    1. Training Set: Composed of 100% NORMAL users (plus extremely few anomalies 
       if we want to simulate real-world noise, but pure normal is best for 
       learning 'normality').
    2. Test Set: Contains the remaining Normal users + ALL Smurfs.
    
    Why? We want to see if the model flags Smurfs it has NEVER seen before.
    """
    print("\n✂️ Performing Anomaly Detection Split...")
    
    # 1. Separate Normals and Smurfs
    normal_mask = df['is_smurf'] == False
    smurf_mask = df['is_smurf'] == True
    
    X_normal = features[normal_mask]
    X_smurf = features[smurf_mask]
    
    # 2. Split Normal Users (e.g., 80% Train, 20% Test)
    X_train, X_test_normal = train_test_split(X_normal, test_size=test_size, random_state=42)
    
    # 3. Construct Final Test Set (Remaining Normals + ALL Smurfs)
    # This mimics production: The system sees mostly normals, and suddenly smurfs appear.
    X_test = pd.concat([X_test_normal, X_smurf])
    
    # Create Labels for Test Set (0=Normal, 1=Smurf) for evaluation metrics later
    y_test_normal = np.zeros(len(X_test_normal))
    y_test_smurf = np.ones(len(X_smurf))
    y_test = np.concatenate([y_test_normal, y_test_smurf])
    
    print(f"   ✓ Training Set: {len(X_train)} users (100% Normal)")
    print(f"   ✓ Test Set:     {len(X_test)} users ({len(X_smurf)} Smurfs hidden inside)")
    
    return X_train, X_test, y_test

if __name__ == "__main__":
    # 1. Load
    full_df, X_features, metadata = load_and_sanitize_data(DATA_PATH)
    
    # 2. Contamination
    contamination = calculate_contamination(metadata)
    
    # 3. Split
    X_train, X_test, y_test = perform_stratified_split(full_df, X_features)
    
    print("\n✅ Step 1 Complete. Data is ready for Isolation Forest.")