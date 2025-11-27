"""
Phase 3, Step 2: The Model Pipeline (Isolation Forest)
We will initialize the Isolation Forest algorithm.
Why: It is efficient, handles high-dimensional data (our 12 features) well, and doesn't require a balanced dataset.
Configuration: We won't just use default settings. We will tune:
n_estimators (Number of trees).
max_samples (How much data each tree sees).
contamination (Expected fraud rate).

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import shap
import json
import os
import sys

# Ensure we can run from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../models")

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

def train_isolation_forest(X_train, contamination):
    """
    Step 2: Initialize and Train the Model.
    """
    print("\n🤖 Initializing Isolation Forest...")
    
    # Tuned Hyperparameters for Financial Fraud
    model = IsolationForest(
        n_estimators=200,        # More trees = more stable decision boundaries
        max_samples=256,         # Classic IF setting (limit sample size per tree)
        contamination=contamination, # The exact % of fraud we expect
        random_state=42,
        n_jobs=-1                # Use all CPU cores
    )
    
    print("   Training model on Normal Behavior...")
    model.fit(X_train)
    print("   ✓ Training Complete.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates how well the model catches Smurfs it has NEVER seen before.
    """
    print("\nscale⚖️ Evaluating Model Performance...")
    
    # Predict (Returns -1 for Outlier, 1 for Inlier)
    y_pred_iso = model.predict(X_test)
    
    # Convert IF output (-1/1) to our Binary Label (1/0)
    # -1 (Anomaly) -> 1 (Smurf)
    #  1 (Normal)  -> 0 (Normal)
    y_pred = [1 if x == -1 else 0 for x in y_pred_iso]
    
    # Metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Smurf']))
    
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"🎯 Smurf Detection Rate (Recall): {recall:.2%}")
    print(f"🛡️ False Alarm Rate (1 - Precision): {1 - precision:.2%}")

    return recall

def train_shap_explainer(model):
    """Step 4: Train SHAP Explainer for Interpretation."""
    print("\n🧠 Initializing SHAP Explainer...")
    # TreeExplainer is highly optimized for Isolation Forests
    explainer = shap.TreeExplainer(model)
    print("   ✓ Explainer ready.")
    return explainer

def save_artifacts(model, explainer, feature_names):
    
    """Saves the trained model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    explainer_path = os.path.join(MODEL_DIR, "shap_explainer.pkl")
    
    # 1. Save Model
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    # 2. Save Explainer
    joblib.dump(explainer, explainer_path)
    print(f"\n💾 Explainer saved to: {explainer_path}")

    metadata = {
        "feature_names": list(feature_names),
        "model_version": "1.0.0",
        "threshold": float(model.offset_) # The decision boundary score
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    # 1. Load & Prep
    full_df, X_features, metadata = load_and_sanitize_data(DATA_PATH)
    contamination = calculate_contamination(metadata)
    
    # 2. Split
    X_train, X_test, y_test = perform_stratified_split(full_df, X_features)
    
    # 3. Train
    model = train_isolation_forest(X_train, contamination)
    
    # 4. Evaluate
    recall = evaluate_model(model, X_test, y_test)
    
    # 5. Explainability (Only if model is good)
    if recall > 0.8:
        explainer = train_shap_explainer(model)
        
        # 6. Save Everything
        save_artifacts(model, explainer, X_features.columns)
        print("\n✅ Phase 3 Complete. Ready for Inference API.")
    else:
        print("\n❌ Model performance too low. Aborting save.")