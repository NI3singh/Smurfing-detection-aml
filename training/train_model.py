import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os

# Configuration
INPUT_FILE = r"C:\Users\ELaunch\OneDrive\Desktop\AML_ENGINE\aml-detection-engine\modules\Smurfing_detection_V2\data\training_data.csv"
# We save the model directly into the app folder so the API can see it immediately
OUTPUT_MODEL = r"C:\Users\ELaunch\OneDrive\Desktop\AML_ENGINE\aml-detection-engine\modules\Smurfing_detection_V2\models\xgb_smurf_v1.json"

def train():
    print("🧠 Starting Model Training...")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Data file {INPUT_FILE} not found. Run generate_data.py first!")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # 2. Separate Features (X) and Target (y)
    # Features must match the order in feature_engine.py:
    # [velocity_24h, structuring_count_24h, std_dev_24h, amount]
    X = df.drop(columns=['is_smurf'])
    y = df['is_smurf']
    
    # 3. Split Data (80% Train, 20% Test)
    # stratify=y ensures we keep the same ratio of Smurfs in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Initialize XGBoost Classifier
    # scale_pos_weight is CRITICAL for imbalanced data (Smurfs are rare)
    # It tells the model: "Pay more attention to the '1' class"
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=10, 
        eval_metric='auc'
    )
    
    # 5. Train
    print(f"   Training on {len(X_train)} records...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    print("\n📊 Model Evaluation:")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, preds))
    print(f"   AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # 7. Save Model
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    
    model.save_model(OUTPUT_MODEL)
    print(f"\n✅ Model saved to: {OUTPUT_MODEL}")
    print("   The API will now be able to detect Smurfs.")

if __name__ == "__main__":
    train()