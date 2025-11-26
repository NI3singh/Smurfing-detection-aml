"""
Feature Engineering Pipeline for AML Detection.
Transforms raw transaction logs into user-level behavioral features for ML models.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Extracts behavioral fingerprints from transaction history.
    Each feature targets specific money laundering red flags.
    """
    
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon  # Prevents division by zero
        
    def load_data(self, filepath):
        """Load and preprocess raw transaction data."""
        print(f"📂 Loading data from {filepath}...")
        
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found at {filepath}")
            print("   Make sure you ran generator.py first!")
            sys.exit(1)
            
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        # Identify round numbers (integers)
        df['is_round_number'] = (df['amount'] % 1 == 0).astype(int)
        
        print(f"   ✓ Loaded {len(df):,} transactions from {df['user_id'].nunique():,} users")
        return df
    
    def _calculate_structuring_features(self, df):
        """
        STRUCTURING DETECTION: Identify threshold-avoiding behavior.
        Red Flag: Repeated deposits just below $10k CTR reporting threshold.
        """
        deposits = df[df['type'] == 'DEPOSIT'].copy()
        features = pd.DataFrame(index=df['user_id'].unique())
        
        # Feature 1: Count of suspicious structuring deposits (9000-9999)
        structuring = deposits[
            (deposits['amount'] >= 9000) & (deposits['amount'] < 10000)
        ].groupby('user_id').size()
        features['structuring_count'] = structuring
        
        # Feature 2: Standard deviation of deposit amounts (robotic precision)
        features['amount_std_dev'] = deposits.groupby('user_id')['amount'].std()
        
        # Feature 3: Round number ratio (criminals avoid round numbers)
        round_numbers = deposits.groupby('user_id')['is_round_number'].mean()
        features['round_number_ratio'] = round_numbers
        
        return features
    
    def _calculate_network_features(self, df):
        """
        NETWORK ANALYSIS: Detect fan-in/fan-out money movement patterns.
        Red Flag: Many counterparties or imbalanced flow ratios.
        """
        features = pd.DataFrame(index=df['user_id'].unique())
        
        # Calculate total inflows and outflows per user
        # Inflows: Money coming IN (Deposits, Wins, Incoming Transfers)
        inflows = df[df['type'].isin(['DEPOSIT', 'BET_WIN', 'TRANSFER'])].groupby('user_id')['amount'].sum()
        
        # Outflows: Money going OUT (Withdrawals, Losses)
        outflows = df[df['type'].isin(['WITHDRAWAL', 'BET_LOSS'])].groupby('user_id')['amount'].sum()
        
        # Feature 4: Flow ratio (Total In / Total Out)
        # Normal users: ~1.0 (balanced), Smurfs: Very high or very low
        features['flow_ratio'] = inflows / (outflows + self.epsilon)
        
        # Feature 5: Distinct counterparties (for transfers/chip dumping)
        counterparties = df[df['related_user_id'].notna()].groupby('user_id')['related_user_id'].nunique()
        features['distinct_counterparties'] = counterparties
        
        return features
    
    def _calculate_gaming_features(self, df):
        """
        GAMING BEHAVIOR: Real gamblers play, money launderers just wash.
        Red Flag: Low betting activity despite high deposits.
        """
        features = pd.DataFrame(index=df['user_id'].unique())
        
        # Calculate key gaming metrics per user
        total_deposited = df[df['type'] == 'DEPOSIT'].groupby('user_id')['amount'].sum()
        total_wagered = df[df['type'].isin(['BET_WIN', 'BET_LOSS'])].groupby('user_id')['amount'].sum()
        total_won = df[df['type'] == 'BET_WIN'].groupby('user_id')['amount'].sum()
        total_lost = df[df['type'] == 'BET_LOSS'].groupby('user_id')['amount'].sum()
        
        # Feature 6: Wager ratio (Total Wagered / Total Deposited)
        # Normal users: >0.9 (high churn), Smurfs: <0.2 (just washing)
        features['wager_ratio'] = total_wagered / (total_deposited + self.epsilon)
        
        # Feature 7: Win/Loss ratio
        # Normal users: ~0.48 (house edge), Chip dumpers: 0.05 (intentional losses)
        features['win_loss_ratio'] = total_won / (total_lost + self.epsilon)
        
        return features
    
    def _calculate_identity_features(self, df):
        """
        IDENTITY & BEHAVIOR: Device rotation and temporal anomalies.
        Red Flag: Many devices/IPs, odd-hour activity, high velocity.
        """
        features = pd.DataFrame(index=df['user_id'].unique())
        
        # Feature 8: Unique device count
        # Normal users: 1-2, Smurfs: 10+ (constant rotation)
        features['unique_device_count'] = df.groupby('user_id')['device_id'].nunique()
        
        # Feature 9: Unique IP count
        # Normal users: 1-3, Smurfs: 20+ (VPN/proxy rotation)
        features['unique_ip_count'] = df.groupby('user_id')['ip_address'].nunique()
        
        # Feature 10: Night owl ratio (00:00-05:00 activity)
        # Normal users: <0.05, Smurfs/Bots: >0.3 (24/7 operation)
        night_tx = df[df['hour'].between(0, 5)].groupby('user_id').size()
        total_tx = df.groupby('user_id').size()
        features['night_owl_ratio'] = night_tx / (total_tx + self.epsilon)
        
        # Feature 11: Transaction velocity (avg tx per day)
        # Normal users: 1-5, Smurfs: 20+ (high-frequency bot)
        user_dates = df.groupby('user_id')['date'].agg(['min', 'max', 'count'])
        days_active = (pd.to_datetime(user_dates['max']) - pd.to_datetime(user_dates['min'])).dt.days + 1
        features['velocity_24h'] = user_dates['count'] / (days_active + self.epsilon)
        
        return features
    
    def _calculate_volume_features(self, df):
        """
        VOLUME METRICS: Transaction size and frequency patterns.
        Completes the 12-feature requirement.
        """
        features = pd.DataFrame(index=df['user_id'].unique())
        
        # Feature 12: Average transaction amount
        # Helps distinguish micro-smurfing from large-sum laundering
        features['avg_transaction_amount'] = df.groupby('user_id')['amount'].mean()
        
        return features
    
    def engineer_features(self, df):
        """
        Orchestrate full feature engineering pipeline.
        Returns a user-level feature matrix with labels.
        """
        print("\n🔧 Engineering features...")
        
        # Generate all feature categories
        structuring_feats = self._calculate_structuring_features(df)
        network_feats = self._calculate_network_features(df)
        gaming_feats = self._calculate_gaming_features(df)
        identity_feats = self._calculate_identity_features(df)
        volume_feats = self._calculate_volume_features(df)
        
        # Merge all features
        features = pd.concat([
            structuring_feats,
            network_feats,
            gaming_feats,
            identity_feats,
            volume_feats
        ], axis=1)
        
        # Add ground truth labels (for supervised learning)
        labels = df.groupby('user_id')['is_smurf'].first()
        features['is_smurf'] = labels
        
        # Add smurf type for analysis (Optional, good for debug)
        smurf_type = df.groupby('user_id')['smurf_type'].first()
        features['smurf_type'] = smurf_type
        
        # Handle missing values (users with no activity in certain categories)
        features = features.fillna(0)
        
        # --- OUTLIER CLIPPING (Critical for Stability) ---
        # Cap extreme outliers (99th percentile) 
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['is_smurf']]
        
        for col in numeric_cols:
            p99 = features[col].quantile(0.99)
            # Only clip if p99 is non-zero to avoid zeroing out sparse features
            if p99 > 0:
                features[col] = features[col].clip(upper=p99)
        
        print(f"   ✓ Generated {len(numeric_cols)} features for {len(features):,} users")
        
        return features
    
    def validate_features(self, features):
        """
        Compare feature distributions between normal and smurf users.
        This validates that features capture meaningful behavioral differences.
        """
        print("\n" + "="*80)
        print("📊 FEATURE VALIDATION: Normal vs Smurf Comparison")
        print("="*80)
        
        feature_cols = [col for col in features.columns 
                       if col not in ['is_smurf', 'smurf_type']]
        
        comparison = features.groupby('is_smurf')[feature_cols].mean().T
        comparison.columns = ['Normal Users', 'Smurf Users']
        comparison['Ratio (Smurf/Normal)'] = (
            comparison['Smurf Users'] / (comparison['Normal Users'] + self.epsilon)
        )
        
        print(comparison.round(3))
        
        print("\n🎯 Key Behavioral Differences:")
        
        # Highlight most discriminative features
        ratios = comparison['Ratio (Smurf/Normal)'].sort_values(ascending=False)
        
        print("\n   TOP SMURF INDICATORS (Ratio > 2.0):")
        for feat, ratio in ratios[ratios > 2.0].items():
            normal_val = comparison.loc[feat, 'Normal Users']
            smurf_val = comparison.loc[feat, 'Smurf Users']
            print(f"      • {feat:25s}: {smurf_val:8.2f} vs {normal_val:8.2f} ({ratio:.1f}x higher)")
        
        print("\n   TOP NORMAL INDICATORS (Ratio < 0.5):")
        for feat, ratio in ratios[ratios < 0.5].items():
            normal_val = comparison.loc[feat, 'Normal Users']
            smurf_val = comparison.loc[feat, 'Smurf Users']
            print(f"      • {feat:25s}: {smurf_val:8.2f} vs {normal_val:8.2f} ({ratio:.1f}x lower)")
        
        return comparison
    
    def export_features(self, features, output_path):
        """Save engineered features to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        features.to_csv(output_path, index=True)
        print(f"\n✅ Features exported to: {output_path}")
        
        return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🧬 AML Feature Engineering Pipeline")
    print("="*80)
    
    # Path handling to ensure it works from any directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "../data/raw/transactions.csv")
    output_path = os.path.join(base_dir, "../data/processed/features.csv")
    
    # Initialize engineer
    engineer = FeatureEngineer(epsilon=1e-6)
    
    # Load raw transactions
    df = engineer.load_data(input_path)
    
    # Engineer features
    features = engineer.engineer_features(df)
    
    # Validate feature quality
    comparison = engineer.validate_features(features)
    
    # Export
    engineer.export_features(features, output_path)
    
    print("\n" + "="*80)
    print("🎯 Feature Engineering Complete!")
    print("="*80)
    