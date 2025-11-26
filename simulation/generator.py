"""
Transaction Generator Engine for i-Betting AML Simulation.
Produces high-fidelity synthetic data with embedded money laundering patterns.
"""

import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from faker import Faker
import os
from config import *

fake = Faker()
Faker.seed(42)
np.random.seed(42)


class TransactionGenerator:
    """
    Orchestrates the generation of synthetic i-Betting transactions.
    Embeds 4 distinct smurfing patterns among normal user behaviors.
    """
    
    def __init__(self, start_date=None):
        self.start_date = start_date or datetime(2024, 1, 1)
        self.transactions = []
        self.device_pool = {}  # user_id -> list of device_ids
        self.ip_pool = {}      # user_id -> list of ip_addresses
        
    def _generate_timestamp(self, day_offset, user_type="normal"):
        """
        Generate timestamp based on time-of-day behavioral patterns.
        Smurfs operate 24/7 (bot-like), Normal users follow evening patterns.
        """
        weights = TIME_OF_DAY_WEIGHTS[user_type]
        hour = np.random.choice(24, p=np.array(weights) / sum(weights))
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        return self.start_date + timedelta(
            days=day_offset,
            hours=hour,
            minutes=minute,
            seconds=second
        )
    
    def _get_device_id(self, user_id, user_type="normal"):
        """
        Device fingerprinting logic with rotation based on user type.
        Normal users: Stick to 1-2 devices (phone + laptop).
        Smurfs: Constantly rotate devices to avoid tracking.
        """
        if user_id not in self.device_pool:
            # Initialize with 1-2 devices for normal users
            num_devices = 1 if user_type == "smurf" else np.random.choice([1, 2], p=[0.7, 0.3])
            self.device_pool[user_id] = [f"DEVICE_{uuid.uuid4().hex[:8]}" for _ in range(num_devices)]
        
        # Rotation logic
        rotation_prob = DEVICE_ROTATION_PROBABILITY[user_type]
        if np.random.random() < rotation_prob:
            # Generate new device
            new_device = f"DEVICE_{uuid.uuid4().hex[:8]}"
            self.device_pool[user_id].append(new_device)
            return new_device
        else:
            # Reuse existing device
            return np.random.choice(self.device_pool[user_id])
    
    def _get_ip_address(self, user_id, user_type="normal"):
        """
        IP assignment. Normal users: stable IP. Smurfs: rotate frequently.
        """
        if user_id not in self.ip_pool:
            self.ip_pool[user_id] = [fake.ipv4()]
        
        rotation_prob = DEVICE_ROTATION_PROBABILITY[user_type]  # Same logic as devices
        if np.random.random() < rotation_prob:
            new_ip = fake.ipv4()
            self.ip_pool[user_id].append(new_ip)
            return new_ip
        else:
            return np.random.choice(self.ip_pool[user_id])
    
    def _add_transaction(self, user_id, timestamp, amount, tx_type, 
                         is_smurf=False, smurf_type="none", related_user=None):
        """
        Core transaction recording method.
        """
        user_type = "smurf" if is_smurf else "normal"
        
        self.transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id,
            "timestamp": timestamp,
            "amount": round(amount, 2),
            "type": tx_type,
            "device_id": self._get_device_id(user_id, user_type),
            "ip_address": self._get_ip_address(user_id, user_type),
            "is_smurf": is_smurf,
            "smurf_type": smurf_type,
            "related_user_id": related_user  # For chip dumping / fan patterns
        })
    
    # ========================================================================
    # NORMAL USER GENERATION
    # ========================================================================
    
    def generate_normal_user(self, user_id):
        """
        Simulate a legitimate gambler over 60 days.
        Pattern: Weekly deposits (payday), high betting activity, occasional cashouts.
        """
        config = NORMAL_USER_CONFIG
        balance = 0
        
        for day in range(SIMULATION_DAYS):
            # Weekly deposit pattern (every ~7 days with variance)
            if day % config["deposit_frequency_days"] == 0 or (day > 0 and np.random.random() < 0.1):
                deposit_amount = np.random.uniform(config["deposit_min"], config["deposit_max"])
                timestamp = self._generate_timestamp(day, "normal")
                self._add_transaction(user_id, timestamp, deposit_amount, "DEPOSIT")
                balance += deposit_amount
            
            # Betting behavior (if sufficient balance)
            if balance > 10 and np.random.random() < config["bet_percentage"]:
                bet_amount = min(balance * np.random.uniform(0.1, 0.3), balance)
                timestamp = self._generate_timestamp(day, "normal")
                
                # Win or lose based on house edge
                if np.random.random() < config["win_loss_ratio"]:
                    win_amount = bet_amount * np.random.uniform(1.5, 3.0)
                    self._add_transaction(user_id, timestamp, win_amount, "BET_WIN")
                    balance += win_amount
                else:
                    self._add_transaction(user_id, timestamp, bet_amount, "BET_LOSS")
                    balance -= bet_amount
            
            # Cashout behavior (when big win occurs)
            if balance > config["deposit_max"] * config["cashout_threshold"]:
                withdrawal = balance * np.random.uniform(0.6, 0.9)
                timestamp = self._generate_timestamp(day, "normal")
                self._add_transaction(user_id, timestamp, withdrawal, "WITHDRAWAL")
                balance -= withdrawal
    
    # ========================================================================
    # SMURF PATTERN 1: STRUCTURING (Placement Phase)
    # ========================================================================
    
    def generate_structuring_smurf(self, user_id):
        """
        RED FLAG: Deposits just below $10k threshold at regular intervals.
        Classic structuring to avoid Currency Transaction Reports (CTR).
        """
        config = STRUCTURING_CONFIG
        
        for day_offset in config["frequency_hours"]:
            # Convert hours to days
            day = day_offset // 24
            if day >= SIMULATION_DAYS:
                break
            
            # Generate suspiciously precise amount near $10k
            base_amount = np.random.uniform(config["min_amount"], config["max_amount"])
            
            # Apply minimal variance (robotic precision)
            amount = base_amount * (1 + np.random.uniform(-config["variance"], config["variance"]))
            
            # Make it look "organic" with non-round cents
            if not config["is_round_number"]:
                amount = amount + np.random.uniform(0.01, 0.99)
            
            timestamp = self._generate_timestamp(day, "smurf")
            self._add_transaction(user_id, timestamp, amount, "DEPOSIT", 
                                is_smurf=True, smurf_type="structuring")
            
            # Minimal betting to "legitimize" (but low churn is suspicious)
            small_bet = amount * 0.05
            bet_timestamp = timestamp + timedelta(minutes=np.random.randint(5, 30))
            self._add_transaction(user_id, bet_timestamp, small_bet, "BET_LOSS",
                                is_smurf=True, smurf_type="structuring")
    
    # ========================================================================
    # SMURF PATTERN 2: FAN-IN (Gathering / Mule Accounts)
    # ========================================================================
    
    def generate_fan_in_smurf(self, user_id, mule_base_id):
        """
        RED FLAG: Many small deposits from different "mule" accounts converging to one.
        Indicates money mule network funneling funds.
        """
        config = FAN_IN_CONFIG
        
        for day in range(SIMULATION_DAYS):
            if np.random.random() < 0.3:  # Active ~30% of days
                for _ in range(config["deposits_per_day"]):
                    # Simulate deposits from different mule accounts
                    mule_id = f"{mule_base_id}_MULE_{np.random.randint(1, config['mule_count']+1)}"
                    amount = np.random.uniform(config["min_amount"], config["max_amount"])
                    timestamp = self._generate_timestamp(day, "smurf")
                    
                    self._add_transaction(user_id, timestamp, amount, "TRANSFER",
                                        is_smurf=True, smurf_type="fan_in",
                                        related_user=mule_id)
                    
                    # Minimal betting (just washing)
                    if np.random.random() < config["bet_percentage"]:
                        bet_amount = amount * 0.1
                        bet_time = timestamp + timedelta(minutes=np.random.randint(1, 10))
                        self._add_transaction(user_id, bet_time, bet_amount, "BET_LOSS",
                                            is_smurf=True, smurf_type="fan_in")
    
    # ========================================================================
    # SMURF PATTERN 3: FAN-OUT (Layering / Distribution)
    # ========================================================================
    
    def generate_fan_out_smurf(self, user_id):
        """
        RED FLAG: One large deposit immediately split into many small withdrawals.
        Classic layering technique to obfuscate fund origin.
        """
        config = FAN_OUT_CONFIG
        
        # Initial large deposit (day 5 to establish pattern)
        deposit_day = 5
        timestamp = self._generate_timestamp(deposit_day, "smurf")
        self._add_transaction(user_id, timestamp, config["initial_deposit"], "DEPOSIT",
                            is_smurf=True, smurf_type="fan_out")
        
        # Rapid sequential splitting
        split_amount = config["initial_deposit"] / config["split_count"]
        current_time = timestamp + timedelta(minutes=config["transfer_delay_minutes"])
        
        for i in range(config["split_count"]):
            # Vary amounts slightly to look "less mechanical"
            amount = split_amount * np.random.uniform(0.9, 1.1)
            
            # Send to different external accounts
            dest_account = f"EXTERNAL_IBAN_{uuid.uuid4().hex[:8]}"
            
            self._add_transaction(user_id, current_time, amount, "WITHDRAWAL",
                                is_smurf=True, smurf_type="fan_out",
                                related_user=dest_account)
            
            current_time += timedelta(minutes=config["transfer_delay_minutes"])
    
    # ========================================================================
    # SMURF PATTERN 4: CHIP DUMPING (i-Betting Specific)
    # ========================================================================
    
    def generate_chip_dumper(self, user_id, accomplice_id):
        """
        RED FLAG: Two players at same poker table. One intentionally loses to the other.
        Statistical anomaly: Loses 95% of hands to the SAME opponent.
        """
        config = CHIP_DUMPING_CONFIG
        
        # Initial deposit for both accounts
        for uid in [user_id, accomplice_id]:
            deposit = np.random.uniform(2000, 5000)
            timestamp = self._generate_timestamp(2, "smurf")
            self._add_transaction(uid, timestamp, deposit, "DEPOSIT",
                                is_smurf=True, smurf_type="chip_dumping")
        
        # Play sessions over 30 days
        for day in range(5, min(35, SIMULATION_DAYS)):
            if np.random.random() < 0.4:  # Session ~40% of days
                session_time = self._generate_timestamp(day, "smurf")
                
                # Generate 10 hands per session
                for hand in range(10):
                    hand_time = session_time + timedelta(minutes=hand*5)
                    loss_amount = config["avg_loss_amount"] * np.random.uniform(0.8, 1.2)
                    
                    # Loser loses to accomplice 95% of the time
                    if np.random.random() < config["opponent_consistency"]:
                        # Loser loses
                        self._add_transaction(user_id, hand_time, loss_amount, "BET_LOSS",
                                            is_smurf=True, smurf_type="chip_dumping",
                                            related_user=accomplice_id)
                        
                        # Accomplice wins
                        self._add_transaction(accomplice_id, hand_time, loss_amount, "BET_WIN",
                                            is_smurf=True, smurf_type="chip_dumping",
                                            related_user=user_id)
                    else:
                        # Occasional win to avoid perfect 0% (too obvious)
                        self._add_transaction(user_id, hand_time, loss_amount*0.5, "BET_WIN",
                                            is_smurf=True, smurf_type="chip_dumping")
        
        # Final cashout by accomplice (integration phase)
        cashout_day = min(40, SIMULATION_DAYS - 1)
        cashout_time = self._generate_timestamp(cashout_day, "smurf")
        self._add_transaction(accomplice_id, cashout_time, 
                            config["avg_loss_amount"] * 50, "WITHDRAWAL",
                            is_smurf=True, smurf_type="chip_dumping")
    
    # ========================================================================
    # ORCHESTRATION
    # ========================================================================
    
    def generate_all_users(self, total_users, smurf_percentage):
        """
        Generate complete dataset with mixed normal and criminal users.
        """
        num_smurfs = int(total_users * smurf_percentage)
        num_normal = total_users - num_smurfs
        
        print(f"🎲 Generating {total_users} users ({num_normal} normal, {num_smurfs} smurfs)...")
        
        # Generate normal users
        for i in range(num_normal):
            user_id = f"USER_{i:05d}"
            self.generate_normal_user(user_id)
            
            if (i + 1) % 500 == 0:
                print(f"   ✓ Normal users: {i+1}/{num_normal}")
        
        # Generate smurfs with different strategies (distribute evenly)
        smurf_strategies = [
            ("structuring", self.generate_structuring_smurf),
            ("fan_in", lambda uid: self.generate_fan_in_smurf(uid, f"MULE_NETWORK_{uid}")),
            ("fan_out", self.generate_fan_out_smurf),
        ]
        
        smurf_id = num_normal
        for strategy_idx in range(num_smurfs):
            strategy_name, strategy_func = smurf_strategies[strategy_idx % len(smurf_strategies)]
            user_id = f"SMURF_{smurf_id:05d}"
            
            strategy_func(user_id)
            smurf_id += 1
        
        # Generate chip dumping pairs (requires 2 accounts)
        num_chip_dumping_pairs = num_smurfs // 10  # 10% of smurfs
        for pair_idx in range(num_chip_dumping_pairs):
            loser_id = f"CHIPDUMP_LOSER_{pair_idx:03d}"
            winner_id = f"CHIPDUMP_WINNER_{pair_idx:03d}"
            self.generate_chip_dumper(loser_id, winner_id)
        
        print(f"   ✓ Smurfs: {num_smurfs} (+ {num_chip_dumping_pairs} chip-dump pairs)")
    
    def export_to_csv(self, output_path=r"C:\Users\ELaunch\OneDrive\Desktop\AML_ENGINE\aml-detection-engine\modules\Smurfing_detection\data\raw\transactions.csv"):
        """
        Convert transaction list to DataFrame and export.
        """
        print(f"\n📊 Converting to DataFrame...")
        df = pd.DataFrame(self.transactions)
        
        # Sort by timestamp (chronological order)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export
        df.to_csv(output_path, index=False)
        
        print(f"✅ Dataset exported: {output_path}")
        print(f"   📈 Total transactions: {len(df):,}")
        print(f"   🚩 Smurf transactions: {df['is_smurf'].sum():,} ({df['is_smurf'].mean()*100:.1f}%)")
        print(f"   📅 Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🎰 i-Betting AML Transaction Generator")
    print("="*70)
    
    generator = TransactionGenerator(start_date=datetime(2024, 1, 1))
    generator.generate_all_users(TOTAL_USERS, SMURF_PERCENTAGE)
    
    df = generator.export_to_csv()
    
    print("\n" + "="*70)
    print("📋 Dataset Summary:")
    print("="*70)
    print(df.groupby("smurf_type").agg({
        "transaction_id": "count",
        "amount": ["mean", "sum"]
    }).round(2))
    
    print("\n🎯 Ready for ML model training!")