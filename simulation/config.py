"""
Simulation Configuration (The Commander's Orders).
Defines sophisticated money laundering topologies and normal user behaviors for i-Betting.
"""

# --- General Simulation Settings ---
SIMULATION_DAYS = 60  # Increased to 60 to see monthly patterns
TOTAL_USERS = 5000    # Higher volume for better anomaly contrast
SMURF_PERCENTAGE = 0.03 # 3% Criminals (150 Smurfs)

# --- GLOBAL BEHAVIORAL PARAMETERS ---
# Real users are consistent with devices/IPs. Criminals rotate them.
DEVICE_ROTATION_PROBABILITY = {
    "normal": 0.05,  # Normal users rarely switch devices
    "smurf": 0.80    # Smurfs switch devices constantly to hide
}

# Real users sleep. Smurfs (bots) operate 24/7 or at odd hours.
TIME_OF_DAY_WEIGHTS = {
    "normal": [0.01]*6 + [0.05]*12 + [0.15]*6, # Active 6 PM - 12 AM
    "smurf": [0.1]*24  # Uniform activity (Bot-like) or Night-shift
}

# ------------------------------------------------------
# STRATEGY 1: The "Structuring" Smurf (Placement)
# Logic: Deposits just below $10k to avoid CTR reporting.
# ------------------------------------------------------
STRUCTURING_CONFIG = {
    "min_amount": 9000,
    "max_amount": 9950, # Tight range near threshold
    "frequency_hours": [24, 48, 72], # Regular intervals
    "variance": 0.02, # Robotic precision (e.g., always $9800 +/- $20)
    "is_round_number": False # Smurfs avoid round numbers to look "organic" (e.g., 9850.50)
}

# ------------------------------------------------------
# STRATEGY 2: The "Fan-In" Smurf (Gathering / Mule)
# Logic: Many small deposits from "mules" converging to one account.
# ------------------------------------------------------
FAN_IN_CONFIG = {
    "mule_count": 10, # 10 different accounts feeding one
    "min_amount": 200,
    "max_amount": 800,
    "deposits_per_day": 8, # High velocity
    "bet_percentage": 0.05 # Minimal betting (just washing)
}

# ------------------------------------------------------
# STRATEGY 3: The "Fan-Out" Smurf (Layering)
# Logic: One large deposit ($50k) broken into small withdrawals to many accounts.
# ------------------------------------------------------
FAN_OUT_CONFIG = {
    "initial_deposit": 50000,
    "split_count": 25, # Breaks into 25 small transfers
    "transfer_delay_minutes": 15, # Fast sequential movement
    "destination_variability": "high" # Sends to many different external IBANs
}

# ------------------------------------------------------
# STRATEGY 4: The "Chip Dumping" (i-Betting Specific)
# Logic: Two accounts sit at same poker table. A loses to B on purpose.
# ------------------------------------------------------
CHIP_DUMPING_CONFIG = {
    "win_loss_ratio": 0.05, # The "Loser" account wins only 5% of hands (Statistical anomaly)
    "avg_loss_amount": 1000, # Large consistent losses
    "opponent_consistency": 0.95 # 95% of losses are to the SAME opponent ID
}

# ------------------------------------------------------
# NORMAL USER BEHAVIOR (The Baseline)
# ------------------------------------------------------
NORMAL_USER_CONFIG = {
    "deposit_frequency_days": 7,  # Payday pattern
    "deposit_min": 20,
    "deposit_max": 300,
    "bet_percentage": 0.90,       # High churn (Real gamblers play)
    "win_loss_ratio": 0.48,       # The "House Edge" (Normal users lose slightly more than win)
    "cashout_threshold": 3.0      # Greed factor (Only withdraws on big wins)
}