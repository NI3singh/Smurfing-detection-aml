import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def check_benfords_law(file_path):
    print(f"📊 Reading {file_path}...")
    df = pd.read_csv(file_path)

    # Filter for DEPOSITS only (where structuring happens)
    deposits = df[df['type'] == 'DEPOSIT'].copy()
    
    # Extract leading digit (convert amount to string, take 1st char)
    # We ignore amounts < 10 to be safe
    deposits = deposits[deposits['amount'] >= 10]
    deposits['leading_digit'] = deposits['amount'].astype(str).str[0].astype(int)

    # 1. Normal Users Distribution
    normal_dist = deposits[deposits['is_smurf'] == False]['leading_digit'].value_counts(normalize=True).sort_index()
    
    # 2. Smurfs Distribution
    smurf_dist = deposits[deposits['is_smurf'] == True]['leading_digit'].value_counts(normalize=True).sort_index()
    
    # 3. Theoretical Benford's Law
    digits = np.arange(1, 10)
    benford = np.log10(1 + 1/digits)

    # --- Print Stats ---
    print("\n🧐 BENFORD'S LAW ANALYSIS (Deposits):")
    print(f"{'Digit':<5} | {'Benford':<10} | {'Normal Users':<15} | {'Smurfs':<15}")
    print("-" * 55)
    for d in digits:
        ben = benford[d-1]
        norm = normal_dist.get(d, 0)
        smurf = smurf_dist.get(d, 0)
        print(f"{d:<5} | {ben:.1%}{'':<5} | {norm:.1%}{'':<10} | {smurf:.1%}")

    print("\n✅ INTERPRETATION:")
    print(" - Normal Users should be close to 'Benford'.")
    print(" - Smurfs should deviate heavily (look for spikes at 9 or 5).")

if __name__ == "__main__":
    # Correct path relative to where you run it
    path = "data/raw/transactions.csv"
    if not os.path.exists(path):
        # Fallback if running from inside simulation folder
        path = "../data/raw/transactions.csv"
        
    check_benfords_law(path)