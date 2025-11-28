# 🕵️ Smurf Hunter: AI-Driven AML Anomaly Detection

**An advanced Unsupervised Machine Learning engine designed to detect financial structuring ("Smurfing") and complex money laundering topologies in i-Betting platforms.**

---

## 📖 What is Smurfing?

In Anti-Money Laundering (AML), **Smurfing** (or Structuring) is the act of breaking up a large sum of illicit money into smaller, less suspicious transactions to evade regulatory reporting thresholds.

This engine detects four sophisticated typologies of financial crime:

1. **Structuring (Placement):** Repeatedly depositing amounts just below the reporting threshold (e.g., $9,000 - $9,900) to avoid triggering automatic alerts.
2. **Fan-In (Gathering):** A "Money Mule" network where many small accounts transfer funds into a single central account.
3. **Fan-Out (Layering):** A single account receives a large deposit and immediately disperses it via small transfers to many different external accounts.
4. **Chip Dumping (Integration):** A form of collusion in peer-to-peer games (like Poker) where one player intentionally loses chips to an accomplice to transfer value.

---

## ⚙️ How It Works (The Pipeline)

This project moves beyond simple "Rule-Based" checks (e.g., `if amount > 10k`). Instead, it uses **Behavioral Biometrics** and **Anomaly Detection** to identify the *intent* behind transactions.

### Phase 1: High-Fidelity Data Simulation

We utilize **Agent-Based Modeling (ABM)** to simulate a realistic i-Betting economy.

* **Normal Agents:** Modeled with organic behaviors (sleeping at night, variance in deposit amounts, high betting churn).
* **Smurf Agents:** Modeled with specific criminal strategies (robotic precision, 24/7 activity, low betting ratios).
* *Result:* A synthetic dataset containing hundreds of thousands of transactions with a known ground truth.

### Phase 2: Advanced Feature Engineering

Raw transaction logs are transformed into **12-Dimensional Behavioral Profiles**. We do not just look at amounts; we look at:

* **Structuring Count:** Frequency of near-threshold deposits.
* **Wager Ratio:** The ratio of money bet vs. money deposited (The "Laundromat" check).
* **Flow Ratio:** The balance of incoming vs. outgoing funds.
* **Night Owl Ratio:** Activity during unusual hours (00:00 - 05:00).
* **Device Rotation:** Analysis of device fingerprint stability.

### Phase 3: Unsupervised Learning (The Brain)

We employ an **Isolation Forest** algorithm.

* **Training Strategy:** The model is trained purely on "Normal" user behavior. It learns what a legitimate player looks like.
* **Inference:** When it encounters a Smurf, it flags them as an anomaly not because it memorized a rule, but because the user's mathematical distance from "Normal" is too high.

### Phase 4: Explainable AI (XAI)

A "Black Box" model is useless for compliance. We integrate **SHAP (SHapley Additive exPlanations)** to provide the *reasoning* behind every block.

* *Output:* "User blocked. Risk Score: 95%. Primary Reason: `wager_ratio` is suspiciously low (0.05)."

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **API Framework:** FastAPI
* **Machine Learning:** Scikit-Learn (Isolation Forest)
* **Explainability:** SHAP
* **Data Processing:** Pandas, NumPy
* **Simulation:** Faker

---

## 📊 Dataset Statistics

The current model is validated against a dataset with the following distribution:

* **Normal Users:** ~96.5%
* **Smurfs:** ~3.5%
* **Detection Rate (Recall):** > 98% on test set.