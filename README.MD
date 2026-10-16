# 💳 FinTech Fraud Guard & Risk Engine

An end-to-end Machine Learning system and interactive risk operations console for real-time banking fraud detection, trained on relational banking telemetry from **CaixaBank Tech**.

---

dataset from 
https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

## 📌 Architecture Overview

Unlike toy single-table datasets, production banking systems operate on relational schemas. This system unifies four distinct relational entities:

```
[Users Table] (2k customers)
       │ (client_id)
       ├───► [Cards Table] (6.1k cards)
       │            │ (card_id)
       ▼            ▼
[Transactions Stream] (8.9M records) ◄─── [MCC Codes] (109 categories)
       │
       ▼
[Fraud Target Labels] ("Yes" / "No")
```

---

## 🏛️ Two-Tier Cascaded Risk Architecture

Real-world payment processors (e.g., Stripe Radar, Visa, Adyen) must return decisions within strict latency windows (<50ms). This system uses a cascaded architecture:

1. **Tier 1: Fast Gatekeeper (Screening Filter)**
   * **Scope:** Evaluates single-transaction telemetry and instant card limits in sub-10ms.
   * **Outcomes:**
     * *Low Risk ($P < 0.10$):* Auto-Approve without customer friction (>95% of volume).
     * *High/Ambiguous Risk ($P \ge 0.10$):* Escalates immediately to Tier 2 Deep Analysis.
     * *Critical Breach:* Instant block (credit limit explosion, card dark-web alerts).

2. **Tier 2: Deep Behavioral Engine (Contextual Profiling)**
   * **Scope:** Evaluates the transaction against the customer's personal historical baseline.
   * **Features:** Geographic travel feasibility (home coordinates vs merchant zip), merchant category novelty (has customer used this MCC before?), personal spending Z-score, and cadence velocity.
   * **Outcomes:** Final Decision (Approve, Step-Up 3D-Secure SMS Challenge, or Freeze Card).


---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.10+
* **Core ML & Data**: `scikit-learn`, `pandas`, `numpy`
* **Model Serialization**: `joblib`
* **Interactive UI**: `streamlit`

---

## 🚀 Quickstart

1. **Activate the Virtual Environment:**
   ```bash
   source frauddetector/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Export Serialized Models & Simulation Pool:**
   ```bash
   python export_models.py
   python build_simulation_pool.py
   ```

4. **Launch the Real-Time Risk Operations Console:**
   ```bash
   streamlit run app.py
   ```

---

## 📖 Engineering Journal
Detailed mathematical formulations, benchmark results, and architectural records are documented step-by-step in [JOURNAL.md](JOURNAL.md).
