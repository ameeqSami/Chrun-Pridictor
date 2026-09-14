# Telco Churn Intelligence & Dynamic Retention Offer Engine

A full-stack machine learning web application serving a trained Decision Tree pipeline for Telco Customer Churn Prediction, explainable decision tree path tracing, and automated targeted retention offer generation.

---

## Key Features

1. **Test Cohort Analysis (1,409 customers)**
   - 20% stratified holdout population matching model training.
   - Interactive data table with multi-criteria filtering (Churn Risk, Contract type, Internet service, customer search).
   - Real-time KPI dashboard (Total population, Churn Predicted, Retained Predicted, Analyzed profiles, Offers Dispatched).

2. **Explainable Decision Tree Path Tracing**
   - Direct integration with `get_customer_path_df` from `trace_path.py`.
   - Visual step-by-step traversal stepper displaying evaluated features, split thresholds, actual customer values, branch decisions (left/right condition met), and node classification distributions.

3. **Dynamic Retention Offer Engine (`offer_engine.py`)**
   - Automatically maps specific failing features and path nodes to targeted retention strategies:
     - **Fiber Optic + Early Tenure:** Fiber Premier Retention & Care Suite (25% off + VIP router + tech support).
     - **Fiber Optic + High Tenure:** VIP Fiber Loyalty Appreciation Bundle (20% credit + streaming pass).
     - **Month-to-Month Contract:** Annual Commitment Freedom Plan ($20/mo savings + price lock).
     - **Electronic Check:** AutoPay & Paperless Seamless Rewards ($15/mo discount).
     - **Early Onboarding (<6 mos):** Early Journey Success & Concierge Pack.
   - Calculates estimated annual retained revenue.

4. **Workflow Execution & Mock Dispatch**
   - One-click "Dispatch Retention Offer" with automated channel simulation (Email + SMS).
   - "Batch Analyze All" bulk operation with instantaneous KPI recomputation.

---

## Directory Structure

```
churn-app/
├── app.py                     # Flask server & REST API endpoints
├── offer_engine.py            # Targeted retention offer rule engine
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore
├── models/
│   └── pipeline_dt.pkl        # Trained Decision Tree Pipeline (clean -> ctf -> model)
├── libs/
│   ├── trace_path.py          # Custom decision path extraction function
│   ├── cleaningcls.py         # Custom data cleaning transformer
│   └── ctf.py                 # Custom CorrelationThresholdFilter transformer
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Telco Customer Dataset
├── templates/
│   └── index.html             # UI template with glassmorphism layout
└── static/
    ├── css/
    │   └── styles.css         # Modern Dark Slate & Glassmorphism Design System
    └── js/
        └── app.js             # Client controller, API consumer, and UI rendering
```

---

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.
