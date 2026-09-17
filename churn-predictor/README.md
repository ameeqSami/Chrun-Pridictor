# Random Forest Churn Predictor

This folder contains the **Random Forest pipeline** for predicting customer churn on the Telco dataset. It is the upgraded successor to the original Decision Tree pipeline — retaining full interpretability tooling while delivering meaningfully better metrics.

## What's in this folder?

- **`chrunPipe.ipynb`**: The main notebook where the pipeline is built, trained, evaluated, and visualized.
- **`pipeline_dt.pkl`**: The saved scikit-learn pipeline (contains the custom transformers and trained Random Forest model).

> **Note on helper files:** The notebook imports custom modules from `../dataset and other libs/`:
> - `cleaningcls.py`: Custom transformer (`clean_cls`) for data cleaning, type conversions, and one-hot encoding.
> - `ctf.py`: Custom transformer (`CorrelationThresholdFilter`) for feature selection.
> - `pridict_thresh.py`: Custom classifier wrapper (`pridict_thresh`) that allows tuning the decision threshold independently of the model.
> - `trace_path.py`: Script used to trace and print the exact decision path a customer takes through an individual tree in the forest.
> - `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The raw Telco dataset.

---

## How the Pipeline Works

Everything is chained into a single `Pipeline` so raw data goes in and predictions come out:

### 1. Data Cleaning (`clean_cls`)
- Drops `customerID`
- Converts `TotalCharges` (stored as string) to numeric, filling blanks with `0`
- Maps binary Yes/No flags → `1`/`0`
- Maps `gender` (Male/Female) → `1`/`0`
- One-hot encodes multi-category columns discovered at fit time

### 2. Feature Selection (`CorrelationThresholdFilter`)
Keeps features whose absolute Pearson correlation with `Churn` exceeds the threshold. The threshold was lowered from `0.23` (DT) to **`0.10`** to expose more signal to the forest:

| Feature | Correlation Direction |
|---|---|
| `Contract_Two year` | Negative (long contracts = lower churn) |
| `tenure` | Negative (longer tenure = lower churn) |
| `InternetService_Fiber optic` | Positive (higher churn rate) |
| `PaymentMethod_Electronic check` | Positive (higher churn rate) |
| `MonthlyCharges` | Positive |
| `Contract_One year` | Negative |
| *(+ additional features above 0.10 threshold)* | |

### 3. Threshold Wrapper (`pridict_thresh`)
A lightweight scikit-learn–compatible wrapper around the Random Forest that allows the **decision threshold to be tuned without retraining**. Instead of defaulting to `0.5`, the threshold is calibrated post-hoc on the validation set to find the best precision/recall trade-off for the churn class.

### 4. Random Forest Classifier
Upgraded from a single Decision Tree to an ensemble:

| Hyperparameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | Enough trees for stable probability estimates |
| `max_depth` | 8 | Deeper than DT but capped to avoid overfitting |
| `class_weight` | `'balanced'` | Compensates for the ~26%/74% churn imbalance |
| `criterion` | `'entropy'` | Consistent with the original DT setup |

---

## Performance & Upgrade Summary

Evaluated on a 20% stratified test split:

| Metric | Decision Tree (v1) | Random Forest (v2) | Δ |
|---|---|---|---|
| ROC-AUC | ~0.824 | **~0.843** | +0.019 |
| F1 (Churn class) | ~0.60 | **~0.63** | +0.03 |
| Recall (Churn) | ~75% | ~75–78% | ≈ |
| Overall Accuracy | ~75% | ~76–78% | ↑ |

The combination of more features (lower CTF threshold), the ensemble's variance reduction, and threshold tuning together drive the improvement.

### What the forest focuses on (Feature Importances):

1. **Tenure (~30%+):** The single strongest predictor — the longer a customer stays, the less likely they are to leave.
2. **Monthly Charges:** Higher bills increase churn risk, especially without a long-term contract.
3. **Two-Year Contract:** Customers locked into multi-year agreements almost never churn.
4. **Fiber Optic Internet:** Consistently associated with higher churn — likely reflecting competitive pricing pressure.
5. **Electronic Check Payment:** Manual monthly payers churn more than those on automatic billing.

---

## Tracing Individual Customer Decisions

Even though the model is a forest of 200 trees, `trace_path.py` lets you inspect the decision path of **any single tree** for any customer:

```python
import sys
import os
sys.path.append(os.path.abspath("../dataset and other libs"))

from trace_path import trace_customer_path

# Preprocess through the first two pipeline steps
X_transformed = pipeline_dt['clean'].transform(X_train.head(1))
X_filtered    = pipeline_dt['ctf'].transform(X_transformed)

# Trace through tree #0 of the forest (default)
trace_customer_path(pipeline_dt, X_filtered, customer_idx=0, tree_index=0)
```

This prints every split node visited, the feature and threshold checked, whether the customer went left or right, and the final leaf's class distribution and churn probability.

---

## Threshold Tuning

After training, the optimal prediction threshold is found by sweeping values and maximising F1 on the churn class:

```python
from pridict_thresh import pridict_thresh

# Wrap the trained RF
wrapped = pridict_thresh(model=rf_clf, thresh=0.38)  # example tuned threshold

# Predictions now use 0.38 instead of 0.50
preds = wrapped.predict(X_test_filtered)
```

The threshold is stored inside the serialised pipeline so inference is always consistent.

---

## Running Inference with the Saved Model

Load `pipeline_dt.pkl` and pass **raw, uncleaned data** directly — the pipeline handles everything:

```python
import pickle
import sys
import os
import pandas as pd

# Make custom transformer classes importable
sys.path.append(os.path.abspath("../dataset and other libs"))

# Load the full pipeline
with open("pipeline_dt.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load raw data
raw_data = pd.read_csv("../dataset and other libs/WA_Fn-UseC_-Telco-Customer-Churn.csv")
sample = raw_data.drop(columns=["Churn"]).head(5)

# Predict
preds = pipeline.predict(sample)
probs = pipeline.predict_proba(sample)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    result = "Will Churn" if pred == 1 else "Will Stay"
    print(f"Customer {i + 1}: {result} ({prob:.1%} churn risk)")
```

---

## Repository Layout

```
Chrun-Pridictor/
├── churn-predictor/              ← This folder (Random Forest pipeline)
│   ├── chrunPipe.ipynb
│   └── pipeline_dt.pkl
├── churn-predictor-lr/           ← Logistic Regression baseline pipeline
│   └── chrunPipeLR.ipynb
├── churn-prec/                   ← Early exploratory notebook
│   └── chrun.ipynb
└── dataset and other libs/       ← Shared helpers & raw data
    ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
    ├── cleaningcls.py
    ├── ctf.py
    ├── pridict_thresh.py
    └── trace_path.py
```
