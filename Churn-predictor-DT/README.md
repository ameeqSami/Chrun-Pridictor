# Decision Tree Churn Predictor

This folder contains the Decision Tree pipeline for predicting customer churn on the Telco dataset. The main focus here was building an end-to-end scikit-learn pipeline that is easy to interpret, fast to run, and capable of explaining *why* a customer is flagged as likely to leave.

## What's in this folder?

- **`chrunPipeDT.ipynb`**: The main notebook where the pipeline is built, trained, evaluated, and visualized.
- **`pipeline_dt.pkl`**: The saved scikit-learn pipeline containing the custom transformers and trained decision tree.

> **Note on helper files:** The notebook imports custom modules from `../dataset and other libs/`:
> - `cleaningcls.py`: Custom transformer (`clean_cls`) for data cleaning, type conversions, and one-hot encoding.
> - `ctf.py`: Custom transformer (`CorrelationThresholdFilter`) for feature selection.
> - `trace_path.py`: Script used to trace and print the exact path a customer takes through the tree.
> - `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The raw dataset.

---

## How the Pipeline Works

Instead of running data prep and modeling as disconnected steps, everything is chained together in a single `Pipeline`:

1. **Data Cleaning (`clean_cls`)**
   Drops `customerID`, converts numeric fields stored as strings (like `TotalCharges`), maps binary flags (`Yes`/`No`, `Male`/`Female`) to `1`/`0`, and creates one-hot dummy variables for multi-category columns.

2. **Feature Selection (`CorrelationThresholdFilter`)**
   Filters out low-signal columns by measuring Pearson correlation with `Churn`. At a threshold of `0.23`, it keeps the four strongest predictors:
   - `Contract_Two year`
   - `tenure`
   - `InternetService_Fiber optic`
   - `PaymentMethod_Electronic check`

3. **Decision Tree Classifier**
   - **`max_depth=4`**: Kept shallow intentionally to avoid overfitting and make the decision rules easy to inspect and explain.
   - **`class_weight='balanced'`**: Churn is imbalanced (~26% churned vs ~74% retained). Balanced weighting penalizes missed churners more heavily, pushing the model to catch at-risk customers rather than defaulting to predicting "no churn".
   - **`criterion='entropy'`**

---

## Performance & Key Takeaways

Evaluated on a 20% stratified test split:

- **ROC-AUC:** ~0.82
- **Recall (Churn):** ~75%
- **Overall Accuracy:** ~75%

Because of the balanced weighting, the tree captures around 3 out of every 4 churners (75% recall). In churn prevention, flagging potential churners early is usually worth the trade-off of a few false alarms.

### What the tree focused on:

1. **Two-Year Contracts (~45% importance):** Customers with multi-year commitments rarely leave.
2. **Tenure (~26% importance):** Newer customers are at much higher risk than long-standing ones.
3. **Fiber Optic Service (~22% importance):** Higher churn rates here, often pointing to competitive pressure, pricing, or service expectations.
4. **Electronic Check (~6% importance):** Customers paying monthly by manual electronic checks churn more often than those enrolled in automatic credit card payments.

---

## Tracing Individual Customer Decisions

Because the tree has a max depth of 4, we don't have to treat it like a black box. With `trace_path.py`, you can follow any customer down the tree to see which conditions triggered their prediction:

```python
import sys
import os
sys.path.append(os.path.abspath("../dataset and other libs"))

from trace_path import trace_customer_path

# Preprocess the customer row through the pipeline's first two steps
X_transformed = pipeline_dt['clean'].transform(X_train.head(1))
X_filtered = pipeline_dt['ctf'].transform(X_transformed)

# Trace the decision path
trace_customer_path(pipeline_dt, X_filtered, customer_idx=0)
```

This prints every node visited, the condition checked (e.g. `tenure <= 5.5`), whether the customer met it, and the final leaf's churn probability.

---

## Running Inference with the Saved Model

You can load `pipeline_dt.pkl` and feed it raw, uncleaned data directly:

```python
import pickle
import sys
import os
import pandas as pd

# Custom transformer classes must be discoverable
sys.path.append(os.path.abspath("../dataset and other libs"))

# Load pipeline
with open("pipeline_dt.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load sample raw data
raw_data = pd.read_csv("../dataset and other libs/WA_Fn-UseC_-Telco-Customer-Churn.csv")
sample = raw_data.drop(columns=["Churn"]).head(5)

# Predict
preds = pipeline.predict(sample)
probs = pipeline.predict_proba(sample)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    result = "Will Churn" if pred == 1 else "Will Stay"
    print(f"Customer {i + 1}: {result} ({prob:.1%} churn risk)")
```
