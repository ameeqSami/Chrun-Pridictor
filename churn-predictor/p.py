"""
p.py  –  Main Random Forest Churn Prediction Pipeline
=======================================================
This script trains a full sklearn Pipeline for Telco customer churn
prediction using a Random Forest classifier, evaluates it, and then
traces the decision path taken by a single customer through one tree
of the forest.

Pipeline stages
---------------
1. clean          – ``clean_cls``   : raw data cleaning & encoding
2. ctf            – ``CorrelationThresholdFilter`` : feature selection
3. pridict_thresh – ``pridict_thresh``  : RF with a custom 0.64 threshold

Why threshold=0.64?
    In churn contexts missing a churner (FN) is costly, but spamming
    every customer with retention offers is also expensive. A threshold
    above 0.5 trades recall for precision — only flag customers where the
    model is very confident they will churn.
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Add the shared utilities directory to the Python path so we can import
# our custom transformers and helpers
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath("../dataset and other libs"))
from cleaningcls import clean_cls                       # Data cleaning transformer
from ctf import CorrelationThresholdFilter              # Feature selection transformer
from pridict_thresh import pridict_thresh               # Custom threshold classifier

# ---------------------------------------------------------------------------
# 1. Load raw dataset
# ---------------------------------------------------------------------------
# Full path used to make the script runnable from any working directory
df = pd.read_csv(
    'D:/Repos/Chrun-Pridictor/dataset and other libs/WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

# ---------------------------------------------------------------------------
# 2. Separate features (X) and target (y)
# ---------------------------------------------------------------------------
X = df.drop(columns=['Churn'])               # Drop the target column from features
y = df['Churn'].map({'Yes': 1, 'No': 0})     # Encode target: 1=Churn, 0=No Churn

# ---------------------------------------------------------------------------
# 3. Stratified Train-Test Split (80/20)
#    stratify=y ensures the churn ratio is preserved in both splits,
#    which is important given the class imbalance (~26% churn rate)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Build the sklearn Pipeline
# ---------------------------------------------------------------------------
pipeline_dt = Pipeline([
    # Stage 1: Clean raw data (drop IDs, encode categoricals, fix TotalCharges)
    ('clean', clean_cls()),

    # Stage 2: Drop features with low correlation to the churn target
    #          threshold=0.14 keeps only features with |r| >= 0.14
    ('ctf', CorrelationThresholdFilter(threshold=0.14)),

    # Stage 3: Train Random Forest with a custom 0.64 decision threshold
    #          - n_estimators=200 : more trees → lower variance
    #          - max_depth=8      : prevents overfitting on training noise
    #          - class_weight='balanced' : up-weights the minority churn class
    #          - thresh=0.64      : requires 64% confidence to flag as churn
    ('pridict_thresh', pridict_thresh(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight='balanced',    # Handle class imbalance automatically
            random_state=42,
        ),
        thresh=0.64                     # Custom high-precision threshold
    )),
])

# ---------------------------------------------------------------------------
# 5. Train the pipeline (all three stages: clean → filter → fit RF)
# ---------------------------------------------------------------------------
pipeline_dt.fit(X_train, y_train)

# Predict probabilities and binary labels on the held-out test set
y_prob_dt = pipeline_dt.predict_proba(X_test)[:, 1]  # Churn probability column
y_pred_dt = pipeline_dt.predict(X_test)               # Binary predictions (0/1)

# Compute ROC-AUC (threshold-independent measure of discrimination power)
auc_dt = roc_auc_score(y_test, y_prob_dt)
# print('=== Pipeline Evaluation ===')
# print(f'ROC-AUC Score: {auc_dt:.4f}')
# print('\nConfusion Matrix:')
# print(confusion_matrix(y_test, y_pred_dt))
# print('\nClassification Report:')
# print(classification_report(y_test, y_pred_dt))

# ---------------------------------------------------------------------------
# 6. Threshold sweep analysis
#    Iterate over a range of thresholds to understand the Precision/Recall
#    trade-off, helping choose the optimal operating point
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

thresholds = np.arange(0.25, 0.66, 0.01)
for t in thresholds:
    y_pred = (y_prob_dt > t).astype(int)
    # Uncomment the line below to print the metrics table:
    # print(f"thresh={t:.2f} | P={precision_score(y_test,y_pred):.2f} | R={recall_score(y_test,y_pred):.2f} | F1={f1_score(y_test,y_pred):.2f}")

# ---------------------------------------------------------------------------
# 7. Feature importance analysis
# ---------------------------------------------------------------------------
import pandas as pd

# Extract the trained underlying RandomForest from the pridict_thresh wrapper
model = pipeline_dt['pridict_thresh'].model

# feature_importances_ is a numpy array: higher = more useful for splitting
importances = model.feature_importances_

# Retrieve column names from the CTF step (these are post-filtering features)
feature_names = pipeline_dt['ctf'].get_feature_names_out()

# Build a ranked DataFrame for easy inspection
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

active_features = feature_importance_df

# print("=== Features Used by the Decision Tree (Ranked by Importance) ===")
# print(active_features.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Decision path tracing — understand how the model decided for one customer
#    We pre-process a small slice of training data and pass it to the tracer
# ---------------------------------------------------------------------------
# Transform the first 5 training rows through the cleaning and filtering steps
sample_output = pipeline_dt['clean'].transform(X_train.head(5))
sample_output = pipeline_dt['ctf'].transform(sample_output)
feature_names = list(sample_output.columns)  # Final feature names after all processing

from trace_path import trace_customer_path
import trace_path

# Transform just the first training customer for decision-path tracing
X_transformed = pipeline_dt['clean'].transform(X_train.head(1))
X_filtered = pipeline_dt['ctf'].transform(X_transformed)

# Print the full decision path for customer #0
trace_customer_path(pipeline_dt, X_filtered)

# ---------------------------------------------------------------------------
# 9. Per-customer feature vectors (useful for debugging specific predictions)
# ---------------------------------------------------------------------------
# Transform the full test set and reset index so iloc works correctly
cleaned_features = pipeline_dt.named_steps['clean'].transform(X_test.reset_index(drop=True))
filtered_features = pipeline_dt.named_steps['ctf'].transform(cleaned_features.reset_index(drop=True))
# Uncomment the block below to print each test customer's features + prediction:
# for i in range(filtered_features.shape[0]):
#     print(filtered_features.iloc[i].to_dict())
#     print('prediction: ', y_pred_dt[i])
#     print('-'*20)

# ---------------------------------------------------------------------------
# 10. Low-level decision path inspection via sklearn's tree API
#     This section manually walks the first estimator's tree for customer #0
# ---------------------------------------------------------------------------
model = pipeline_dt['pridict_thresh'].model    # The RandomForest
tree_model = model.estimators_[0]              # First decision tree in the forest

# decision_path returns a sparse binary matrix: rows=customers, cols=nodes
node_indicator = tree_model.decision_path(X_filtered)

# apply() returns the leaf node ID for each customer
leave_id = tree_model.apply(X_filtered)

# Extract the sequence of node IDs visited by customer 0
node_index = node_indicator.getrow(0).indices

# Re-fetch model and tree structure references
model = pipeline_dt['pridict_thresh'].model
tree = tree_model.tree_

# Print the decision rule at each node along the path for customer 0
feature_names = list(pipeline_dt['ctf'].get_feature_names_out())
for i in range(len(node_indicator.indices)):
    # Feature used at this split node
    print(feature_names[tree.feature[node_index[i]]])
    # Threshold value for the split
    print(tree.threshold[node_index[i]])
    # The customer's actual value for that feature
    print(X_filtered.iloc[0, tree.feature[node_index[i]]])
    # Class sample counts at this node [no_churn_samples, churn_samples]
    print(tree.value[node_index[i]])
    print('-' * 20)
