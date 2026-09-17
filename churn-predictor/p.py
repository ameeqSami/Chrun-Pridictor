import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

sys.path.append(os.path.abspath("../dataset and other libs"))
from cleaningcls import clean_cls
from ctf import CorrelationThresholdFilter
from pridict_thresh import pridict_thresh

# Load raw dataset
df = pd.read_csv('D:/Repos/Chrun-Pridictor/dataset and other libs/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline_dt = Pipeline([
    ('clean', clean_cls()),
    ('ctf', CorrelationThresholdFilter(threshold=0.14)),
    ('pridict_thresh', pridict_thresh(RandomForestClassifier(
            n_estimators=200,           # More trees for stability
            max_depth=8,             # Prevent hyper-specific splits
            class_weight='balanced',    # Handle churn class imbalance
            random_state=42), thresh=0.64))])    
                             # Your custom high-precision threshold
pipeline_dt.fit(X_train, y_train)
y_prob_dt = pipeline_dt.predict_proba(X_test)[:,1]
y_pred_dt = pipeline_dt.predict(X_test)

auc_dt = roc_auc_score(y_test, y_prob_dt)
# print('=== Pipeline Evaluation ===')
# print(f'ROC-AUC Score: {auc_dt:.4f}')
# print('\nConfusion Matrix:')
# print(confusion_matrix(y_test, y_pred_dt))
# print('\nClassification Report:')
# print(classification_report(y_test, y_pred_dt))

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

thresholds = np.arange(0.25, 0.66, 0.01)
for t in thresholds:
    y_pred = (y_prob_dt > t).astype(int)
    # print(f"thresh={t:.2f} | P={precision_score(y_test,y_pred):.2f} | R={recall_score(y_test,y_pred):.2f} | F1={f1_score(y_test,y_pred):.2f}")

import pandas as pd

# 1. Get the trained model from your pipeline
model = pipeline_dt['pridict_thresh'].model

# 2. Get the feature importances (scores of how much the tree uses each feature)
importances = model.feature_importances_

# 3. Get the feature names from your pipeline's preprocessing steps

feature_names = pipeline_dt['ctf'].get_feature_names_out()


# 4. Combine them into a clean DataFrame and sort by most important
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

active_features = feature_importance_df

# print("=== Features Used by the Decision Tree (Ranked by Importance) ===")
# print(active_features.to_string(index=False))
# feature_names

sample_output = pipeline_dt['clean'].transform(X_train.head(5))
sample_output = pipeline_dt['ctf'].transform(sample_output)
feature_names = list(sample_output.columns)
from trace_path import trace_customer_path
import trace_path

X_transformed = pipeline_dt['clean'].transform(X_train.head(1))
X_filtered = pipeline_dt['ctf'].transform(X_transformed)
trace_customer_path(pipeline_dt, X_filtered )
# print(feature_names)
cleaned_features = pipeline_dt.named_steps['clean'].transform(X_test.reset_index(drop=True))
filtered_features = pipeline_dt.named_steps['ctf'].transform(cleaned_features.reset_index(drop=True))
# for i in range(filtered_features.shape[0]):
    # print(filtered_features.iloc[i].to_dict())
    # print('prediction: ', y_pred_dt[i])
    # print('-'*20)
model = pipeline_dt['pridict_thresh'].model
tree_model = model.estimators_[0]
node_indicator = tree_model.decision_path(X_filtered)
print(node_indicator)
leave_id= tree_model.apply(X_filtered)
print(leave_id)
node_index = node_indicator.getrow(0).indices
print(node_index)
model = pipeline_dt['pridict_thresh'].model

tree = tree_model.tree_
print(tree.children_left[node_index])
print(tree.children_right[node_index])
print(tree.feature[node_index])
print(tree.threshold[node_index])
print(tree.value[node_index])
# --- Handle both DecisionTree and RandomForest ---
# if isinstance(model, RandomForestClassifier):
#     # For RF: trace through one individual tree
#     tree_model = model.estimators_[tree_index]
#     print(f"ℹ️  RandomForest detected — tracing through tree #{tree_index} of {len(model.estimators_)}")
#     print("=" * 40)
#     node_indicator = tree_model.decision_path(X_filtered)
# else:
#     # Single DecisionTree: decision_path returns sparse matrix directly
#     tree_model = model
#     result = tree_model.decision_path(X_filtered)
#     # DT returns sparse matrix directly
#     node_indicator = result

# leave_id = tree_model.apply(X_filtered)
# node_index = node_indicator.indices[
#     node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]
# ]

# print(f"Customer's Decision Path (Node IDs): {node_index}")
# print(f"Final Leaf Node: {leave_id[customer_idx]}")
# print("=" * 40)

# tree = tree_model.tree_
# feature_names = list(pipeline_dt['ctf'].get_feature_names_out())
# sample_customer = X_filtered.iloc[customer_idx]

# for node_id in node_index:
#     if tree.children_left[node_id] != tree.children_right[node_id]:
#         # Split node
#         feat_idx = tree.feature[node_id]
#         thresh = tree.threshold[node_id]
#         feat_name = feature_names[feat_idx]
#         customer_value = sample_customer[feat_name]

#         print(f"Node {node_id} (Split Node):")
#         print(f"  - Feature checked: '{feat_name}'")
#         print(f"  - Tree rule: <= {thresh:.4f}")
#         print(f"  - Customer's value: {customer_value}  →  Goes {'LEFT (<=)' if customer_value <= thresh else 'RIGHT (>)'}")
#         print("-" * 40)
#     else:
#         # Leaf node
#         class_dist = tree.value[node_id][0]
#         total = class_dist.sum()
#         churn_prob = class_dist[1] / total if total > 0 else 0
#         print(f"Node {node_id} (Leaf — Final Destination):")
#         print(f"  - Class distribution  →  No Churn: {int(class_dist[0])}, Churn: {int(class_dist[1])}")
#         print(f"  - Churn probability at this leaf: {churn_prob:.1%}")
#         print("=" * 40)
