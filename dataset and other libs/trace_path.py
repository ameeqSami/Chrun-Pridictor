def trace_customer_path(pipeline_dt, X_filtered, customer_idx=0, tree_index=0):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    model = pipeline_dt['pridict_thresh'].model

    # --- Handle both DecisionTree and RandomForest ---
    if isinstance(model, RandomForestClassifier):
        # For RF: trace through one individual tree
        tree_model = model.estimators_[tree_index]
        print(f"ℹ️  RandomForest detected — tracing through tree #{tree_index} of {len(model.estimators_)}")
        print("=" * 40)
        node_indicator = tree_model.decision_path(X_filtered)
    else:
        # Single DecisionTree: decision_path returns sparse matrix directly
        tree_model = model
        result = tree_model.decision_path(X_filtered)
        # DT returns sparse matrix directly
        node_indicator = result

    leave_id = tree_model.apply(X_filtered)
    node_index = node_indicator.indices[
        node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]
    ]

    print(f"Customer's Decision Path (Node IDs): {node_index}")
    print(f"Final Leaf Node: {leave_id[customer_idx]}")
    print("=" * 40)

    tree = tree_model.tree_
    feature_names = list(pipeline_dt['ctf'].get_feature_names_out())
    sample_customer = X_filtered.iloc[customer_idx]

    for node_id in node_index:
        if tree.children_left[node_id] != tree.children_right[node_id]:
            # Split node
            feat_idx = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            feat_name = feature_names[feat_idx]
            customer_value = sample_customer[feat_name]

            print(f"Node {node_id} (Split Node):")
            print(f"  - Feature checked: '{feat_name}'")
            print(f"  - Tree rule: <= {thresh:.4f}")
            print(f"  - Customer's value: {customer_value}  →  Goes {'LEFT (<=)' if customer_value <= thresh else 'RIGHT (>)'}")
            print("-" * 40)
        else:
            # Leaf node
            class_dist = tree.value[node_id][0]
            total = class_dist.sum()
            churn_prob = class_dist[1] / total if total > 0 else 0
            print(f"Node {node_id} (Leaf — Final Destination):")
            print(f"  - Class distribution  →  No Churn: {int(class_dist[0])}, Churn: {int(class_dist[1])}")
            print(f"  - Churn probability at this leaf: {churn_prob:.1%}")
            print("=" * 40)


import pandas as pd

def get_customer_path_df(pipeline_dt, X_filtered, customer_idx=0):
    tree_model = pipeline_dt['model']
    node_indicator = tree_model.decision_path(X_filtered)
    leave_id = tree_model.apply(X_filtered)
    node_index = node_indicator.indices[node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]]

    tree = tree_model.tree_
    feature_names = pipeline_dt['ctf'].get_feature_names_out()
    sample_customer = X_filtered.iloc[customer_idx] 

    path_data = []

    for node_id in node_index:
        # Dynamically check the churn count for the current node_id
        if tree.value[node_id][0][1] <= 0.45:    
            if tree.children_left[node_id] != tree.children_right[node_id]:
                feat_idx = tree.feature[node_id]
                thresh = tree.threshold[node_id]
                feat_name = feature_names[feat_idx]
                customer_value = sample_customer[feat_name]
                
                # Append split node details
                path_data.append({
                    "Node ID": node_id,
                    "Node Type": "Split Node",
                    "Feature Checked": feat_name,
                    "Threshold Rule": f"<= {thresh:.4f}",
                    "Customer Value": customer_value,
                    "Condition Met": thresh <= customer_value,
                    "Class Distribution": str(tree.value[node_id])
                })
            else:
                # Append leaf node details
                path_data.append({
                    "Node ID": node_id,
                    "Node Type": "Leaf Node",
                    "Feature Checked": "N/A (Final Destination)",
                    "Threshold Rule": "N/A",
                    "Customer Value": "N/A",
                    "Condition Met": "N/A",
                    "Class Distribution": str(tree.value[node_id])
                })
                
    # Convert the collected records into a pandas DataFrame
    return pd.DataFrame(path_data)