def trace_customer_path(pipeline_dt, X_filtered, customer_idx=0):
    tree_model = pipeline_dt['model']
    node_indicator = tree_model.decision_path(X_filtered)
    leave_id = tree_model.apply(X_filtered)
    node_index = node_indicator.indices[node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]]

    print(f"Customer's Decision Path (Node IDs): {node_index}")
    print(f"Final Leaf Node: {leave_id[customer_idx]}")
    print("=" * 40)
    
    tree = tree_model.tree_
    feature_names = pipeline_dt['ctf'].get_feature_names_out()
    sample_customer = X_filtered.iloc[customer_idx] 

    for node_id in node_index:
        # Dynamically checks the churn count for the current node_id
        if tree.value[node_id][0][1] <= 0.45:    
            if tree.children_left[node_id] != tree.children_right[node_id]:
                feat_idx = tree.feature[node_id]
                thresh = tree.threshold[node_id]
                feat_name = feature_names[feat_idx]
                
                # Get this specific customer's value for this feature
                customer_value = sample_customer[feat_name]
                
                print(f"Node {node_id} (Split Node):")
                print(f"  - Feature checked: '{feat_name}'")
                print(f"  - Tree rule: <= {thresh:.4f}")
                print(f"  - Customer's actual value: {customer_value} (Condition met: {thresh <= customer_value})")
                print("-" * 40)
            else:
                print(f"Node {node_id} (Leaf Node - Final Destination):")
                print(f"  - Class distribution (No Churn, Churn): {tree.value[node_id]}")
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