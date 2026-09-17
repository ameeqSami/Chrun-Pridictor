import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def _extract_predictor_and_model(pipeline_dt):
    """Extracts predictor step and underlying classifier model from pipeline or estimator."""
    if hasattr(pipeline_dt, 'named_steps'):
        if 'pridict_thresh' in pipeline_dt.named_steps:
            predictor = pipeline_dt.named_steps['pridict_thresh']
            model = getattr(predictor, 'model', predictor)
        elif 'model' in pipeline_dt.named_steps:
            predictor = pipeline_dt.named_steps['model']
            model = getattr(predictor, 'model', predictor)
        else:
            predictor = pipeline_dt.steps[-1][1]
            model = getattr(predictor, 'model', predictor)
    else:
        predictor = pipeline_dt
        model = getattr(predictor, 'model', predictor)
    return predictor, model


def _get_feature_names(pipeline_dt, X_filtered):
    """Retrieves feature names from DataFrame or pipeline CTF step."""
    if isinstance(X_filtered, pd.DataFrame):
        return list(X_filtered.columns)
    if hasattr(pipeline_dt, 'named_steps'):
        if 'ctf' in pipeline_dt.named_steps and hasattr(pipeline_dt['ctf'], 'get_feature_names_out'):
            return list(pipeline_dt['ctf'].get_feature_names_out())
    if hasattr(pipeline_dt, 'feature_names_in_'):
        return list(pipeline_dt.feature_names_in_)
    return [f"feature_{i}" for i in range(X_filtered.shape[1])]


def _select_tree_estimator(predictor, model, sample_customer, tree_index=None):
    """
    Selects a decision tree from RandomForest that has the same output as the
    RandomForest output after voting (and custom thresholding if applicable).
    If model is a single DecisionTree, returns that model directly.
    """
    if isinstance(sample_customer, pd.DataFrame):
        sample_df = sample_customer
        sample_arr = sample_customer.values
    elif isinstance(sample_customer, pd.Series):
        sample_df = sample_customer.to_frame().T
        sample_arr = sample_df.values
    else:
        sample_arr = np.atleast_2d(sample_customer)
        sample_df = pd.DataFrame(sample_arr)

    # 1. Ensemble prediction after voting / thresholding
    if hasattr(predictor, 'predict'):
        ensemble_pred = int(predictor.predict(sample_df)[0])
    else:
        ensemble_pred = int(model.predict(sample_arr)[0])

    ensemble_prob = None
    if hasattr(predictor, 'predict_proba'):
        prob_arr = predictor.predict_proba(sample_df)[0]
        ensemble_prob = float(prob_arr[1]) if len(prob_arr) > 1 else float(prob_arr[0])
    elif hasattr(model, 'predict_proba'):
        prob_arr = model.predict_proba(sample_arr)[0]
        ensemble_prob = float(prob_arr[1]) if len(prob_arr) > 1 else float(prob_arr[0])

    # 2. Check if model is RandomForest / ensemble with estimators_
    if not isinstance(model, RandomForestClassifier) and not hasattr(model, 'estimators_'):
        return {
            'tree_model': model,
            'tree_index': None,
            'is_rf': False,
            'ensemble_pred': ensemble_pred,
            'ensemble_prob': ensemble_prob,
            'tree_pred': ensemble_pred,
            'matching_count': 1,
            'total_trees': 1,
        }

    estimators = model.estimators_
    total_trees = len(estimators)

    # 3. Evaluate each tree's prediction for this customer
    tree_preds = []
    tree_probs = []
    for est in estimators:
        leaf_idx = est.apply(sample_arr)[0]
        counts = est.tree_.value[leaf_idx][0]
        total_count = counts.sum()
        p1 = counts[1] / total_count if (len(counts) > 1 and total_count > 0) else 0.0
        t_pred = int(np.argmax(counts))
        tree_preds.append(t_pred)
        tree_probs.append(p1)

    # Filter trees that match the ensemble output after voting
    matching_indices = [idx for idx, p in enumerate(tree_preds) if p == ensemble_pred]
    matching_count = len(matching_indices)

    if tree_index is not None:
        chosen_index = tree_index
    else:
        if matching_indices:
            # Pick the matching tree closest to the ensemble's predicted probability
            if ensemble_prob is not None:
                chosen_index = min(matching_indices, key=lambda idx: abs(tree_probs[idx] - ensemble_prob))
            else:
                chosen_index = matching_indices[0]
        else:
            chosen_index = 0

    chosen_tree = estimators[chosen_index]
    chosen_tree_pred = tree_preds[chosen_index]

    return {
        'tree_model': chosen_tree,
        'tree_index': chosen_index,
        'is_rf': True,
        'ensemble_pred': ensemble_pred,
        'ensemble_prob': ensemble_prob,
        'tree_pred': chosen_tree_pred,
        'matching_count': matching_count,
        'total_trees': total_trees,
    }


def trace_customer_path(pipeline_dt, X_filtered, customer_idx=0, tree_index=None):
    """
    Traces and prints the decision path of a customer through a decision tree.
    For RandomForest, automatically selects a tree with the same output as
    the forest after voting (or uses tree_index if explicitly specified).
    """
    predictor, model = _extract_predictor_and_model(pipeline_dt)

    if isinstance(X_filtered, pd.DataFrame):
        sample_customer = X_filtered.iloc[[customer_idx]]
        X_arr = X_filtered.values
    else:
        X_arr = np.atleast_2d(X_filtered)
        sample_customer = X_arr[customer_idx:customer_idx + 1]

    info = _select_tree_estimator(predictor, model, sample_customer, tree_index=tree_index)
    tree_model = info['tree_model']

    if info['is_rf']:
        chosen_idx = info['tree_index']
        total = info['total_trees']
        matches = info['matching_count']
        ens_pred = info['ensemble_pred']
        tree_pred = info['tree_pred']
        ens_prob_str = f" ({info['ensemble_prob']:.1%} churn prob)" if info['ensemble_prob'] is not None else ""
        print(f"[INFO] RandomForest detected - Ensemble voted: {ens_pred}{ens_prob_str}")
        print(f"[INFO] {matches}/{total} trees agreed with ensemble vote. Selected Tree #{chosen_idx} (tree pred: {tree_pred})")
        if tree_pred != ens_pred:
            print(f"[WARNING] Tree #{chosen_idx} predicted {tree_pred}, which differs from ensemble vote {ens_pred}!")
        print("=" * 40)
    else:
        print(f"[INFO] Single DecisionTree detected - Prediction: {info['ensemble_pred']}")
        print("=" * 40)

    node_indicator = tree_model.decision_path(X_arr)
    leave_id = tree_model.apply(X_arr)
    node_index = node_indicator.indices[
        node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]
    ]

    print(f"Customer's Decision Path (Node IDs): {node_index.tolist()}")
    print(f"Final Leaf Node: {leave_id[customer_idx]}")
    print("=" * 40)

    tree = tree_model.tree_
    feature_names = _get_feature_names(pipeline_dt, X_filtered)
    sample_series = X_filtered.iloc[customer_idx] if isinstance(X_filtered, pd.DataFrame) else pd.Series(X_arr[customer_idx], index=feature_names)

    for node_id in node_index:
        if tree.children_left[node_id] != tree.children_right[node_id]:
            # Split node
            feat_idx = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            feat_name = feature_names[feat_idx]
            customer_value = sample_series[feat_name]
            direction = "LEFT (<=)" if customer_value <= thresh else "RIGHT (>)"

            print(f"Node {node_id} (Split Node):")
            print(f"  - Feature checked: '{feat_name}'")
            print(f"  - Tree rule: <= {thresh:.4f}")
            print(f"  - Customer's value: {customer_value}  ->  Goes {direction}")
            print("-" * 40)
        else:
            # Leaf node
            class_dist = tree.value[node_id][0]
            total = class_dist.sum()
            p0 = class_dist[0] / total if total > 0 else 0.0
            p1 = class_dist[1] / total if (len(class_dist) > 1 and total > 0) else 0.0
            leaf_pred = int(np.argmax(class_dist))
            samples = int(tree.n_node_samples[node_id])

            print(f"Node {node_id} (Leaf - Final Destination):")
            print(f"  - Node samples: {samples}")
            print(f"  - Class distribution  ->  No Churn: {p0:.1%}, Churn: {p1:.1%}")
            print(f"  - Churn probability at this leaf: {p1:.1%}")
            print(f"  - Tree decision: {'Churn (1)' if leaf_pred == 1 else 'No Churn (0)'}")
            print("=" * 40)


def get_customer_path_df(pipeline_dt, X_filtered, customer_idx=0, tree_index=None):
    """
    Returns the decision path of a customer as a structured pandas DataFrame.
    For RandomForest, automatically selects a tree with the same output as
    the forest after voting (or uses tree_index if explicitly specified).
    """
    predictor, model = _extract_predictor_and_model(pipeline_dt)

    if isinstance(X_filtered, pd.DataFrame):
        sample_customer = X_filtered.iloc[[customer_idx]]
        X_arr = X_filtered.values
    else:
        X_arr = np.atleast_2d(X_filtered)
        sample_customer = X_arr[customer_idx:customer_idx + 1]

    info = _select_tree_estimator(predictor, model, sample_customer, tree_index=tree_index)
    tree_model = info['tree_model']

    node_indicator = tree_model.decision_path(X_arr)
    leave_id = tree_model.apply(X_arr)
    node_index = node_indicator.indices[
        node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]
    ]

    tree = tree_model.tree_
    feature_names = _get_feature_names(pipeline_dt, X_filtered)
    sample_series = X_filtered.iloc[customer_idx] if isinstance(X_filtered, pd.DataFrame) else pd.Series(X_arr[customer_idx], index=feature_names)

    path_data = []

    for node_id in node_index:
        class_dist = tree.value[node_id][0]
        total = class_dist.sum()
        p0 = round(class_dist[0] / total, 4) if total > 0 else 0.0
        p1 = round(class_dist[1] / total, 4) if (len(class_dist) > 1 and total > 0) else 0.0
        samples = int(tree.n_node_samples[node_id])

        if tree.children_left[node_id] != tree.children_right[node_id]:
            feat_idx = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            feat_name = feature_names[feat_idx]
            customer_value = sample_series[feat_name]
            condition_met = customer_value <= thresh

            path_data.append({
                "Node ID": node_id,
                "Node Type": "Split Node",
                "Feature Checked": feat_name,
                "Threshold Rule": f"<= {thresh:.4f}",
                "Customer Value": customer_value,
                "Condition Met": condition_met,
                "Direction": "Left (<=)" if condition_met else "Right (>)",
                "Samples": samples,
                "Churn Probability": p1,
                "Class Distribution": f"No Churn: {p0:.1%}, Churn: {p1:.1%}"
            })
        else:
            leaf_pred = int(np.argmax(class_dist))
            path_data.append({
                "Node ID": node_id,
                "Node Type": "Leaf Node",
                "Feature Checked": "N/A (Final Destination)",
                "Threshold Rule": "N/A",
                "Customer Value": "N/A",
                "Condition Met": "N/A",
                "Direction": "N/A",
                "Samples": samples,
                "Churn Probability": p1,
                "Class Distribution": f"No Churn: {p0:.1%}, Churn: {p1:.1%}",
                "Prediction": leaf_pred
            })

    return pd.DataFrame(path_data)