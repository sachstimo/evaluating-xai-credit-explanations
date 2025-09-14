"""
Consolidated sampling and clustering utilities for prediction analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score


### Used for initial sampling before making the predictions and generating SHAP and counterfactuals
def cluster_stratified_sample(df, sample_size, cluster_mapping, random_state=42):
    """
    Simple cluster-stratified sampling
    
    Args:
        df: DataFrame to sample from
        sample_size: Number of samples to take
        cluster_mapping: Dict mapping index to cluster_id
        random_state: Random seed
    
    Returns:
        Sampled DataFrame
    """
    if sample_size >= len(df):
        return df.copy()
    
    # Group indices by cluster
    cluster_groups = {}
    for idx in df.index:
        cluster_id = cluster_mapping.get(idx, 0)  # Default cluster 0 if not found
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(idx)
    
    # Calculate samples per cluster (proportional)
    np.random.seed(random_state)
    selected_indices = []
    
    for cluster_id, indices in cluster_groups.items():
        cluster_size = len(indices)
        cluster_proportion = cluster_size / len(df)
        cluster_samples = max(1, int(sample_size * cluster_proportion))
        
        # Sample from this cluster
        if cluster_samples >= cluster_size:
            selected_indices.extend(indices)
        else:
            sampled = np.random.choice(indices, cluster_samples, replace=False)
            selected_indices.extend(sampled)
    
    # If we have too many/few samples, adjust randomly
    if len(selected_indices) > sample_size:
        selected_indices = np.random.choice(selected_indices, sample_size, replace=False)
    elif len(selected_indices) < sample_size:
        remaining = [i for i in df.index if i not in selected_indices]
        if remaining:
            extra_needed = sample_size - len(selected_indices)
            extra = np.random.choice(remaining, min(extra_needed, len(remaining)), replace=False)
            selected_indices.extend(extra)
    
    return df.loc[selected_indices].copy()

def load_predictions(predictions_file="../output/predictions/prediction_results.json"):
    """Load and flatten prediction results with parsed counterfactuals
    
    Args:
        predictions_file: Path to the predictions file
    
    Returns:
        DataFrame with flattened predictions
    """

    with open(predictions_file, "r") as f:
        data = json.load(f)
    
    prediction_results = data["predictions"]
    metadata = data.get("metadata", {})
    
    # Flatten to DataFrame
    rows = []
    for pred_id, pred_dict in prediction_results.items():
        row = {"prediction_id": pred_id}
        
        # Basic prediction info
        pred_info = pred_dict.get("prediction", {})
        for k, v in pred_info.items():
            if k == "input_data":
                for in_k, in_v in v.items():
                    row[f"input_{in_k}"] = in_v
            else:
                row[k] = v
        
        # Cluster and explanations
        row["cluster_id"] = pred_dict.get("cluster_id")
        
        # SHAP
        shap = pred_dict.get("shap_explanation", {})
        for k, v in shap.items():
            if k == "contributions":
                for feat, contrib in v.items():
                    row[f"shap_{feat}"] = contrib
            else:
                row[f"shap_{k}"] = v
        
        # Counterfactuals - keep only parsed data
        counterfactuals_raw = pred_dict.get("counterfactuals", [])
        
        # Parse counterfactuals for easy analysis
        row["cf_count"] = len(counterfactuals_raw)
        row["counterfactuals"] = counterfactuals_raw
        
        # Extract percentile changes if available
        cf_percentile_changes = []
        for cf in counterfactuals_raw:
            if isinstance(cf, dict):
                for feature_name, feature_data in cf.items():
                    if isinstance(feature_data, dict) and 'percentile_change' in feature_data:
                        cf_percentile_changes.append(abs(feature_data['percentile_change']))
                        break
        
        row["cf_percentile_changes"] = cf_percentile_changes
        row["cf_avg_percentile_change"] = np.mean(cf_percentile_changes) if cf_percentile_changes else 0
        
        rows.append(row)
    
    return pd.DataFrame(rows), metadata


def categorize_by_probability(df, bins, labels):
    """Simple probability-based categorization with external boundaries"""
    df['sampling_category'] = pd.cut(
        df['probability'], 
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    return df


def stratified_sample_with_targets(df, sampling_targets, total_samples=100, random_state=42, preserve_age_groups=False):
    """Sample specific numbers from each category, optionally preserving age group proportions"""
    samples = []
    sampled_indices = set()
    
    # Calculate overall age group proportions if preservation is enabled
    age_group_proportions = None
    if preserve_age_groups and 'age_group' in df.columns:
        age_group_proportions = df['age_group'].value_counts(normalize=True).to_dict()
        print(f"Preserving age group proportions: {age_group_proportions}")
        
        # Age-first approach: sample by age groups first, then by categories within each age group
        if preserve_age_groups:
            print("🎯 Using age-first stratification")
            
            # Calculate target samples per age group, last one gets remainder
            age_targets = []
            allocated = 0
            for i, (age_group, target_prop) in enumerate(age_group_proportions.items()):
                if i == len(age_group_proportions) - 1:  # Last gets remainder
                    target = total_samples - allocated
                else:
                    target = max(1, int(total_samples * target_prop))
                    allocated += target
                age_targets.append((age_group, target))
            
            for age_group, age_group_target in age_targets:
                age_group_data = df[df['age_group'] == age_group]
                print(f"  {age_group}: {len(age_group_data)} available → target {age_group_target}")
                
                # Within this age group, try to sample according to category preferences
                age_group_samples = []
                age_group_remaining = age_group_target
                
                # Calculate category proportions based on global targets
                total_target_samples = sum(sampling_targets.values())
                category_props = {cat: n/total_target_samples for cat, n in sampling_targets.items()}
                
                # Priority order for categories within age group, respecting target proportions
                for category in ['BOUNDARY', 'CLEAR_REJECT', 'CLEAR_ACCEPT', 'LEAN_REJECT', 'LEAN_ACCEPT']:
                    if age_group_remaining <= 0:
                        break
                    
                    cat_in_age = age_group_data[age_group_data['sampling_category'] == category]
                    if len(cat_in_age) > 0 and category in category_props:
                        # Sample based on target proportions, not equally
                        target_from_cat = min(age_group_remaining, len(cat_in_age), 
                                           max(1, int(age_group_target * category_props[category])))
                        sampled_cat = cat_in_age.sample(n=target_from_cat, random_state=random_state)
                        age_group_samples.append(sampled_cat)
                        age_group_remaining -= target_from_cat
                
                # Fill remaining randomly from this age group
                if age_group_remaining > 0:
                    already_sampled_indices = set()
                    for sample_df in age_group_samples:
                        already_sampled_indices.update(sample_df.index)
                    remaining_in_age = age_group_data[~age_group_data.index.isin(already_sampled_indices)]
                    if len(remaining_in_age) > 0:
                        additional = remaining_in_age.sample(n=min(age_group_remaining, len(remaining_in_age)), random_state=random_state)
                        age_group_samples.append(additional)
                
                if age_group_samples:
                    age_group_final = pd.concat(age_group_samples, ignore_index=False)
                    samples.append(age_group_final)
                    sampled_indices.update(age_group_final.index)
            
            # Return the age-first result
            df_sampled = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
            return df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Original category-first approach
    for category, n_samples in sampling_targets.items():
        available = df[(df['sampling_category'] == category) & (~df.index.isin(sampled_indices))]
        
        if len(available) == 0:
            print(f"Warning: No samples available for category '{category}'")
            continue
            
        n_to_sample = min(n_samples, len(available))
        if n_to_sample < n_samples:
            print(f"Warning: Only {n_to_sample} samples available for '{category}' (requested {n_samples})")
        
        # Default sampling without age group preservation
        sampled = available.sample(n=n_to_sample, random_state=random_state)
        
        samples.append(sampled)
        sampled_indices.update(sampled.index)
    
    df_sampled = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    
    # Fill remaining if under target
    current_count = len(df_sampled)
    if current_count < total_samples:
        remaining = df[~df.index.isin(sampled_indices)]
        n_fill = min(total_samples - current_count, len(remaining))
        if n_fill > 0:
            print(f"Filling {n_fill} additional samples to reach {total_samples}")
            fill_samples = remaining.sample(n=n_fill, random_state=42)
            df_sampled = pd.concat([df_sampled, fill_samples], ignore_index=True)
    
    return df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)


def export_sampled_predictions(df_sampled, output_dir="../output/predictions", filename=None):
    """Export sampled predictions for detailed analysis"""
    prediction_ids = df_sampled['prediction_id'].tolist()
    sample_size = len(df_sampled)
    
    # Load original predictions
    with open("../output/predictions/prediction_results.json", "r") as f:
        original_data = json.load(f)
    
    # Create sampled version
    sampled_data = {
        "metadata": original_data["metadata"].copy(),
        "predictions": {pid: original_data["predictions"][pid] for pid in prediction_ids if pid in original_data["predictions"]}
    }
    
    # Update metadata
    sampled_data["metadata"]["sample_size"] = sample_size
    sampled_data["metadata"]["sampling_strategy"] = "intelligent_stratified"
    sampled_data["metadata"]["sample_timestamp"] = pd.Timestamp.now().isoformat()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"prediction_results_sampled_{sample_size}.json"
    sample_file = output_path / filename
    
    with open(sample_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)
    
    print(f"✅ Exported {sample_size} sampled predictions to: {sample_file}")
    return sample_file


def extract_baseline_and_cf_values(df_row):
    """
    Extract baseline values and counterfactual values from a single prediction row
    
    Args:
        df_row: Single row from predictions DataFrame
    
    Returns:
        tuple: (baseline_values, counterfactual_list, prediction_info)
    """
    # Extract baseline values (original input data)
    baseline_values = {}
    for col in df_row.index:
        if col.startswith('input_'):
            feature_name = col.replace('input_', '')
            baseline_values[feature_name] = df_row[col]
    
    # Extract counterfactuals
    counterfactuals = df_row['counterfactuals'] if 'counterfactuals' in df_row.index else []
    
    # Extract prediction info
    prediction_info = {
        'prediction_id': df_row.get('prediction_id'),
        'prediction': df_row.get('prediction'),
        'probability': df_row.get('probability'),
        'cluster_id': df_row.get('cluster_id')
    }
    
    return baseline_values, counterfactuals, prediction_info


def extract_cf_feature_changes(counterfactuals):
    """
    Extract feature changes from counterfactuals for verification
    
    Args:
        counterfactuals: List of counterfactual dictionaries
    
    Returns:
        list: List of dictionaries containing feature changes for each CF
    """
    cf_changes = []
    
    for cf in counterfactuals:
        cf_dict = {}
        for feature, change_info in cf.items():
            if isinstance(change_info, dict):
                cf_dict[feature] = {
                    'baseline_value': change_info.get('from'),
                    'cf_value': change_info.get('to'),
                    'difference': change_info.get('diff'),
                    'baseline_percentile': change_info.get('from_percentile'),
                    'cf_percentile': change_info.get('to_percentile'),
                    'percentile_change': change_info.get('percentile_change')
                }
        cf_changes.append(cf_dict)
    
    return cf_changes


def create_cf_verification_dataset(df, max_rows=None):
    """
    Create a structured dataset for counterfactual verification
    
    Args:
        df: Predictions DataFrame
        max_rows: Maximum number of rows to process (None for all)
    
    Returns:
        list: List of dictionaries with baseline and CF data for verification
    """
    verification_data = []
    
    rows_to_process = df.head(max_rows) if max_rows else df
    
    for idx, row in rows_to_process.iterrows():
        baseline_values, counterfactuals, prediction_info = extract_baseline_and_cf_values(row)
        cf_changes = extract_cf_feature_changes(counterfactuals)
        
        verification_data.append({
            'prediction_info': prediction_info,
            'baseline_values': baseline_values,
            'counterfactuals': counterfactuals,
            'cf_feature_changes': cf_changes,
            'num_counterfactuals': len(counterfactuals)
        })
    
    return verification_data

def verify_counterfactual_batch(verification_data, model, max_instances=None):
    """
    Verify multiple counterfactuals by running them through the model
    
    Args:
        verification_data: Output from create_cf_verification_dataset
        model: CreditPredictor model instance
        max_instances: Maximum instances to verify (None for all)
    
    Returns:
        list: Verification results for each counterfactual
    """
    results = []
    instances_to_process = verification_data[:max_instances] if max_instances else verification_data
    
    for instance_data in instances_to_process:
        instance_results = {
            'prediction_id': instance_data['prediction_info']['prediction_id'],
            'original_prediction': instance_data['prediction_info']['prediction'],
            'original_probability': instance_data['prediction_info']['probability'],
            'counterfactuals': []
        }
        
        # Verify each counterfactual for this instance
        for i, cf_changes in enumerate(instance_data['cf_feature_changes']):
            # Create counterfactual instance
            cf_instance = instance_data['baseline_values'].copy()
            for feature, changes in cf_changes.items():
                cf_instance[feature] = changes['cf_value']
            
            # Convert to DataFrame and predict
            cf_df = pd.DataFrame([cf_instance])
            try:
                cf_prediction = model.predict_single(cf_df)
                
                cf_result = {
                    'cf_index': i,
                    'cf_prediction': cf_prediction['prediction'],
                    'cf_probability': cf_prediction['probability'],
                    'prediction_flipped': cf_prediction['prediction'] != instance_data['prediction_info']['prediction'],
                    'probability_change': cf_prediction['probability'] - instance_data['prediction_info']['probability'],
                    'features_changed': list(cf_changes.keys()),
                    'num_features_changed': len(cf_changes)
                }
                
            except Exception as e:
                cf_result = {
                    'cf_index': i,
                    'error': str(e),
                    'prediction_flipped': False
                }
            
            instance_results['counterfactuals'].append(cf_result)
        
        # Calculate summary stats for this instance
        successful_cfs = [cf for cf in instance_results['counterfactuals'] if cf.get('prediction_flipped', False)]
        instance_results['num_successful_cfs'] = len(successful_cfs)
        instance_results['success_rate'] = len(successful_cfs) / len(instance_results['counterfactuals']) if instance_results['counterfactuals'] else 0
        
        results.append(instance_results)
    
    return results