#!/usr/bin/env python3
"""
Batch prediction script that generates prediction_results.json
Optimized for batch processing with SHAP and Counterfactual explanations
Uses simplified preprocessing without log transformations for better interpretability
"""

import sys
import os
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

from config.config import settings
import os

import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from xai_pkg.model_explainer.predictor import CreditPredictor
from xai_pkg.model_explainer.base_explainer import BaseExplainer
from xai_pkg.model_explainer.shap_explainer import SHAPExplainer
from xai_pkg.model_explainer.counterfactual_explainer import CounterfactualExplainer
from xai_pkg.sampling.utils import cluster_stratified_sample

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
CF_per_instance = settings.get("COUNTERFACTUALS_PER_INSTANCE", 3)
shap_background_samples = settings.get("SHAP_BACKGROUND_SAMPLES", 500)
save_outputs = True
verbose = False

# File paths
model_path = "output/models/best_model.pkl"
metadata_path = settings["METADATA_PATH"]
processed_train_data_path = os.path.join(settings["DATA_DIR"], "processed/X_train_undersampled.csv")
y_train_path = os.path.join(settings["DATA_DIR"], "processed/y_train_undersampled.csv")

# Test data paths
test_data_path = os.path.join(settings["DATA_DIR"], "raw/cs-test.csv")
sample_size = int(os.getenv("PREDICTION_SAMPLE_SIZE", settings.get("PREDICTION_SAMPLE_SIZE", 500)))  # Allow env override
output_path = settings["PREDICTIONS_PATH"]

logger.info("🚀 Starting batch prediction and explanation generation...")

# Load model and training data
with open(metadata_path, 'r') as f:
    model_metadata = json.load(f)

loaded_model = joblib.load(model_path)

# Load training data for explainers
train_df = pd.read_csv(processed_train_data_path)
y_train = pd.read_csv(y_train_path)

if "SeriousDlqin2yrs" in train_df.columns:
    logger.info("Removing target column from training features")
    train_df.drop(columns=["SeriousDlqin2yrs"], inplace=True)

# Prepare training data with target for counterfactual explainer
train_df_with_target = train_df.copy()
train_df_with_target["SeriousDlqin2yrs"] = y_train.values.flatten()

logger.info(f"Training data loaded: {len(train_df)} instances")

# Load test data and sample for batch processing
df_test_full = pd.read_csv(test_data_path)
logger.info(f"Loaded full test dataset: {len(df_test_full)} instances")

# Load cluster assignments if available
cluster_mapping = {}
cluster_file_path = os.path.join(settings["PROJECT_ROOT"], "output", "clustering", "cluster_assignments.json")

if os.path.exists(cluster_file_path):
    try:
        with open(cluster_file_path, 'r') as f:
            cluster_data = json.load(f)
        cluster_mapping = {int(k): int(v) for k, v in cluster_data['index_to_cluster'].items()}
        logger.info(f"Loaded cluster assignments for {len(cluster_mapping)} instances")
    except Exception as e:
        logger.warning(f"Failed to load clusters: {e}")
        cluster_mapping = {}

# Sample test data (cluster-stratified if clusters available)
if sample_size < len(df_test_full):
    if cluster_mapping:
        logger.info(f"Using cluster-stratified sampling for {sample_size} instances")
        df_test = cluster_stratified_sample(df_test_full, sample_size, cluster_mapping)
    else:
        logger.info(f"Using random sampling for {sample_size} instances")
        df_test = df_test_full.sample(n=sample_size, random_state=42)
else:
    logger.info(f"Using all {len(df_test_full)} test instances")
    df_test = df_test_full

# Initialize explainers
predictor = CreditPredictor(
    model_pipeline=loaded_model,
    model_metadata=model_metadata
)

base_explainer = BaseExplainer(
    predictor=predictor,
    model_training_data=train_df,
    outcome_name="SeriousDlqin2yrs"
)

shap_explainer = SHAPExplainer(
    base_explainer=base_explainer,
    n_background_samples=shap_background_samples,
    warning_threshold=0.2,
    suppress_shap_warnings=True
)

cf_explainer = CounterfactualExplainer(
    base_explainer=base_explainer,
    train_df_with_target=train_df_with_target,
    total_CFs=CF_per_instance,
    immutable_features=settings.get('IMMUTABLE_FEATURES', ['age', 'imputed_income', 'NumberOfDependents']),
    desired_class="opposite"
)

logger.info("✅ All explainers initialized successfully")

# Generate predictions and setup results structure
logger.info(f"🔮 Generating predictions for {len(df_test)} instances...")
batch_predictions = predictor.predict_batch(df_test)
prediction_ids = list(batch_predictions.keys())

logger.info(f"✅ Generated {len(batch_predictions)} predictions")

# Prepare results structure
results = {
    "metadata": {
        "generation_timestamp": datetime.now().isoformat(),
        "num_predictions": len(batch_predictions),
        "test_instances": len(df_test),
        "cf_per_instance": CF_per_instance,
        "shap_background_samples": shap_background_samples,
        "model_type": model_metadata.get('model_type', 'unknown')
    },
    "predictions": {}
}

# Initialize prediction entries with cluster information (index-aligned, O(n))
df_test_indices = list(df_test.index)
for i, (pred_id, prediction) in enumerate(batch_predictions.items()):
    cluster_id = None
    if cluster_mapping:
        original_idx = df_test_indices[i]
        cluster_id = cluster_mapping.get(original_idx)

    results["predictions"][pred_id] = {
        "prediction": prediction,
        "cluster_id": cluster_id,
        "shap_explanation": None,
        "counterfactuals": []
    }

# Generate SHAP explanations using ThreadPool for parallelization
logger.info("🔍 Generating SHAP explanations...")

def process_shap_explanation(item):
    i, (pred_id, prediction) = item
    instance_row = df_test.iloc[i:i+1].copy()
    try:
        shap_explanation = shap_explainer.explain_single(
            instance_row, prediction, verbose=False
        )
        return pred_id, shap_explanation
    except Exception as e:
        logger.warning(f"SHAP generation failed for {pred_id}: {e}")
        return pred_id, {"error": str(e)}

# Use ThreadPool for parallel SHAP processing
max_workers = min(settings["MAX_CONCURRENT_WORKERS"], len(batch_predictions))  # Limit to 4 threads to avoid overwhelming
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all SHAP tasks
    future_to_pred = {
        executor.submit(process_shap_explanation, (i, item)): item[0] 
        for i, item in enumerate(batch_predictions.items())
    }
    
    # Process completed SHAP explanations
    for future in tqdm(as_completed(future_to_pred), total=len(batch_predictions), desc="SHAP explanations"):
        pred_id, shap_result = future.result()
        results["predictions"][pred_id]["shap_explanation"] = shap_result


# Generate counterfactual explanations
logger.info("🔮 Generating Counterfactual explanations...")

def process_single_counterfactual(pred_id: str, instance_row: pd.DataFrame, timeout: int = 5) -> List[Dict[str, Any]]:
    """Process counterfactual explanation for a single instance with timeout protection."""
    import time
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"CF generation timeout after {timeout} seconds")
    
    try:
        # Set up signal-based timeout for more reliable interruption
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        start_time = time.time()
        
        try:
            cf_result = cf_explainer.explain_single(
                instance_row, verbose=False, show_only_changes=True
            )
            
            elapsed = time.time() - start_time
            signal.alarm(0)  # Cancel the alarm
            
            # Normalize result format
            if not cf_result:
                return [{"info": "No counterfactuals found"}]
            elif not isinstance(cf_result, list):
                return [cf_result]
            else:
                return cf_result
                
        except TimeoutError:
            elapsed = time.time() - start_time
            return [{"info": f"No counterfactuals found (timeout after {elapsed:.1f}s)"}]
            
    except Exception as e:
        signal.alarm(0)  # Make sure to cancel alarm
        return [{"info": f"No counterfactuals found (error: {str(e)[:100]})"}]
    
    finally:
        signal.alarm(0)  # Ensure alarm is always cancelled

# Process each instance with progress tracking
with tqdm(total=len(prediction_ids), desc="Counterfactual explanations") as pbar:
    for i, pred_id in enumerate(prediction_ids):
        instance_row = df_test.iloc[i:i+1].copy()
        
        # Generate and store counterfactuals
        counterfactuals = process_single_counterfactual(pred_id, instance_row)
        results["predictions"][pred_id]["counterfactuals"] = counterfactuals
        
        # Update progress
        pbar.update(1)
        pbar.set_postfix({"Instance": f"{i+1}/{len(prediction_ids)}"})


# Save results
if save_outputs:
    logger.info(f"💾 Saving results to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"✅ Successfully saved {len(results['predictions'])} predictions with explanations")
    
    # Print summary
    successful_cf = sum(1 for pred in results['predictions'].values() 
                        if pred['counterfactuals'] and not any('error' in str(cf) for cf in pred['counterfactuals']))
    successful_shap = sum(1 for pred in results['predictions'].values() 
                            if pred['shap_explanation'] and 'error' not in pred['shap_explanation'])
    
    logger.info(f"📊 Summary:")
    logger.info(f"  - Total predictions: {len(results['predictions'])}")
    logger.info(f"  - Successful SHAP explanations: {successful_shap}")
    logger.info(f"  - Successful Counterfactual explanations: {successful_cf}")
    
logger.info("🏁 Batch prediction and explanation generation complete!")