#!/usr/bin/env python3
"""
Generate LLM explanations for credit decisions.
"""

import sys
import json
import time
import logging
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

# Configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

# Load configuration
config_dir = project_root / "config"
sys.path.insert(0, str(config_dir.parent))
from config.config import settings

# Load custom packages
from xai_pkg.llm_integration.decision_explainer import CreditDecisionExplainer
from xai_pkg.model_explainer.utils import format_system_variables
from xai_pkg.storage.dynamodb_client import DynamoDBClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress HTTP request logging from httpx (used by Ollama and OpenAI)
logging.getLogger("httpx").setLevel(logging.WARNING)

def sanitize_for_json(obj):
    """
    Recursively sanitize data to ensure JSON serialization compatibility.
    Handles NaN, Infinity, and other problematic values similar to DynamoDB client.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"  # Convert NaN to string
        elif math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"  # Convert Infinity to string
        else:
            return obj
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    else:
        return obj

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LLM explanations for credit decisions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of predictions to process")
    parser.add_argument("--no-dynamodb", action="store_true", help="Disable DynamoDB storage for this run")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing explanations in DynamoDB")
    args = parser.parse_args()
    
    # Initialize explainer
    credit_explainer = CreditDecisionExplainer()
    
    # Initialize DynamoDB client if enabled
    dynamo_client = None
    if settings.get("DYNAMODB", {}).get("enabled", False) and not args.no_dynamodb:
        try:
            config = settings["DYNAMODB"]
            dynamo_client = DynamoDBClient(region_name=config.get("region", "us-east-1"))
            
            # Create tables
            dynamo_client.create_tables(
                explanation_table=config.get("explanation_table", "llm_explanations"),
                evaluation_table=config.get("evaluation_table", "evaluation_results")
            )
            
            logger.info("✅ DynamoDB enabled")
        except Exception as e:
            logger.error(f"❌ Failed to setup DynamoDB: {e}")
            dynamo_client = None
    else:
        if args.no_dynamodb:
            logger.info("⚠️  DynamoDB disabled by --no-dynamodb flag")
        else:
            logger.info("⚠️  DynamoDB disabled")
    
    # Load predictions from the sampled file
    predictions_file = str(Path(__file__).parent.parent / "output" / "predictions" / "prediction_results_sampled.json")
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)
    
    # Extract predictions
    if 'predictions' in predictions_data:
        predictions = predictions_data['predictions']
        prediction_items = list(predictions.items())
    else:
        # Fallback for different structures
        prediction_items = list(predictions_data.items())
    
    # Filter out positive predictions (approvals) if requested - BEFORE applying any limits
    if settings.get("EXCLUDE_POSITIVE_PREDICTIONS", False):
        original_count = len(prediction_items)
        prediction_items = [item for item in prediction_items if item[1]['prediction']['prediction'] == 1]
        logger.info(f"🔍 Filtered from {original_count} to {len(prediction_items)} predictions (defaults only)")
    
    # Apply command line limit 
    if args.limit is not None:
        prediction_items = prediction_items[:args.limit]
        logger.info(f"🔢 Limited to first {len(prediction_items)} predictions due to --limit flag")
    else:
        # Limit predictions for testing based on settings
        max_samples = settings.get("NUM_EXPLANATION_SAMPLES", len(prediction_items))
        if max_samples < len(prediction_items):
            prediction_items = prediction_items[:max_samples]
    
    logger.info(f"📊 Processing {len(prediction_items)} predictions from {predictions_file}")
    
    # Clear DynamoDB table at start if overwrite flag is set
    if dynamo_client and args.overwrite:
        try:
            table_name = settings["DYNAMODB"]["explanation_table"]
            dynamo_client.clear_table(table_name)
            logger.info(f"🗑️ Cleared DynamoDB table {table_name} for fresh start (--overwrite flag)")
        except Exception as e:
            logger.error(f"❌ Failed to clear DynamoDB table: {e}")
    
    # Output file path
    output_filepath = str(Path(__file__).parent.parent / "output" / "llm_explanations" / "explanations.json")
    
    # Load model metadata from separate file for system variables
    metadata_path = settings["METADATA_PATH"]
    with open(metadata_path, 'r') as f:
        model_metadata = json.load(f)
    
    # Prepare system variables (once for all predictions)
    system_vars = format_system_variables(model_metadata)
    
    if not prediction_items:
        logger.error("No predictions found")
        return
    
    # Process predictions with optimized concurrent processing and caching
    start_time = time.time()
    total_predictions = len(prediction_items)
    enabled_llms = [name for name, llm in credit_explainer.llms.items()]
    
    # Calculate total work
    total_possible = total_predictions * len(enabled_llms) * settings["NUM_REGENERATIONS"]
    
    # Simple progress tracking
    progress_stats = {
        'successful_explanations': 0,
        'failed_explanations': 0,
        'llm_stats': defaultdict(lambda: {'success': 0, 'failed': 0})
    }
    
    logger.info(f"Starting explanation generation:")
    logger.info(f"{total_predictions} predictions × {len(enabled_llms)} LLMs × {settings['NUM_REGENERATIONS']} regenerations = {total_possible} total")
    logger.info(f"Concurrent processing: {settings['USE_CONCURRENT_PROCESSING']} (workers: {settings['MAX_CONCURRENT_WORKERS']})")
    logger.info(f"LLMs: {enabled_llms}")
    
    # Check for existing explanations if resume mode enabled (i.e., not overwriting)
    existing_explanations = set()
    if dynamo_client and not args.overwrite:
        try:
            table_name = settings["DYNAMODB"]["explanation_table"]
            logger.info("🔍 Checking for existing explanations to enable resume...")
            
            # Scan DynamoDB to get all existing explanation_ids
            table = dynamo_client.dynamodb.Table(table_name)
            response = table.scan(ProjectionExpression="explanation_id")
            items = response.get('Items', [])
            
            # Continue scanning if there are more items
            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ProjectionExpression="explanation_id",
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            existing_explanations = {item['explanation_id'] for item in items}
            logger.info(f"📋 Found {len(existing_explanations)} existing explanations in DynamoDB")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to check existing explanations: {e}")
            logger.info("🔄 Proceeding without resume (will regenerate all)")

    # Collect all explanation tasks (filtering out existing ones if resume enabled)
    all_tasks = []
    skipped_tasks = 0
    for prediction_id, prediction in prediction_items:
        for llm_name in enabled_llms:
            for regeneration_num in range(1, settings["NUM_REGENERATIONS"] + 1):
                # Generate explanation_id using same logic as generate_single_explanation
                explanation_id = f"{prediction_id}_exp_{regeneration_num}_{llm_name}"
                
                # Skip if explanation already exists and resume mode is enabled
                if explanation_id in existing_explanations:
                    skipped_tasks += 1
                    continue
                    
                all_tasks.append((prediction_id, prediction, llm_name, regeneration_num))
    
    if existing_explanations:
        logger.info(f"⏭️ Resume mode: Skipping {skipped_tasks} existing explanations")
        logger.info(f"🎯 Will generate {len(all_tasks)} missing explanations")
    
    # Execute with progress tracking
    explanations = []
    
    if settings["USE_CONCURRENT_PROCESSING"]:
        # Concurrent execution
        with ThreadPoolExecutor(max_workers=settings["MAX_CONCURRENT_WORKERS"]) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    credit_explainer.generate_single_explanation,
                    pred_id, pred, llm, regen, system_vars,
                    dynamo_client, settings.get("DYNAMODB", {})
                ): (pred_id, llm, regen)
                for pred_id, pred, llm, regen in all_tasks
            }
            
            # Process completed tasks with simple progress
            print(f"Generating {len(all_tasks)} explanations...")
            for i, future in enumerate(as_completed(future_to_task), 1):
                pred_id, llm_name, regen_num = future_to_task[future]
                
                try:
                    result = future.result()
                    if result.get('error'):
                        progress_stats['failed_explanations'] += 1
                        progress_stats['llm_stats'][llm_name]['failed'] += 1
                    else:
                        progress_stats['successful_explanations'] += 1
                        progress_stats['llm_stats'][llm_name]['success'] += 1
                    explanations.append(result)
                    
                except Exception as e:
                    logger.error(f"Task execution failed for {pred_id} {llm_name}: {e}")
                    progress_stats['failed_explanations'] += 1
                
                # In-place progress update
                print(f"\rProgress: {i}/{len(all_tasks)} explanations completed", end="", flush=True)
            print()  # Final newline when done
    
    else:
        # Sequential execution
        with tqdm(total=len(all_tasks), desc="Generating explanations", unit="exp") as pbar:
            for pred_id, pred, llm_name, regen_num in all_tasks:
                result = credit_explainer.generate_single_explanation(
                    pred_id, pred, llm_name, regen_num, system_vars,
                    dynamo_client, settings.get("DYNAMODB", {})
                )
                
                if result.get('error'):
                    progress_stats['failed_explanations'] += 1
                    progress_stats['llm_stats'][llm_name]['failed'] += 1
                else:
                    progress_stats['successful_explanations'] += 1
                    progress_stats['llm_stats'][llm_name]['success'] += 1
                explanations.append(result)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': progress_stats['successful_explanations'],
                    'Failed': progress_stats['failed_explanations']
                })
    
    # Fresh generation for final run
    
    # Save results
    final_data = {
        "explanations": sanitize_for_json(explanations),
        "summary": {
            "total_explanations": len(explanations),
            "successful_explanations": progress_stats['successful_explanations'],
            "failed_explanations": progress_stats['failed_explanations'],
            "llm_stats": dict(progress_stats['llm_stats']),
            "timestamp": datetime.now().isoformat(),
            "settings_used": {
                "NUM_REGENERATIONS": settings["NUM_REGENERATIONS"],
                "USE_CONCURRENT_PROCESSING": settings["USE_CONCURRENT_PROCESSING"],
                "MAX_CONCURRENT_WORKERS": settings["MAX_CONCURRENT_WORKERS"]
            }
        }
    }
    
    try:
        with open(output_filepath, 'w') as f:
            json.dump(final_data, f, indent=2)
    except Exception as e:
        logger.error(f"❌ JSON export failed: {e}")
        logger.error(f"   Data successfully saved to DynamoDB but JSON export failed")
    
    # Print summary
    total_time = time.time() - start_time
    logger.info(f"\n🎯 EXPLANATION GENERATION SUMMARY:")
    logger.info(f"   📊 Total explanations: {len(explanations)}")
    logger.info(f"   ✅ Successful: {progress_stats['successful_explanations']}")
    logger.info(f"   ❌ Failed: {progress_stats['failed_explanations']}")
    logger.info(f"   ⏱️ Total time: {total_time:.1f}s")
    logger.info(f"   💾 Results saved to: {output_filepath}")
    if dynamo_client:
        logger.info(f"   🗄️  DynamoDB: {settings['DYNAMODB']['explanation_table']} (stored immediately per explanation)")
    
    # Per-LLM statistics
    logger.info(f"\n📈 PER-LLM STATISTICS:")
    for llm_name, stats in progress_stats['llm_stats'].items():
        logger.info(f"   {llm_name}: {stats['success']} successful, {stats['failed']} failed")

if __name__ == "__main__":
    main()