#!/usr/bin/env python3
"""
Simplified evaluation script for LLM explanations.
Evaluates explanations using the simplified evaluation framework.
"""

import sys
import os
from pathlib import Path

# Add src to path if package not installed (backwards compatibility)
try:
    import xai_pkg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

# Add config path
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config')
sys.path.insert(0, os.path.dirname(config_dir))

import logging
import argparse
import time
import joblib
from xai_pkg.evaluator.response_evaluator import ExplanationEvaluator
from xai_pkg.storage.dynamodb_client import DynamoDBClient
from xai_pkg.model_explainer.predictor import CreditPredictor
from config.config import settings
import json

# Configure logging (concise)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP and LLM logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("xai_pkg.evaluator.judgeLLM").setLevel(logging.WARNING)

def main():
    """Run evaluation using the simplified framework."""
    parser = argparse.ArgumentParser(description="Evaluate LLM explanations")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of explanations to evaluate")
    parser.add_argument("--metrics", nargs="+", help="Specific metrics to run (e.g., --metrics flesch_kincaid fidelity)")
    parser.add_argument("--resume", action="store_true", help="Skip already evaluated explanations")
    parser.add_argument("--no-dynamo", action="store_true", help="Disable DynamoDB integration")
    parser.add_argument("--judge-model", type=str, default="gemini-2.5-flash", help="Judge model to use for evaluation (default: gemini-2.5-flash)")
    parser.add_argument("--dynamo-upload", action="store_true", help="Enable DynamoDB storage after evaluation (disabled by default)")
    args = parser.parse_args()
    
    # File paths
    explanations_file = Path(__file__).parent.parent / "output" / "llm_explanations" / "explanations.json"
    predictions_file = Path(__file__).parent.parent / "output" / "predictions" / "prediction_results_sampled.json"
    model_path = Path(__file__).parent.parent / "output" / "models" / "best_model.pkl"
    output_file = Path(__file__).parent.parent / "output" / "evaluations" / "evaluations.json"
    
    # Load predictor if available
    predictor = None
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            model_metadata_path = model_path.parent / "model_metadata.json"
            model_metadata = json.load(open(model_metadata_path)) if model_metadata_path.exists() else {}
            predictor = CreditPredictor(model, model_metadata)
            logger.info("✅ Predictor loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load predictor: {e} - counterfactual verification will be skipped")
    
    # Create evaluator
    evaluator = ExplanationEvaluator(predictor=predictor, enabled_metrics=args.metrics, judge_model=args.judge_model)
    
    # Setup DynamoDB if enabled (only for upload if --dynamo-upload flag is set)
    dynamo_client = None
    dynamo_settings = None
    if not args.no_dynamo and args.dynamo_upload and settings.get("DYNAMODB", {}).get("enabled", False):
        try:
            dynamo_settings = settings["DYNAMODB"]
            dynamo_client = DynamoDBClient(region_name=dynamo_settings.get("region", "us-east-1"))
            
            # Create tables if needed
            dynamo_client.create_tables(
                explanation_table=dynamo_settings.get("explanation_table", "llm_explanations"),
                evaluation_table=dynamo_settings.get("evaluation_table", "evaluation_results")
            )
            
            logger.info("✅ DynamoDB enabled for evaluation")
        except Exception as e:
            logger.error(f"❌ Failed to setup DynamoDB: {e}")
            dynamo_client = None
            dynamo_settings = None
    elif args.dynamo_upload:
        logger.info("⚠️  DynamoDB upload requested but disabled in config - using JSON files only")
    else:
        logger.info("ℹ️  DynamoDB upload disabled - using JSON files only")
    
    # Run evaluation
    try:
        logger.info("🚀 Starting LLM explanation evaluation")
        start_time = time.time()
        
        summary_stats = evaluator.evaluate_all(
            explanations_file=explanations_file,
            predictions_file=predictions_file,
            output_file=output_file,
            limit=args.limit,
            resume=args.resume,
            dynamo_client=dynamo_client,
            dynamo_settings=dynamo_settings
        )
        
        total_time = time.time() - start_time
        logger.info(f"⏱️  Total evaluation time: {total_time:.2f}s")
        
        # Print summary
        evaluator._print_evaluation_summary(summary_stats, output_file)
        logger.info("✅ Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()