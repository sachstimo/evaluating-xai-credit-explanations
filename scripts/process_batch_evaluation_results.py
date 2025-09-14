#!/usr/bin/env python3
"""
Process Google Cloud batch evaluation results and complete the evaluation pipeline.

This script:
1. Takes raw batch results (containing CEMAT, regulatory compliance, counterfactual from LLM judge)
2. Loads explanations from JSON/DynamoDB
3. Merges batch results with explanations (NO additional LLM calls)
4. Computes remaining metrics (flesch-kincaid, fidelity, semscore, technical/immutable checks)
5. Stores complete evaluation results to DynamoDB

This is the post-batch processing pipeline that avoids redundant LLM calls.
"""

import sys
import os
from pathlib import Path

# Add src to path if package not installed
try:
    import xai_pkg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

# Add config path
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config')
sys.path.insert(0, os.path.dirname(config_dir))

import json
import logging
import argparse
import time
import joblib
from datetime import datetime
from typing import Dict, Any, List, Optional

from xai_pkg.evaluator.response_evaluator import ExplanationEvaluator
from xai_pkg.storage.dynamodb_client import DynamoDBClient
from xai_pkg.model_explainer.predictor import CreditPredictor
from config.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class BatchResultsProcessor:
    """Process batch evaluation results and compute remaining metrics."""
    
    def __init__(self, predictor=None):
        self.predictor = predictor
        # Use the existing ExplanationEvaluator instead of duplicating logic
        self.evaluator = ExplanationEvaluator(predictor=predictor, enabled_metrics=[
            'flesch_kincaid', 'fidelity', 'immutable_features_check', 'technical_features_check'
        ])
    
    def process_batch_results(
        self,
        batch_results_file: Path,
        explanations_file: Path,
        predictions_file: Path,
        dynamo_client=None,
        dynamo_settings: Dict[str, Any] = None,
        limit: int = None
    ) -> Dict[str, Any]:
        """Process batch results and complete evaluation pipeline."""
        
        # Load batch results
        logger.info(f"📂 Loading batch results from: {batch_results_file}")
        with open(batch_results_file, 'r') as f:
            batch_data = json.load(f)
        
        # Extract results (handle different structures)
        if 'results' in batch_data:
            batch_results = batch_data['results']
        elif isinstance(batch_data, list):
            batch_results = batch_data
        else:
            batch_results = [batch_data]
        
        # Load explanations
        logger.info(f"📂 Loading explanations from: {explanations_file}")
        with open(explanations_file, 'r') as f:
            explanations_data = json.load(f)
        
        explanations = explanations_data.get('explanations', explanations_data)
        if limit:
            explanations = explanations[:limit]
        
        # Load predictions
        logger.info(f"📂 Loading predictions from: {predictions_file}")
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
            predictions = predictions_data['predictions']
            pred_lookup = {pred_data['prediction']['prediction_id']: pred_data 
                         for pred_data in predictions.values()}
        else:
            pred_lookup = {p['prediction_id']: p for p in predictions_data}
        
        # Create batch results lookup by explanation_id
        batch_lookup = {}
        for result in batch_results:
            metadata = result.get('metadata', {}) or result.get('custom_metadata', {})
            explanation_id = metadata.get('explanation_id')
            if explanation_id:
                batch_lookup[explanation_id] = result.get('evaluation_result', {})
        
        logger.info(f"🔄 Processing {len(explanations)} explanations with batch results")
        
        # Track statistics
        summary_stats = {
            'total_explanations': len(explanations),
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics_computed': ['flesch_kincaid', 'fidelity', 'cemat', 'regulatory_compliance', 
                               'counterfactual_verification', 'immutable_features_check', 'technical_features_check'],
            'results_by_llm': {},
            'results_by_metric': {},
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'batch_results_merged': 0
        }
        
        # Process each explanation
        for i, explanation in enumerate(explanations, 1):
            print(f"\rProgress: {i}/{len(explanations)}", end="", flush=True)
            
            pred_id = explanation.get('prediction_id')
            llm_name = explanation.get('llm_name', 'unknown')
            explanation_id = explanation.get('explanation_id')
            
            if pred_id not in pred_lookup:
                summary_stats['failed_evaluations'] += 1
                continue
            
            prediction = pred_lookup[pred_id]
            
            # Initialize evaluation results
            explanation['evaluation_results'] = {}
            explanation['evaluation_metadata'] = {
                'evaluated_at': datetime.now().isoformat(),
                'evaluator_version': '1.0'
            }
            
            # Add prediction outcome for semscore analysis
            if 'prediction' in prediction:
                explanation['prediction_outcome'] = 'approved' if prediction['prediction']['prediction'] == 0 else 'declined'
            else:
                explanation['prediction_outcome'] = 'unknown'
            
            try:
                # Merge batch results (CEMAT, regulatory compliance, counterfactual)
                if explanation_id in batch_lookup:
                    batch_result = batch_lookup[explanation_id]
                    
                    # Add CEMAT evaluation
                    if 'cemat_evaluation' in batch_result:
                        explanation['evaluation_results']['cemat'] = self._format_cemat_result(batch_result['cemat_evaluation'])
                    
                    # Add regulatory compliance  
                    if 'regulatory_compliance' in batch_result:
                        explanation['evaluation_results']['regulatory_compliance'] = self._format_regulatory_result(batch_result['regulatory_compliance'])
                    
                    # Add counterfactual extraction results and perform verification
                    if 'counterfactual_extraction' in batch_result:
                        # Store the extracted counterfactual changes
                        explanation['llm_response'] = {'counterfactual_extraction': batch_result['counterfactual_extraction']}
                        
                        # Perform counterfactual verification directly using CounterfactualVerifier (no LLM calls)
                        if self.evaluator.cf_verifier:
                            verification_result = self.evaluator.cf_verifier.verify_llm_counterfactual(explanation, prediction)
                            explanation['evaluation_results']['counterfactual_verification'] = verification_result
                        else:
                            explanation['evaluation_results']['counterfactual_verification'] = {
                                'score': None,
                                'error': 'No predictor available for counterfactual verification'
                            }
                    
                    summary_stats['batch_results_merged'] += 1
                
                # Compute remaining metrics using the existing evaluator (no LLM calls)
                explanation['evaluation_results']['flesch_kincaid'] = self.evaluator._evaluate_flesch_kincaid_readability(explanation)
                explanation['evaluation_results']['fidelity'] = self.evaluator._evaluate_fidelity(explanation, prediction)
                explanation['evaluation_results']['immutable_features_check'] = self.evaluator._check_immutable_features(explanation)
                explanation['evaluation_results']['technical_features_check'] = self.evaluator._check_technical_feature_names(explanation)
                
                # Update statistics
                self._update_statistics(explanation, summary_stats, llm_name)
                summary_stats['successful_evaluations'] += 1
                
                # Store to DynamoDB
                if dynamo_client and dynamo_settings:
                    self._store_evaluation_result(explanation, dynamo_client, dynamo_settings)
                
            except Exception as e:
                logger.error(f"Failed to process {explanation_id}: {e}")
                explanation['evaluation_results'] = {'error': str(e)}
                summary_stats['failed_evaluations'] += 1
        
        print()
        
        # Calculate semscore analysis using the existing evaluator
        logger.info("🔍 Computing semantic similarity analysis")
        semscore_analysis = self.evaluator.semscore_evaluator.evaluate_explanation_similarity(explanations)
        summary_stats['semscore_analysis'] = semscore_analysis
        
        # Calculate summary statistics
        self._finalize_statistics(summary_stats)
        
        logger.info(f"✅ Processed {summary_stats['successful_evaluations']}/{len(explanations)} explanations")
        logger.info(f"📊 Merged {summary_stats['batch_results_merged']} batch results")
        
        return summary_stats, explanations
    
    def _format_cemat_result(self, cemat_eval: Dict[str, Any]) -> Dict[str, Any]:
        """Format CEMAT evaluation result from batch processing."""
        if not cemat_eval:
            return {'error': 'CEMAT evaluation failed'}
        
        # Calculate scores as done in response_evaluator.py
        understandability_items = cemat_eval.get('understandability_items', {})
        understandability_earned = sum(1 for score in understandability_items.values() if score == 1)
        understandability_applicable = len([score for score in understandability_items.values() 
                                     if score is not None and score not in ["N/A", "n/a"]])
        
        actionability_items = cemat_eval.get('actionability_items', {})
        actionability_earned = sum(1 for score in actionability_items.values() if score == 1)
        actionability_applicable = len([score for score in actionability_items.values() 
                                 if score is not None and score not in ["N/A", "n/a"]])
        
        # Calculate overall score using the same formula as response_evaluator.py
        total_earned = actionability_earned + understandability_earned
        total_applicable = actionability_applicable + understandability_applicable
        overall_cemat = total_earned / max(1, total_applicable)
        
        # Calculate percentage scores
        understandability_percentage = understandability_earned / max(1, understandability_applicable)
        actionability_percentage = actionability_earned / max(1, actionability_applicable)
        
        return {
            'overall_score': float(overall_cemat),
            'understandability_items': cemat_eval.get('understandability_items'),
            'actionability_items': cemat_eval.get('actionability_items'), 
            'understandability_score': float(understandability_percentage),
            'understandability_applicable': understandability_applicable,
            'actionability_score': float(actionability_percentage),
            'actionability_applicable': actionability_applicable
        }
    
    def _format_regulatory_result(self, reg_eval: Dict[str, Any]) -> Dict[str, Any]:
        """Format regulatory compliance result from batch processing."""
        if not reg_eval:
            return {'error': 'Regulatory compliance evaluation failed'}
        
        return {
            'principal_reason_identification': {'score': reg_eval.get('principal_reason_identification', {}).get('score', 0)},
            'individual_specific_content': {'score': reg_eval.get('individual_specific_content', {}).get('score', 0)},
            'technical_jargon_check': {'score': reg_eval.get('technical_jargon_check', {}).get('score', 0)},
            'feature_name_matching': {'score': reg_eval.get('feature_name_matching', {}).get('score', 0)}
        }
    
    
    
    def _update_statistics(self, explanation: Dict[str, Any], summary_stats: Dict[str, Any], llm_name: str):
        """Update summary statistics."""
        for metric_name, result in explanation.get('evaluation_results', {}).items():
            # Update by LLM
            if llm_name not in summary_stats['results_by_llm']:
                summary_stats['results_by_llm'][llm_name] = {}
            if metric_name not in summary_stats['results_by_llm'][llm_name]:
                summary_stats['results_by_llm'][llm_name][metric_name] = []
            
            # Update by metric
            if metric_name not in summary_stats['results_by_metric']:
                summary_stats['results_by_metric'][metric_name] = []
            
            if isinstance(result, dict) and 'score' in result and result['score'] is not None:
                summary_stats['results_by_llm'][llm_name][metric_name].append(result['score'])
                summary_stats['results_by_metric'][metric_name].append(result['score'])
    
    def _finalize_statistics(self, summary_stats: Dict[str, Any]):
        """Calculate final averages and standard deviations."""
        for llm_name, metrics in summary_stats['results_by_llm'].items():
            for metric_name, scores in metrics.items():
                if scores:
                    summary_stats['results_by_llm'][llm_name][metric_name] = {
                        'mean': float(sum(scores) / len(scores)),
                        'std': float((sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5),
                        'count': len(scores)
                    }
        
        for metric_name, scores in summary_stats['results_by_metric'].items():
            if scores:
                summary_stats['results_by_metric'][metric_name] = {
                    'mean': float(sum(scores) / len(scores)),
                    'std': float((sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5),
                    'count': len(scores)
                }
    
    def _store_evaluation_result(self, explanation: Dict[str, Any], dynamo_client, dynamo_settings: Dict[str, Any]):
        """Store evaluation result to DynamoDB."""
        try:
            table_name = dynamo_settings.get("evaluation_table")
            if table_name:
                dynamo_client.store(explanation, table_name, dynamo_settings.get("overwrite", True))
                logger.debug(f"✅ Stored evaluation for {explanation.get('explanation_id', 'unknown')} to DynamoDB")
        except Exception as e:
            logger.warning(f"⚠️ Failed to store evaluation to DynamoDB: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process batch evaluation results and upload to DynamoDB")
    parser.add_argument("batch_results", help="Path to raw batch results JSON file from Google Cloud")
    parser.add_argument("--explanations", help="Path to explanations JSON file", 
                       default="output/llm_explanations/explanations.json")
    parser.add_argument("--predictions", help="Path to predictions JSON file",
                       default="output/predictions/prediction_results_sampled.json")
    parser.add_argument("--limit", type=int, help="Limit number of explanations to process")
    parser.add_argument("--dynamo", action="store_true", help="Enable DynamoDB upload (disabled by default for thesis documentation)")
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    batch_results_file = Path(args.batch_results) if args.batch_results else ""
    explanations_file = project_root / args.explanations
    predictions_file = project_root / args.predictions
    model_path = project_root / "output" / "models" / "best_model.pkl"
    
    if not batch_results_file.exists():
        logger.error(f"❌ Batch results file not found: {batch_results_file}")
        sys.exit(1)
    
    if not explanations_file.exists():
        logger.error(f"❌ Explanations file not found: {explanations_file}")
        sys.exit(1)
    
    if not predictions_file.exists():
        logger.error(f"❌ Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    # Load predictor
    predictor = None
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            model_metadata_path = model_path.parent / "model_metadata.json"
            model_metadata = json.load(open(model_metadata_path)) if model_metadata_path.exists() else {}
            predictor = CreditPredictor(model, model_metadata)
            logger.info("✅ Predictor loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load predictor: {e} - fidelity evaluation will be limited")
    
    # Setup DynamoDB (enabled by default)
    dynamo_client = None
    dynamo_settings = None
    if args.dynamo and settings.get("DYNAMODB", {}).get("enabled", False):
        try:
            dynamo_settings = settings["DYNAMODB"]
            dynamo_client = DynamoDBClient(region_name=dynamo_settings.get("region", "us-east-1"))
            
            # Delete and recreate evaluation_results table with new structure
            try:
                dynamo_client.dynamodb.Table("evaluation_results").delete()
                logger.info("🗑️ Deleted old evaluation_results table")
                # Wait for table deletion to complete
                import time as time_module
                time_module.sleep(10)
            except:
                pass  # Table might not exist
            
            # Create tables if needed
            dynamo_client.create_tables(
                explanation_table=dynamo_settings.get("explanation_table", "llm_explanations"),
                evaluation_table=dynamo_settings.get("evaluation_table", "evaluation_results")
            )
            
            logger.info("✅ DynamoDB enabled for storage")
        except Exception as e:
            logger.error(f"❌ Failed to setup DynamoDB: {e}")
            dynamo_client = None
            dynamo_settings = None
    else:
        logger.info("ℹ️ DynamoDB storage disabled (use --dynamo to enable)")
    
    # Process batch results
    processor = BatchResultsProcessor(predictor=predictor)
    
    logger.info("🚀 Starting batch results processing")
    start_time = time.time()
    
    # Load explanations for saving to JSON
    with open(explanations_file, 'r') as f:
        explanations_data = json.load(f)
    explanations = explanations_data.get('explanations', explanations_data)
    if args.limit:
        explanations = explanations[:args.limit]
    
    summary_stats, processed_explanations = processor.process_batch_results(
        batch_results_file=batch_results_file,
        explanations_file=explanations_file,
        predictions_file=predictions_file,
        dynamo_client=dynamo_client,
        dynamo_settings=dynamo_settings,
        limit=args.limit
    )
    
    total_time = time.time() - start_time
    logger.info(f"⏱️ Total processing time: {total_time:.2f}s")
    
    # Save complete results to evaluations.json
    evaluations_output = {
        "evaluation_summary": summary_stats,
        "explanations": processed_explanations
    }
    
    evaluations_file = project_root / "output" / "evaluations" / "evaluations.json"
    evaluations_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(evaluations_file, 'w') as f:
        json.dump(evaluations_output, f, indent=2, default=str)
    
    logger.info(f"📁 Complete evaluation results saved to: {evaluations_file}")
    
    # Print summary
    logger.info("📊 SUMMARY")
    logger.info(f"✅ Successfully processed: {summary_stats['successful_evaluations']}")
    logger.info(f"❌ Failed: {summary_stats['failed_evaluations']}")
    logger.info(f"📈 Batch results merged: {summary_stats['batch_results_merged']}")
    logger.info(f"📊 Total: {summary_stats['total_explanations']}")
    
    # Print semscore summary if available
    if 'semscore_analysis' in summary_stats:
        semscore = summary_stats['semscore_analysis']
        if 'error' not in semscore:
            logger.info("🔍 SEMSCORE ANALYSIS:")
            if 'intra_cluster_similarity' in semscore and semscore['intra_cluster_similarity']['mean'] is not None:
                logger.info(f"  Intra-cluster: {semscore['intra_cluster_similarity']['mean']:.4f} ± {semscore['intra_cluster_similarity']['std']:.4f} (n={semscore['intra_cluster_similarity']['count']})")
            if 'inter_cluster_similarity' in semscore and semscore['inter_cluster_similarity']['mean'] is not None:
                logger.info(f"  Inter-cluster: {semscore['inter_cluster_similarity']['mean']:.4f} ± {semscore['inter_cluster_similarity']['std']:.4f} (n={semscore['inter_cluster_similarity']['count']})")
            if 'consistency_score' in semscore and semscore['consistency_score'] is not None:
                logger.info(f"  Consistency score: {semscore['consistency_score']:.4f}")
        else:
            logger.info(f"⚠️ SemScore analysis failed: {semscore['error']}")
    
    if dynamo_client:
        logger.info("✅ All results uploaded to DynamoDB")
    else:
        logger.info("ℹ️ Results not uploaded (DynamoDB disabled)")

if __name__ == "__main__":
    main()