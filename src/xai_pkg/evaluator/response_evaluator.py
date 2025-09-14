"""
Simplified evaluation framework for LLM explanations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from .counterfactual_verification import CounterfactualVerifier  
from .semscore_eval import SemScoreEvaluator
from .judgeLLM import JudgeLLM

logger = logging.getLogger(__name__)

class ExplanationEvaluator:
    """Simplified evaluator that handles all explanation evaluation metrics."""
    
    def __init__(self, predictor=None, enabled_metrics=None, judge_model=None):
        self.predictor = predictor
        self.cf_verifier = CounterfactualVerifier(self.predictor) if predictor else None
        self.semscore_evaluator = SemScoreEvaluator()
        self._judge_llm = JudgeLLM(judge_name=judge_model)
        self._judge_results_cache = {}
        
        # Define available metrics
        self._available_metrics = {
            'flesch_kincaid': self._evaluate_flesch_kincaid_readability,
            'fidelity': self._evaluate_fidelity,
            'counterfactual_verification': self._verify_llm_counterfactual if self.cf_verifier else None,
            'cemat': self.get_cemat_results,
            'regulatory_compliance': self.get_regulatory_evaluation,
            'immutable_features_check': self._check_immutable_features,
            'technical_features_check': self._check_technical_feature_names
        }
        
        # Filter enabled metrics
        self.metrics = {k: v for k, v in self._available_metrics.items() 
                       if v is not None and (enabled_metrics is None or k in enabled_metrics)}

    def evaluate_all(
        self,
        explanations_file: Path,
        predictions_file: Path,
        output_file: Path | None = None,
        limit: int | None = None,
        resume: bool = False,
        dynamo_client=None,
        dynamo_settings: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run all evaluations and save results."""
        
        # Load and prepare data
        explanations, pred_lookup = self._load_and_prepare_data(explanations_file, predictions_file, limit, resume)
        
        # Set output file
        if output_file is None:
            output_file = explanations_file.parent / f"evaluation/{explanations_file.stem}_evaluated.json"
        
        # Initialize statistics
        summary_stats = self._initialize_statistics(explanations)
        
        # Check batch results availability
        if self._check_batch_results_available():
            logger.info("✅ Batch judge results found - using pre-computed results")
        else:
            logger.warning("⚠️  No batch judge results found - evaluation may be slow")
        
        # Evaluate explanations
        print(f"Evaluating {len(explanations)} explanations...")
        for i, explanation in enumerate(explanations, 1):
            print(f"\rProgress: {i}/{len(explanations)}", end="", flush=True)
            
            pred_id = explanation.get('prediction_id')
            if pred_id not in pred_lookup:
                summary_stats['failed_evaluations'] += 1
                continue
            
            prediction = pred_lookup[pred_id]
            self._evaluate_single_explanation(explanation, prediction, summary_stats, resume, dynamo_client, dynamo_settings)
        
        print()
        
        # Finalize results
        self._finalize_statistics(summary_stats, explanations)
        
        # Save results
        enhanced_data = {
            'evaluation_summary': summary_stats,
            'explanations': explanations
        }
        
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        return summary_stats

    def _print_evaluation_summary(self, summary_stats: Dict[str, Any], output_file: Path):
        """Print formatted evaluation summary."""
        logger.info("SUMMARY")
        logger.info(f"ok: {summary_stats['successful_evaluations']}  fail: {summary_stats['failed_evaluations']}  total: {summary_stats['total_explanations']}")
        
        # Results by LLM
        if summary_stats['results_by_llm']:
            logger.info("Results by LLM:")
            for llm_name, metrics in summary_stats['results_by_llm'].items():
                logger.info(f"  {llm_name}:")
                for metric_name, stats in metrics.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        logger.info(f"    {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        # Results by metric
        if summary_stats['results_by_metric']:
            logger.info("Results by Metric:")
            for metric_name, stats in summary_stats['results_by_metric'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        
        # SemScore analysis
        if 'semscore_analysis' in summary_stats:
            semscore = summary_stats['semscore_analysis']
            if 'random_baseline' in semscore:
                logger.info("Semantic Similarity Analysis:")
                logger.info(f"  Random baseline: {semscore['random_baseline']['mean']:.4f} ± {semscore['random_baseline']['std']:.4f}")
                
                if 'intra_cluster_similarity' in semscore and semscore['intra_cluster_similarity']['mean'] is not None:
                    logger.info(f"  Intra-cluster: {semscore['intra_cluster_similarity']['mean']:.4f} ± {semscore['intra_cluster_similarity']['std']:.4f} (n={semscore['intra_cluster_similarity']['count']})")
                else:
                    logger.info("  Intra-cluster: N/A (insufficient data)")
                
                if 'inter_cluster_similarity' in semscore and semscore['inter_cluster_similarity']['mean'] is not None:
                    logger.info(f"  Inter-cluster: {semscore['inter_cluster_similarity']['mean']:.4f} ± {semscore['inter_cluster_similarity']['std']:.4f} (n={semscore['inter_cluster_similarity']['count']})")
                else:
                    logger.info("  Inter-cluster: N/A (insufficient data)")
            elif 'error' in semscore:
                logger.info(f"  SemScore: {semscore['error']}")
        
        logger.info(f"Saved: {output_file}")

    def _load_and_prepare_data(self, explanations_file: Path, predictions_file: Path, 
                              limit: int | None, resume: bool) -> tuple[List, Dict]:
        """Load and prepare explanations and predictions data."""
        # Load data
        with open(explanations_file, 'r') as f:
            explanations_data = json.load(f)
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        # Extract explanations and predictions
        explanations = explanations_data.get('explanations', explanations_data)
        if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
            predictions = predictions_data['predictions']
            pred_lookup = {pred_data['prediction']['prediction_id']: pred_data 
                         for pred_data in predictions.values()}
        else:
            pred_lookup = {p['prediction_id']: p for p in predictions_data}
        
        # Filter out positive predictions if requested
        from config.config import settings
        if settings.get("EXCLUDE_POSITIVE_PREDICTIONS", False):
            original_count = len(explanations)
            explanations = [exp for exp in explanations 
                          if exp.get('prediction_id') in pred_lookup and 
                          pred_lookup[exp.get('prediction_id')].get('prediction', {}).get('prediction', 0) == 1]
            logger.info(f"🔍 Filtered from {original_count} to {len(explanations)} explanations (rejected applications only)")
        
        # Filter already evaluated if resume
        if resume:
            explanations = [exp for exp in explanations if not exp.get('evaluation_results')]
            
        # Apply limit after filtering
        if limit:
            explanations = explanations[:limit]
        
        return explanations, pred_lookup

    def _initialize_statistics(self, explanations: List) -> Dict[str, Any]:
        """Initialize statistics tracking."""
        return {
            'total_explanations': len(explanations),
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics_computed': list(self.metrics.keys()),
            'results_by_llm': {},
            'results_by_metric': {},
            'successful_evaluations': 0,
            'failed_evaluations': 0
        }

    def _evaluate_single_explanation(self, explanation: Dict[str, Any], prediction: Dict[str, Any], 
                                   summary_stats: Dict[str, Any], resume: bool, 
                                   dynamo_client=None, dynamo_settings: Dict[str, Any] = None):
        """Evaluate a single explanation."""
        pred_id = explanation.get('prediction_id')
        llm_name = explanation.get('llm_name', 'unknown')
        
        # Initialize results
        explanation['evaluation_results'] = {}
        explanation['evaluation_metadata'] = {
            'evaluated_at': datetime.now().isoformat(),
            'evaluator_version': '1.0'
        }
        
        try:
            # Run evaluations
            for metric_name, metric_func in self.metrics.items():
                if resume and metric_name in explanation.get('evaluation_results', {}):
                    continue
                
                result = metric_func(explanation, prediction)
                explanation['evaluation_results'][metric_name] = result
                
                # Update statistics
                self._update_statistics(summary_stats, llm_name, metric_name, result)
            
            summary_stats['successful_evaluations'] += 1
            
            # Store to DynamoDB if enabled
            if dynamo_client and dynamo_settings:
                self._store_evaluation_result(explanation, dynamo_client, dynamo_settings)
                
        except Exception as e:
            logger.error(f"Failed to evaluate {pred_id}: {e}")
            explanation['evaluation_results'] = {'error': str(e)}
            summary_stats['failed_evaluations'] += 1

    def _update_statistics(self, summary_stats: Dict[str, Any], llm_name: str, 
                          metric_name: str, result: Dict[str, Any]):
        """Update statistics with evaluation result."""
        # Initialize nested dictionaries if needed
        if llm_name not in summary_stats['results_by_llm']:
            summary_stats['results_by_llm'][llm_name] = {}
        if metric_name not in summary_stats['results_by_llm'][llm_name]:
            summary_stats['results_by_llm'][llm_name][metric_name] = []
        if metric_name not in summary_stats['results_by_metric']:
            summary_stats['results_by_metric'][metric_name] = []
        
        # Track scores and skipped items
        score_to_track = None
        
        if isinstance(result, dict):
            # Handle different score formats
            if 'score' in result and result['score'] is not None:
                score_to_track = result['score']
            elif 'cemat_score' in result and result['cemat_score'] is not None:
                score_to_track = result['cemat_score']
            elif result.get('skipped', False):
                skipped_key = f"{metric_name}_skipped"
                summary_stats[skipped_key] = summary_stats.get(skipped_key, 0) + 1
        
        if score_to_track is not None:
            summary_stats['results_by_llm'][llm_name][metric_name].append(score_to_track)
            summary_stats['results_by_metric'][metric_name].append(score_to_track)

    def _finalize_statistics(self, summary_stats: Dict[str, Any], explanations: List):
        """Calculate final statistics and semscore."""
        # Calculate semscore
        semscore_analysis = self.semscore_evaluator.evaluate_explanation_similarity(explanations)
        summary_stats['semscore_analysis'] = semscore_analysis
        
        # Calculate averages for all metrics
        for results_dict in [summary_stats['results_by_llm'], summary_stats['results_by_metric']]:
            for key, metrics in results_dict.items():
                if isinstance(metrics, dict):
                    for metric_name, scores in metrics.items():
                        if isinstance(scores, list) and scores:
                            mean = sum(scores) / len(scores)
                            variance = sum((x - mean) ** 2 for x in scores) / len(scores)
                            results_dict[key][metric_name] = {
                                'mean': float(mean),
                                'std': float(variance ** 0.5),
                                'count': len(scores)
                            }

    def _get_judge_evaluation(self, explanation: Dict[str, Any], prediction: Dict[str, Any]) -> Optional[Any]:
        """Get evaluation results from JudgeLLM with caching."""
        explanation_id = explanation.get('explanation_id', 'unknown')
        
        if explanation_id in self._judge_results_cache:
            return self._judge_results_cache[explanation_id]
        
        # Try batch results first
        batch_result = self._get_batch_result(explanation_id)
        if batch_result:
            self._judge_results_cache[explanation_id] = batch_result
            return batch_result
        
        # Fallback to single call
        if self._judge_llm is None:
            return None
        
        try:
            result = self._judge_llm.evaluate_llm_explanation(
                explanation['explanation_text'], 
                prediction,
                prediction_id=explanation.get('prediction_id', 'unknown')
            )
            self._judge_results_cache[explanation_id] = result
            return result
        except Exception as e:
            logger.error(f"JudgeLLM evaluation failed for {explanation_id}: {e}")
            return None

    def _get_batch_result(self, explanation_id: str) -> Optional[Any]:
        """Get judge evaluation result from batch results."""
        try:
            # Check local evaluations folder first
            evaluations_folder = Path(__file__).parent.parent.parent / "output" / "evaluations"
            batch_results_file = evaluations_folder / "batch_judge_results.json"
            
            if batch_results_file.exists():
                with open(batch_results_file, 'r') as f:
                    batch_results = json.load(f)
                
                results = batch_results.get('results', {})
                if explanation_id in results:
                    from ..llm_integration.extraction_models import EvaluationResult
                    return EvaluationResult(**results[explanation_id])
            
            # Fallback to Cloud Storage
            return self._get_batch_result_from_cloud(explanation_id)
            
        except Exception as e:
            logger.debug(f"Failed to get batch result for {explanation_id}: {e}")
            return None

    def _get_batch_result_from_cloud(self, explanation_id: str) -> Optional[Any]:
        """Get batch result from Cloud Storage."""
        try:
            from config.config import settings
            from google.cloud import storage
            
            google_cloud_settings = settings.get("GOOGLE_CLOUD", {})
            project_id = google_cloud_settings.get("project_id")
            
            if not project_id:
                return None
            
            bucket_name = f"{project_id}-batch-predictions"
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix="judge_output/")
            
            for blob in blobs:
                if blob.name.endswith('.jsonl'):
                    content = blob.download_as_text()
                    for line in content.strip().split('\n'):
                        if line:
                            prediction = json.loads(line)
                            if prediction.get('explanation_id') == explanation_id:
                                from ..llm_integration.extraction_models import EvaluationResult
                                return EvaluationResult(**prediction)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get batch result from cloud for {explanation_id}: {e}")
            return None

    def _check_batch_results_available(self) -> bool:
        """Check if batch results are available."""
        try:
            # Check local evaluations folder
            evaluations_folder = Path(__file__).parent.parent.parent / "output" / "evaluations"
            batch_results_file = evaluations_folder / "batch_judge_results.json"
            
            if batch_results_file.exists():
                return True
            
            # Check Cloud Storage
            from config.config import settings
            from google.cloud import storage
            
            google_cloud_settings = settings.get("GOOGLE_CLOUD", {})
            project_id = google_cloud_settings.get("project_id")
            
            if not project_id:
                return False
            
            bucket_name = f"{project_id}-batch-predictions"
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix="judge_output/"))
            
            return len(blobs) > 0
            
        except Exception as e:
            logger.debug(f"Failed to check batch results availability: {e}")
            return False

    def get_regulatory_evaluation(self, explanation: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Get regulatory compliance evaluation."""
        llm_eval_result = self._get_judge_evaluation(explanation, prediction)
        
        if not llm_eval_result or not llm_eval_result.regulatory_compliance:
            return {'error': 'Regulatory compliance evaluation failed'}
        
        reg = llm_eval_result.regulatory_compliance
        
        # Check consumer explanation text for technical feature names
        consumer_text = self._extract_consumer_text(explanation)
        technical_feature_names = [
            'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 
            'NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate'
        ]
        
        feature_match_count = len([f for f in technical_feature_names if f.lower() in consumer_text.lower()])
        
        return {
            'principal_reason_identification': {'score': reg.principal_reason_identification.score},
            'individual_specific_content': {'score': reg.individual_specific_content.score},
            'technical_jargon_check': {'score': reg.technical_jargon_check.score},
            'feature_name_matching': {'score': feature_match_count}
        }
    
    def get_cemat_results(self, explanation: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Get CEMAT evaluation results."""
        llm_eval_result = self._get_judge_evaluation(explanation, prediction)
        
        if not llm_eval_result or not llm_eval_result.cemat_evaluation:
            return {'error': 'CEMAT evaluation failed'}
        
        cemat = llm_eval_result.cemat_evaluation
        
        # Calculate understandability scores
        understandability_items = cemat.understandability_items or {}
        understandability_earned = sum(1 for score in understandability_items.values() if score == 1)
        understandability_applicable = len([score for score in understandability_items.values() 
                                     if score is not None and score not in ["N/A", "n/a"]])
        
        # Calculate actionability scores
        actionability_items = cemat.actionability_items or {}
        actionability_earned = sum(1 for score in actionability_items.values() if score == 1)
        actionability_applicable = len([score for score in actionability_items.values() 
                                 if score is not None and score not in ["N/A", "n/a"]])
        
        # Calculate overall score
        total_earned = actionability_earned + understandability_earned
        total_applicable = actionability_applicable + understandability_applicable
        overall_cemat = total_earned / max(1, total_applicable)
        
        return {
            'cemat_score': float(overall_cemat),
            'actionability_score': actionability_earned,
            'understandability_score': understandability_earned,
            'understandability_applicable': understandability_applicable,
            'actionability_applicable': actionability_applicable,
        }

    def _extract_consumer_text(self, explanation: Dict[str, Any]) -> str:
        """Extract consumer explanation text from explanation."""
        import json
        
        explanation_text = explanation.get('explanation_text', '')
        try:
            explanation_json = json.loads(explanation_text)
            return explanation_json.get('consumer_explanation', explanation_text)
        except:
            return explanation_text

    def _verify_llm_counterfactual(self, explanation: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Verify counterfactual suggestions from LLM explanations."""
        if self.cf_verifier is None:
            return None
        
        try:
            llm_eval_result = self._get_judge_evaluation(explanation, prediction)
            result = self.cf_verifier.verify_llm_counterfactual(explanation, prediction, llm_eval_result)
            return result
        except Exception as e:
            logger.warning(f"Counterfactual verification failed: {e}")
            return None

    def _evaluate_flesch_kincaid_readability(self, explanation: Dict[str, Any], prediction: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate readability using Flesch-Kincaid metrics."""
        import textstat
        
        explanation_text = explanation.get('explanation_text', '')
        
        if not explanation_text.strip():
            return {'score': None, 'error': 'No explanation text found'}
        
        try:
            flesch_reading_ease = textstat.flesch_reading_ease(explanation_text)
            normalized_score = max(0, min(100, flesch_reading_ease)) / 100
            
            return {
                'score': normalized_score,
                'flesch_reading_ease': flesch_reading_ease,
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(explanation_text)
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return {'score': None, 'error': str(e)}

    def _evaluate_fidelity(self, explanation: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate fidelity by comparing LLM feature ranking with SHAP ranking."""
        from scipy.stats import spearmanr
        from ..llm_integration.extraction_models import parse_llm_json_response
        from ..model_explainer.utils import FEATURE_NAME_MAPPING
        
        def normalize_to_technical_name(feature_name: str) -> str:
            """Convert any feature name (friendly or technical) to technical name."""
            # First check if it's already a technical name (in the keys)
            if feature_name in FEATURE_NAME_MAPPING.keys():
                return feature_name
            
            # Otherwise map from friendly to technical using reverse mapping
            friendly_to_technical = {v: k for k, v in FEATURE_NAME_MAPPING.items()}
            return friendly_to_technical.get(feature_name, feature_name)
        
        # Extract LLM top 5 features
        try:
            parsed_response = parse_llm_json_response(explanation["explanation_text"])
            raw_llm_features = parsed_response.analysis.top_features if parsed_response and parsed_response.analysis else []
            # Normalize all features to technical names
            llm_features = [normalize_to_technical_name(feature) for feature in raw_llm_features]
        except:
            llm_features = []
        
        # Get SHAP top 5 features
        shap_values = None
        if 'prediction' in prediction:
            if 'shap_values' in prediction['prediction']:
                shap_values = prediction['prediction']['shap_values']
            elif 'shap_explanation' in prediction and 'contributions' in prediction['shap_explanation']:
                shap_values = prediction['shap_explanation']['contributions']
        elif 'shap_values' in prediction:
            shap_values = prediction['shap_values']
        elif 'shap_explanation' in prediction and 'contributions' in prediction['shap_explanation']:
            shap_values = prediction['shap_explanation']['contributions']
        
        if not shap_values:
            return {'score': 0.0, 'error': 'No SHAP values found'}
        
        # Sort by absolute SHAP value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        shap_features = [feature[0] for feature in sorted_features[:5]]
        
        # Calculate correlation
        if not llm_features or not shap_features:
            return {'score': 0.0, 'error': 'No features to compare'}
        
        common_features = list(set(llm_features) & set(shap_features))
        if len(common_features) < 2:
            return {'score': 0.0, 'error': 'Insufficient common features'}
        
        llm_ranks = [llm_features.index(feature) + 1 if feature in llm_features else len(llm_features) + 1 for feature in common_features]
        shap_ranks = [shap_features.index(feature) + 1 if feature in shap_features else len(shap_features) + 1 for feature in common_features]
        
        try:
            correlation, p_value = spearmanr(llm_ranks, shap_ranks)
            if correlation != correlation:  # NaN check
                return {'score': 0.0, 'error': 'Invalid correlation'}
            return {
                'score': correlation,
                'p_value': p_value,
                'llm_features': llm_features,
                'shap_features': shap_features,
                'matching_features': len(set(llm_features) & set(shap_features))
            }
        except:
            return {'score': 0.0, 'error': 'Correlation calculation failed'}

    def _check_immutable_features(self, explanation: Dict[str, Any], prediction: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if explanation mentions immutable features that shouldn't be actionable."""
        import re
        from config.config import settings

        consumer_text = self._extract_consumer_text(explanation).lower()
        immutable_features = settings.get('IMMUTABLE_FEATURES', ['age', 'imputed_income', 'NumberOfDependents'])

        # Define mapping of technical to human-readable terms
        immutable_terms = {
            'age': ['age', 'years old', 'older', 'younger', 'how old', 'aging'],
            'imputed_income': ['imputed income'],
            'NumberOfDependents': ['dependents', 'children', 'family size', 'number of dependents', 'kids']
        }

        mentioned_features = []
        for feature in immutable_features:
            terms_to_check = [feature.lower()]
            if feature in immutable_terms:
                terms_to_check.extend(immutable_terms[feature])

            for term in terms_to_check:
                # Use word boundaries for single words, and simple substring for phrases
                if re.match(r"^[a-zA-Z0-9_]+$", term):
                    # Single word: use word boundaries
                    pattern = r'\b{}\b'.format(re.escape(term))
                else:
                    # Phrase: just use lowercased substring match
                    pattern = re.escape(term)
                if re.search(pattern, consumer_text):
                    mentioned_features.append({'feature': feature, 'term_found': term})
                    break

        count = len(mentioned_features)
        return {
            'count': count,
            'mentioned_features': mentioned_features
        }
    
    def _check_technical_feature_names(self, explanation: Dict[str, Any], prediction: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if explanation uses exact technical feature names from the model."""
        import re
        
        explanation_text = explanation.get('explanation_text', '')
        
        # Extract only the consumer explanation part to avoid false positives from analysis metadata
        consumer_text = self._extract_consumer_text(explanation)
        
        # Get technical features from model metadata if predictor is available
        if self.predictor and hasattr(self.predictor, 'feature_names') and self.predictor.feature_names:
            technical_features = self.predictor.feature_names
        else:
            # Fallback to hardcoded list if predictor not available
            technical_features = [
                'NumberOfOpenCreditLinesAndLoans', 'RevolvingUtilizationOfUnsecuredLines', 
                'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate', 'DebtRatio', 'MonthlyIncome', 'NumberOfDependents', 'age'
            ]
        
        found_features = []
        for feature in technical_features:
            # Use word boundary matching to avoid false positives from substrings
            # For multi-word features, use exact matching
            # For single words like 'age', use word boundaries
            if len(feature.split()) == 1:
                # Single word - use word boundary regex
                pattern = r'\b' + re.escape(feature) + r'\b'
                if re.search(pattern, consumer_text, re.IGNORECASE):
                    # Skip 'age' entirely since it's both a technical feature name and a common word
                    # This avoids the ambiguity of trying to distinguish between natural and technical usage
                    if feature == 'age':
                        continue
                    found_features.append(feature)
            else:
                # Multi-word feature - use exact matching (these are unlikely to be substrings)
                if feature in consumer_text:
                    found_features.append(feature)
        
        return {
            'count': len(found_features),
            'found_features': found_features
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