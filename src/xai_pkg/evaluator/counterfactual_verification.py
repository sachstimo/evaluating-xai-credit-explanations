"""
Counterfactual verification module for evaluating LLM explanations.

This module specifically handles the verification of counterfactual explanations
by checking if the LLM's selected counterfactual actually works when applied.
"""

import logging
import numpy as np
from typing import Dict, List, Any

from ..llm_integration.extraction_models import parse_llm_json_response
from ..model_explainer.utils import FEATURE_NAME_MAPPING

logger = logging.getLogger(__name__)


class CounterfactualVerifier:
    """Handles verification of LLM counterfactual explanations."""
    
    def __init__(self, predictor):
        """Initialize with a predictor for testing counterfactuals."""
        self.predictor = predictor
    
    def verify_llm_counterfactual(self, explanation: Dict[str, Any], 
                                prediction: Dict[str, Any], 
                                comprehensive_result: Any = None) -> Dict[str, Any]:
        """ 
        Verify if the LLM's selected counterfactual actually works.
        
        This tests:
        1. Can we extract the counterfactual from the structured JSON response
        2. Does applying those changes flip the prediction as intended
        3. Bonus: Is it an exact match to any DiCE-generated counterfactual
        
        Returns:
            Dict with verification results including effectiveness score
        """
        # Skip counterfactual verification for approved applications
        # (It doesn't make sense to verify "how to get approved" when already approved)
        original_prediction = prediction['prediction']['prediction']
        if original_prediction == 0:  # 0 = approved
            return {
                'score': None,
                'skipped': True,
                'reason': 'Counterfactual verification skipped for approved applications'
            }
        
        if self.predictor is None:
            return {
                'score': None,
                'error': 'No predictor available for counterfactual verification'
            }
            
        try:
            # Extract original data and prediction (raw intake expected by predictor)
            original_data = prediction['prediction']['input_data'].copy()
            original_prediction = prediction['prediction']['prediction']
            original_probability = prediction['prediction']['probability']
            available_counterfactuals = prediction.get('counterfactuals', [])
            
            # Extract LLM's counterfactual
            llm_counterfactual = self._extract_counterfactual(explanation, prediction, comprehensive_result)
            
            if not llm_counterfactual:
                return {
                    'score': 0.0,
                    'error': 'No counterfactual found in LLM analysis section',
                    'available_counterfactuals': len(available_counterfactuals)
                }
            
            # Match baseline values from original prediction data (for new extraction format)
            llm_counterfactual_with_baselines = self._match_baseline_values(llm_counterfactual, original_data)
            
            # Apply LLM's counterfactual changes
            modified_data = original_data.copy()
            success = self._apply_llm_counterfactual_changes(modified_data, llm_counterfactual_with_baselines)
            
            if not success:
                return {
                    'score': 0.0,
                    'error': 'Could not apply LLM counterfactual changes',
                    'llm_counterfactual': llm_counterfactual
                }
            
            # Ensure no None values in modified data (predictor will handle full preprocessing)
            for key, value in modified_data.items():
                if value is None:
                    modified_data[key] = 0.0
            
            # Test if the counterfactual works
            new_prediction_result = self.predictor.predict_single(modified_data, verbose=False)
            new_prediction = new_prediction_result['prediction']
            new_probability = new_prediction_result['probability']
            
            # Check if prediction flipped as expected
            prediction_flipped = (new_prediction != original_prediction)
            probability_improved = (
                (original_prediction == 0 and new_probability > original_probability) or
                (original_prediction == 1 and new_probability < original_probability)
            )
            
            # Check if it's an exact match to any available counterfactual
            exact_match = self._check_exact_match_to_dice_counterfactuals(
                llm_counterfactual, available_counterfactuals
            )
            
            # Calculate score: 1.0 if flip works, bonus 0.2 if exact match
            base_score = 1.0 if prediction_flipped else 0.0
            bonus_score = 0.2 if exact_match['is_match'] else 0.0
            final_score = min(1.0, base_score + bonus_score)
            
            return {
                'score': final_score,
                'llm_counterfactual': llm_counterfactual,
                'changes_applied': True,
                'original_prediction': original_prediction,
                'new_prediction': new_prediction,
                'original_probability': round(original_probability, 4),
                'new_probability': round(new_probability, 4),
                'prediction_flipped': prediction_flipped,
                'probability_improved': probability_improved,
                'probability_change': round(new_probability - original_probability, 4),
                'exact_match_to_dice': exact_match['is_match'],
                'matched_dice_id': exact_match.get('matched_id'),
                'available_counterfactuals': len(available_counterfactuals)
            }
            
        except Exception as e:
            logger.error(f"LLM counterfactual verification failed: {e}")
            return {
                'score': None,
                'error': str(e)
            }
    
    def _extract_counterfactual(self, explanation: Dict[str, Any], prediction: Dict[str, Any], 
                               comprehensive_result: Any = None) -> Dict[str, Dict[str, float]]:
        """Extract counterfactual changes from various sources."""
        # Try comprehensive result first
        if comprehensive_result is not None:
            return self._extract_from_comprehensive_result(comprehensive_result)
        
        # Try batch result
        if 'llm_response' in explanation and 'counterfactual_extraction' in explanation['llm_response']:
            return self._extract_from_batch_result(explanation['llm_response'])
        
        # Fallback to judge model
        explanation_text = explanation['explanation_text']
        if isinstance(explanation_text, str):
            try:
                import json
                parsed = json.loads(explanation_text)
                explanation_text = parsed.get('consumer_explanation', explanation_text)
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(explanation_text, dict):
            explanation_text = explanation_text.get('consumer_explanation', str(explanation_text))
        
        return self._extract_from_judgellm_result(explanation_text, prediction)
    
    def _extract_from_judgellm_result(self, explanation_text: str, prediction: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract counterfactual using JudgeLLM."""
        try:
            from .judgeLLM import JudgeLLM
            
            if not hasattr(self, '_judge_llm'):
                self._judge_llm = JudgeLLM()
            
            comprehensive_result = self._judge_llm.evaluate_llm_explanation(explanation_text, prediction['prediction'])
            
            if comprehensive_result and comprehensive_result.counterfactual_extraction:
                cf_extract = self._judge_llm.extract_counterfactual_changes(comprehensive_result)
                return cf_extract.get('counterfactual_changes', {})
        except Exception as e:
            logger.debug(f"JudgeLLM counterfactual extraction failed: {e}")
        
        return {}
    
    def _extract_from_batch_result(self, llm_response: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract counterfactual changes from batch results."""
        try:
            changes = llm_response.get('counterfactual_extraction', {}).get('changes', [])
            return {change['feature_name']: {'to': change['target_value']} 
                   for change in changes 
                   if change.get('feature_name') and change.get('target_value') is not None}
        except Exception as e:
            logger.debug(f"Batch result counterfactual extraction failed: {e}")
            return {}
    
    def _extract_from_comprehensive_result(self, comprehensive_result: Any) -> Dict[str, Dict[str, float]]:
        """Extract counterfactual changes from comprehensive evaluation result."""
        try:
            if not comprehensive_result or not comprehensive_result.counterfactual_extraction:
                return {}
            
            cf_extract = comprehensive_result.counterfactual_extraction
            
            # Handle multiple changes structure
            if hasattr(cf_extract, 'changes') and cf_extract.changes:
                return {change.feature_name: {'to': float(change.target_value)} 
                       for change in cf_extract.changes}
            
            # Handle single counterfactual
            if hasattr(cf_extract, 'feature_name') and hasattr(cf_extract, 'to_value'):
                return {cf_extract.feature_name: {
                    'from': float(cf_extract.from_value),
                    'to': float(cf_extract.to_value),
                    'diff': float(cf_extract.to_value) - float(cf_extract.from_value)
                }}
            
            # Handle old array structure
            if hasattr(cf_extract, 'counterfactual_changes'):
                return {change.feature_name: {
                    'from': float(change.from_value),
                    'to': float(change.to_value),
                    'diff': float(change.to_value) - float(change.from_value)
                } for change in cf_extract.counterfactual_changes}
            
            return {}
            
        except Exception as e:
            logger.debug(f"Failed to extract counterfactuals from comprehensive result: {e}")
            return {}
    
    def _match_baseline_values(self, llm_counterfactual: Dict[str, Any], original_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Match target values with baseline values from original data."""
        matched_counterfactuals = {}
        
        for feature_name, change_info in llm_counterfactual.items():
            # Use fuzzy matching to find the correct feature name
            mapped_feature_name = self._map_feature_name(feature_name, original_data)
            
            if mapped_feature_name in original_data:
                # Add baseline if not present
                if 'to' in change_info and 'from' not in change_info:
                    matched_counterfactuals[mapped_feature_name] = {
                        'from': float(original_data[mapped_feature_name]),
                        'to': float(change_info['to']),
                        'diff': float(change_info['to']) - float(original_data[mapped_feature_name]),
                        'confidence': change_info.get('confidence', 'medium')
                    }
                else:
                    matched_counterfactuals[mapped_feature_name] = change_info
            else:
                logger.warning(f"Feature {feature_name} not found in original data, skipping")
        
        return matched_counterfactuals
    
    
    def _apply_llm_counterfactual_changes(self, modified_data: Dict[str, Any], 
                                        llm_counterfactual: Dict[str, Dict[str, float]]) -> bool:
        """Apply the LLM's counterfactual changes to the data."""
        try:
            changes_dict = llm_counterfactual.get('counterfactual_changes', llm_counterfactual)
            
            for feature_name, change_info in changes_dict.items():
                target_feature = self._map_feature_name(feature_name, modified_data)
                
                if target_feature in modified_data:
                    to_value = change_info['to']
                    old_value = modified_data[target_feature]
                    modified_data[target_feature] = 0.0 if to_value is None else to_value
                    logger.debug(f"Applied change: {target_feature}: {old_value} → {to_value}")
                else:
                    logger.warning(f"Feature {target_feature} not found in original data")
                    continue
            
            return True
        except Exception as e:
            logger.error(f"Error applying counterfactual changes: {e}")
            return False
    
    def _map_feature_name(self, feature_name: str, original_data: Dict[str, Any]) -> str:
        """Map LLM feature names to original data feature names using existing mapping."""
        # If feature exists directly, use it
        if feature_name in original_data:
            return feature_name
        
        # If LLM supplies a _log name, always map back to the raw base feature
        if feature_name.endswith('_log'):
            base_feature = feature_name[:-4]
            return base_feature if base_feature in original_data else feature_name
        
        # Use reverse mapping from existing FEATURE_NAME_MAPPING
        friendly_to_technical = {friendly: technical for technical, friendly in FEATURE_NAME_MAPPING.items()}
        
        # Try mapping
        if feature_name in friendly_to_technical:
            mapped_name = friendly_to_technical[feature_name]
            if mapped_name in original_data:
                return mapped_name
        
        # Comprehensive fuzzy matching for all LLM variations
        # Handle late payment features
        if "Days" in feature_name:
            if "30-59" in feature_name or "30_59" in feature_name:
                target = "NumberOfTime30-59DaysPastDueNotWorse"
                if target in original_data:
                    logger.debug(f"Fuzzy matched: {feature_name} -> {target}")
                    return target
            elif "60-89" in feature_name or "60_89" in feature_name:
                target = "NumberOfTime60-89DaysPastDueNotWorse"
                if target in original_data:
                    logger.debug(f"Fuzzy matched: {feature_name} -> {target}")
                    return target
            elif "90" in feature_name:
                target = "NumberOfTimes90DaysLate"
                if target in original_data:
                    logger.debug(f"Fuzzy matched: {feature_name} -> {target}")
                    return target
        
        # Handle real estate loans
        if "RealEstate" in feature_name or "real estate" in feature_name.lower():
            target = "NumberRealEstateLoansOrLines"
            if target in original_data:
                logger.debug(f"Fuzzy matched: {feature_name} -> {target}")
                return target
        
        # If no mapping found, return original name (will likely fail but logged)
        logger.warning(f"Feature {feature_name} not found in original data, skipping")
        return feature_name
    
    
    def _check_exact_match_to_dice_counterfactuals(self, llm_counterfactual: Dict[str, Dict[str, float]], 
                                                 available_counterfactuals: List[Dict]) -> Dict[str, Any]:
        """Check if LLM counterfactual exactly matches any DiCE-generated counterfactual."""
        for i, dice_cf in enumerate(available_counterfactuals):
            if self._counterfactuals_match(llm_counterfactual, dice_cf):
                return {
                    'is_match': True,
                    'matched_id': i + 1,
                    'matched_counterfactual': dice_cf
                }
        
        return {'is_match': False}
    
    def _counterfactuals_match(self, llm_cf: Dict[str, Dict[str, float]], 
                             dice_cf: Dict[str, Dict[str, float]], 
                             tolerance: float = 0.001) -> bool:
        """Check if two counterfactuals match within tolerance."""
        if set(llm_cf.keys()) != set(dice_cf.keys()):
            return False
        
        return all(abs(llm_cf[feature]['to'] - dice_cf[feature]['to']) <= tolerance 
                  for feature in llm_cf.keys())