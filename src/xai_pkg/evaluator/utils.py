"""
Utility functions for batch evaluation processing and CSV export.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import time

logger = logging.getLogger(__name__)


def prepare_batch_jsonl(
    explanations: List[Dict[str, Any]], 
    predictions_lookup: Dict[str, Any],
    output_dir: Path,
    filename_prefix: str = "batch_requests",
    cache_name: Optional[str] = None
) -> Path:
    """
    Prepare a JSONL file for Gemini API batch submission.
    
    Follows the exact format from the Gemini API documentation:
    {"key": "request-1", "request": {"contents": [...], "generation_config": {...}}}
    
    Args:
        explanations: List of explanation dictionaries from DynamoDB/JSON
        predictions_lookup: Dictionary mapping prediction_id to prediction data
        output_dir: Directory to save the JSONL file
        filename_prefix: Prefix for the filename
    
    Returns:
        Path to the created JSONL file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    jsonl_file = output_dir / f"{filename_prefix}_{timestamp}.jsonl"
    
    logger.info(f"📝 Preparing JSONL file: {jsonl_file}")
    logger.info(f"🔄 Converting {len(explanations)} explanations to batch format")
    
    with open(jsonl_file, 'w') as f:
        for i, explanation in enumerate(explanations):
            pred_id = explanation.get('prediction_id')
            if pred_id not in predictions_lookup:
                logger.warning(f"⚠️ Skipping explanation {i}: prediction_id {pred_id} not found")
                continue
            
            try:
                # Extract context from prediction
                pred_data = predictions_lookup[pred_id]
                if 'prediction' in pred_data:
                    prediction_outcome = "DECLINED" if pred_data['prediction']['prediction'] == 1 else "APPROVED"
                    input_data = pred_data['prediction'].get('input_data', {})
                else:
                    prediction_outcome = "APPROVED" if pred_data.get('original_prediction', 0) == 0 else "DECLINED"
                    input_data = pred_data.get('input_data', {})
                
                # Create user prompt
                from ..llm_integration.prompts import EVALUATOR_USER_PROMPT
                explanation_text = explanation.get('explanation_text', '')
                
                user_prompt = EVALUATOR_USER_PROMPT.format(
                    prediction_outcome=prediction_outcome,
                    age=input_data.get('age', 0),
                    monthly_income=input_data.get('MonthlyIncome', 0),
                    debt_ratio=input_data.get('DebtRatio', 0),
                    revolving_utilization=input_data.get('RevolvingUtilizationOfUnsecuredLines', 0),
                    number_of_open_credit_lines=input_data.get('NumberOfOpenCreditLinesAndLoans', 0),
                    number_30_59_late=input_data.get('NumberOfTime30-59DaysPastDueNotWorse', 0),
                    number_60_89_late=input_data.get('NumberOfTime60-89DaysPastDueNotWorse', 0),
                    number_90_days_late=input_data.get('NumberOfTimes90DaysLate', 0),
                    consumer_explanation=explanation_text
                )
                
                # Create request in exact format from Gemini API docs
                request_data = {
                    "key": f"request-{i+1}",
                    "request": {
                        "contents": [
                            {
                                "parts": [{"text": user_prompt}]
                            }
                        ],
                        "generation_config": {
                            "temperature": 0.2,
                            "maxOutputTokens": 8000
                        }
                    }
                }
                
                # Add cached content reference inside request object
                if cache_name:
                    request_data["request"]["cached_content"] = cache_name
                
                # Add metadata to trace results back to explanations  
                request_data["custom_metadata"] = {
                    "prediction_id": pred_id,
                    "llm_name": explanation.get('llm_name', 'unknown'),
                    "explanation_id": explanation.get('explanation_id', f'batch-{i+1}')
                }
                
                # Write as JSONL (one JSON object per line)
                f.write(json.dumps(request_data) + "\n")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to prepare explanation {i}: {e}")
                continue
    
    logger.info(f"✅ JSONL file created: {jsonl_file}")
    return jsonl_file


def process_batch_results(batch_results: List[Dict[str, Any]], job_info: Dict[str, Any]) -> pd.DataFrame:
    """Process raw batch results into a pandas DataFrame."""
    processed_results = []
    
    for result in batch_results:
        try:
            # Extract metadata
            metadata = result.get('metadata', {})
            prediction_id = metadata.get('prediction_id', 'unknown')
            llm_name = metadata.get('llm_name', 'unknown')
            explanation_id = metadata.get('explanation_id', 'unknown')
            
            # Extract evaluation result
            evaluation_result = result.get('evaluation_result', {})
            
            # Extract scores
            cemat_score = _get_cemat_score(evaluation_result)
            reg_score = _get_reg_score(evaluation_result)
            cf_changes_count = _get_cf_changes_count(evaluation_result)
            cf_changes = _get_cf_changes(evaluation_result)
            
            processed_results.append({
                'prediction_id': prediction_id,
                'llm_name': llm_name,
                'explanation_id': explanation_id,
                'cemat_score': cemat_score,
                'regulatory_score': reg_score,
                'cf_changes_count': cf_changes_count,
                'cf_changes': cf_changes,
                'raw_evaluation': evaluation_result
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to process result: {e}")
            continue
    
    return pd.DataFrame(processed_results)


def _get_cemat_score(evaluation_result: Dict[str, Any]) -> Optional[float]:
    """Extract CEMAT score from evaluation result."""
    return _safe_get(evaluation_result, 'cemat_score')


def _get_reg_score(evaluation_result: Dict[str, Any]) -> Optional[float]:
    """Extract regulatory compliance score from evaluation result."""
    return _safe_get(evaluation_result, 'regulatory_compliance_score')


def _get_cf_changes_count(evaluation_result: Dict[str, Any]) -> Optional[int]:
    """Extract number of counterfactual changes from evaluation result."""
    cf_changes = _get_cf_changes(evaluation_result)
    return len(cf_changes) if cf_changes else 0


def _get_cf_changes(evaluation_result: Dict[str, Any]) -> Optional[List[str]]:
    """Extract counterfactual changes from evaluation result."""
    cf_extraction = _safe_get(evaluation_result, 'counterfactual_extraction', {})
    if cf_extraction:
        return _safe_get(cf_extraction, 'changes', [])
    return []


def _safe_get(data: Dict[str, Any], key: str, default=None):
    """Safely get value from nested dictionary."""
    try:
        return data.get(key, default)
    except (KeyError, TypeError, AttributeError):
        return default


def save_batch_results(
    batch_results: List[Dict[str, Any]],
    job_info: Dict[str, Any], 
    output_dir: Path, 
    filename_prefix: str = "gemini_batch",
    return_format: Optional[str] = 'json'
) -> Dict[str, Path]:
    """Save batch results in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    
    # Sanitize job_id for filename (remove invalid characters)
    job_id = job_info.get('job_id', str(timestamp))
    safe_job_id = job_id.replace('batches/', '')
    
    if return_format == 'json':
        # Save raw results as JSON
        json_file = output_dir / f"{filename_prefix}_raw_{safe_job_id}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'job_info': job_info,
                'results': batch_results,
                'timestamp': timestamp
            }, f, indent=2)
        return {
            'raw_json': json_file
        }
        
    elif return_format == 'csv':
        # Process and save as CSV
        df = process_batch_results(batch_results, job_info)
        csv_file = output_dir / f"{filename_prefix}_processed_{safe_job_id}.csv"
        df.to_csv(csv_file, index=False)
        
        return {
            'processed_csv': csv_file
        }

    else:
        raise ValueError(f"Invalid return format: {return_format}")
    


def load_and_process_evaluations(
    input_path: Path, 
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load and process evaluation results from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Handle different input structures
    if 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        results = [data]
    
    df = process_batch_results(results, data.get('job_info', {}))
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = output_dir / f"processed_evaluations_{int(time.time())}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 Processed evaluations saved to: {csv_file}")
    
    return df
