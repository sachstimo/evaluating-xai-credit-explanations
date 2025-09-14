# src/model_explainer/utils.py
import numpy as np
import pandas as pd
import re
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import settings

logger = logging.getLogger(__name__)

# Centralized feature name mapping for consumer-friendly explanations
FEATURE_NAME_MAPPING = {
    'RevolvingUtilizationOfUnsecuredLines': 'credit card usage',
    'DebtRatio': 'debt-to-income ratio', 
    'MonthlyIncome': 'monthly income',
    'NumberOfOpenCreditLinesAndLoans': 'number of open credit lines',
    'NumberRealEstateLoansOrLines': 'real estate loans',
    'NumberOfDependents': 'number of dependents',
    'NumberOfTime30-59DaysPastDueNotWorse': 'recent late payments (30-59 days)',
    'NumberOfTime60-89DaysPastDueNotWorse': 'recent late payments (60-89 days)', 
    'NumberOfTimes90DaysLate': 'serious late payments (90+ days)',
    'age': 'age'
}

def get_friendly_feature_name(technical_name: str) -> str:
    """Convert technical feature name to consumer-friendly name."""
    return FEATURE_NAME_MAPPING.get(technical_name, technical_name)

def format_system_variables(model_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create system variables for LangChain system prompt.
    Extracts basic model metadata without complex transformations.
    
    Args:
        model_metadata: Model configuration and training info
        
    Returns:
        Dictionary of system variables for prompt templates
    """
    
    # System prompt variables (stable across predictions)
    system_vars = {
        'model_type': model_metadata['model_type'],
        'train_set_shape': model_metadata['train_set_shape'],
        'cv_folds': model_metadata['cv_folds'],
        'feature_count': len(model_metadata['used_features']),
        'log_scale_features': model_metadata['log_scale_features'],
        'FEATURE_NAME_MAPPING': str(FEATURE_NAME_MAPPING),
        'used_features (consumer-friendly)': [get_friendly_feature_name(feature) for feature in model_metadata['used_features']],
        'immutable_features': [get_friendly_feature_name(feature) for feature in settings.get("immutable_features", ["age", "imputed_income", "NumberOfDependents"])]
    }
    
    return system_vars

def format_prediction_variables(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create variables for LangChain system and user prompts.
    Simple extraction with minimal preprocessing.
    
    Args:
        prediction_data: Single prediction with SHAP and counterfactuals
        model_metadata: Model configuration and training info
        
    Returns:
        Dict with user_vars for prompt templates
    """
    
    # Extract prediction components
    pred_info = prediction_data['prediction'] 
    shap_info = prediction_data['shap_explanation']
    cf_info = prediction_data['counterfactuals']
    
    # User prompt variables (specific to each prediction) - handle None values to prevent format errors
    user_vars = {
        'prediction_outcome': 'declined' if pred_info['prediction'] == 1 else 'approved',
        'prediction': pred_info['prediction'],
        'prediction_id': pred_info['prediction_id'],
        'timestamp': pred_info['timestamp'] or 'N/A',
        'age': pred_info['input_data']['age'] or 0,
        'monthly_income': pred_info['input_data']['MonthlyIncome'] or 0.0,
        'debt_ratio': pred_info['input_data']['DebtRatio'] or 0.0,
        'revolving_utilization': pred_info['input_data']['RevolvingUtilizationOfUnsecuredLines'] or 0.0,
        'number_of_open_credit_lines': pred_info['input_data']['NumberOfOpenCreditLinesAndLoans'] or 0,
        'number_real_estate_loans': pred_info['input_data']['NumberRealEstateLoansOrLines'] or 0,
        'number_of_dependents': pred_info['input_data']['NumberOfDependents'] or 0,
        'number_30_59_late': pred_info['input_data']['NumberOfTime30-59DaysPastDueNotWorse'] or 0,
        'number_60_89_late': pred_info['input_data']['NumberOfTime60-89DaysPastDueNotWorse'] or 0,
        'number_90_days_late': pred_info['input_data']['NumberOfTimes90DaysLate'] or 0,
        'shap_contributions_formatted': format_shap(shap_info['contributions']),
        'counterfactuals_formatted': format_counterfactuals(cf_info)
    }

    return user_vars

def format_shap(contributions: Dict[str, float]) -> str:
    """
    Format SHAP contributions with consumer-friendly feature names.
    Translates technical names to human-readable terms.
    """
    formatted_items = []
    
    # Sort by absolute impact
    sorted_contributions = sorted(
        contributions.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    for feature, value in sorted_contributions:
        # Use friendly name from centralized mapping
        friendly_name = get_friendly_feature_name(feature)
        formatted_items.append(f"{friendly_name}: {value:.4f}")
    
    return "\n".join(formatted_items)


def format_counterfactuals(counterfactuals: List[Dict[str, Any]]) -> str:
    """
    Simple formatting of counterfactuals with safety margin rounding.
    Only uses the first counterfactual in the list.
    """
    if not counterfactuals:
        return "No counterfactual scenarios available."

    # Safety margin rounding: (direction, precision)  
    # age and NumberOfDependents are immutable, so excluded
    feature_rounding_directions = {
        'MonthlyIncome': ('up', 10),                                      # Round up to nearest $10
        'DebtRatio': ('down', 0.01),                                      # Round down to nearest 0.01
        'RevolvingUtilizationOfUnsecuredLines': ('down', 0.01),          # Round down to nearest 0.01
        'NumberOfOpenCreditLinesAndLoans': ('up', 1),                    # Round up to nearest integer
        'NumberRealEstateLoansOrLines': ('up', 1),                       # Round up to nearest integer
        'NumberOfTime30-59DaysPastDueNotWorse': ('down', 1),             # Round down to nearest integer
        'NumberOfTime60-89DaysPastDueNotWorse': ('down', 1),             # Round down to nearest integer
        'NumberOfTimes90DaysLate': ('down', 1),                          # Round down to nearest integer
    }
    
    # Handle error cases - sometimes counterfactuals might be error strings
    if isinstance(counterfactuals, dict) and "error" in counterfactuals:
        return f'Counterfactual generation failed: {counterfactuals["error"]}' #type: ignore

    # Only use the first counterfactual
    if not counterfactuals:
        return "No counterfactual scenarios available."
    
    cf = counterfactuals[0]  # Take only the first counterfactual
    
    # Skip if this is an error case
    if isinstance(cf, dict) and "error" in cf:
        return "First counterfactual contains an error."
    
    changes = []
    for feature, change_info in cf.items():
        # Make sure change_info is a dict with the expected structure
        if isinstance(change_info, dict) and all(key in change_info for key in ['from', 'to', 'diff']):
            from_val = change_info['from']
            to_val = change_info['to']
            diff = change_info['diff']
            
            # Apply safety margin rounding to the target value
            rounding_config = feature_rounding_directions.get(feature)
            if isinstance(to_val, (int, float)) and not pd.isna(to_val) and rounding_config:
                direction, precision = rounding_config
                
                if direction == 'up':
                    to_val_rounded = np.ceil(to_val / precision) * precision
                elif direction == 'down':
                    to_val_rounded = np.floor(to_val / precision) * precision
                else:
                    to_val_rounded = round(to_val / precision) * precision
            else:
                # No rounding config (likely immutable feature) - keep original value
                to_val_rounded = to_val
            
            # Use friendly name from centralized mapping
            friendly_name = get_friendly_feature_name(feature)
            changes.append(f"{friendly_name}: {from_val} → {to_val_rounded} (change: {diff:+.4f})")
    
    if changes:  # Only return if it has changes
        return "Scenario 1:\n" + "\n".join(changes)
    else:
        return "No valid counterfactual scenarios available."


def convert_logit_to_probability(logit: float) -> float:
    """Convert logit value to probability."""
    return 1 / (1 + np.exp(-logit))


def unlog(value: float) -> Optional[float]:
    """
    Unlog a value (assumes natural log1p transformation).
    
    Args:
        value: Log-transformed value to unlog
        
    Returns:
        Original value or None if conversion is not possible
    """
    import pandas as pd
    import logging
    
    if value is None or pd.isna(value):
        return None
    
    # Handle negative values (invalid for expm1 transformation)
    if value < 0:
        # Log a warning but don't raise an error
        logger = logging.getLogger(__name__)
        logger.debug(f"Cannot unlog negative value {value}, returning None")
        return None
        
    try:
        return np.expm1(value)
    except (ValueError, OverflowError) as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Error unlogging value {value}: {e}, returning None")
        return None


def add_original_values_to_dict(
    input_dict: dict, 
    log_scale_features: list, 
    keep_log_features: bool = False
) -> dict:
    """
    Add original unlogged values to a dictionary containing log-transformed features.
    
    Args:
        input_dict: Dictionary containing processed data with log-transformed features
        log_scale_features: List of original feature names that were log-transformed
        keep_log_features: If False, removes the _log features after adding originals
        
    Returns:
        Dictionary with original unlogged values added (and optionally log features removed)
        
    Example:
        input_dict = {"MonthlyIncome_log": 9.19, "age": 35}
        log_scale_features = ["MonthlyIncome"]
        result = add_original_values_to_dict(input_dict, log_scale_features, keep_log_features=False)
        # Returns: {"MonthlyIncome": 9800.0, "age": 35}
    """
    
    # Make sure it's a dictionary
    if not isinstance(input_dict, dict):
        try:
            input_dict = dict(input_dict)  # Convert to dict if not already
        except Exception as e:
            logger.error(f"Failed to convert input to dict: {e}")
            raise

    # Create a copy to avoid modifying the original
    result_dict = input_dict.copy()
    
    # Track log features to potentially remove later
    log_features_to_remove = []
    
    # Add unlogged values for log-transformed features
    keys_to_check = list(result_dict.keys()) 
    for key in keys_to_check:
        if key.endswith('_log'):
            base_feature = key.replace('_log', '')
            if base_feature in log_scale_features:
                log_value = result_dict[key]
                if log_value is not None and not pd.isna(log_value):
                    result_dict[base_feature] = unlog(log_value)
                else:
                    result_dict[base_feature] = None
                
                # Mark for removal if requested
                if not keep_log_features:
                    log_features_to_remove.append(key)
    
    # Remove log features if requested
    for log_feature in log_features_to_remove:
        result_dict.pop(log_feature, None)
    
    return result_dict

def flatten_explanations(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert the explanations JSON to a flat DataFrame with parsed explanation columns.
    
    Args:
        data: Dictionary containing explanation data
    
    Returns:
        pd.DataFrame: DataFrame with explanation data and parsed explanation columns
    """
    
    # Convert to DataFrame directly (already flat)
    df = pd.DataFrame(data)
    
    # Handle token_usage if it's a nested dictionary
    if 'token_usage' in df.columns and df['token_usage'].notna().any():
        # Extract relevant token usage metrics if they exist
        try:
            df['prompt_tokens'] = df['token_usage'].apply(
                lambda x: x.get('prompt_tokens', None) if isinstance(x, dict) else None
            )
            df['completion_tokens'] = df['token_usage'].apply(
                lambda x: x.get('completion_tokens', None) if isinstance(x, dict) else None
            )
            df['total_tokens'] = df['token_usage'].apply(
                lambda x: x.get('total_tokens', None) if isinstance(x, dict) else None
            )
        except Exception as e:
            pass

    # Parse explanation_text JSON strings into separate columns
    if 'explanation_text' in df.columns:
        def safe_parse_explanation(text_value):
            """Safely parse explanation text JSON string."""
            if pd.isna(text_value) or text_value == '':
                return {}
            
            if isinstance(text_value, dict):
                return text_value
            
            if isinstance(text_value, str):
                try:
                    return json.loads(text_value)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse explanation text as JSON: {e}")
                    return {}
            
            return {}
        
        # Parse all explanation texts
        parsed_explanations = df['explanation_text'].apply(safe_parse_explanation)
        
        # Dynamically extract all available keys from the parsed JSON
        # Get all unique keys across all parsed explanations
        all_keys = set()
        for parsed in parsed_explanations:
            if isinstance(parsed, dict):
                all_keys.update(parsed.keys())
        
        # Create columns for each available key
        for key in all_keys:
            if key != 'analysis':  # Handle analysis separately if it's nested
                df[key] = parsed_explanations.apply(
                    lambda x: x.get(key, '') if x else ''
                )
        
        # Handle nested analysis object if it exists
        if 'analysis' in all_keys:
            df['analysis_top_features'] = parsed_explanations.apply(
                lambda x: x.get('analysis', {}).get('top_features', []) if x else []
            )
        
        # Add a flag to indicate successful parsing
        df['explanation_parsed'] = parsed_explanations.apply(lambda x: bool(x))
        
        logger.info(f"Successfully parsed {df['explanation_parsed'].sum()}/{len(df)} explanation texts")
        if all_keys:
            logger.info(f"Available explanation keys: {sorted(all_keys)}")

    return df


def preprocess_llm_json_response(response_content: str, llm_name: str = "unknown") -> str:
    """
    Simple preprocessing for LLM JSON responses.
    Only handles basic markdown cleanup and lets Pydantic handle validation.
    
    Args:
        response_content: Raw response content from LLM
        llm_name: Name of the LLM (for logging)
        
    Returns:
        Cleaned JSON string ready for Pydantic parsing
    """
    if not response_content or not response_content.strip():
        raise ValueError("Empty response provided")
    
    cleaned = response_content.strip()
    
    # Remove markdown code blocks if present
    if '```json' in cleaned:
        start = cleaned.find('```json') + 7
        end = cleaned.find('```', start)
        if end != -1:
            cleaned = cleaned[start:end].strip()
        else:
            cleaned = cleaned[start:].strip()
    elif cleaned.startswith('```'):
        start = cleaned.find('```') + 3
        newline_pos = cleaned.find('\n', start)
        if newline_pos != -1:
            start = newline_pos + 1
        end = cleaned.find('```', start)
        if end != -1:
            cleaned = cleaned[start:end].strip()
        else:
            cleaned = cleaned[start:].strip()
    
    # Extract JSON from mixed responses
    if not cleaned.startswith('{'):
        start_idx = cleaned.find('{')
        if start_idx != -1:
            end_idx = cleaned.rfind('}')
            if end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx + 1]
    
    # Handle case where JSON is followed by extra text (like "For troubleshooting...")
    if cleaned.startswith('{'):
        # Find the end of the JSON object by counting braces
        brace_count = 0
        json_end = -1
        
        for i, char in enumerate(cleaned):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i
                    break
        
        if json_end != -1:
            cleaned = cleaned[:json_end + 1]
    
    # Remove JSON comments that break parsing
    import re
    cleaned = re.sub(r'//.*?(?=\n|$)', '', cleaned)  # Remove // comments
    cleaned = re.sub(r'#.*?(?=\n|$)', '', cleaned)   # Remove # comments
    
    # Fix common JSON issues that prevent parsing
    # Fix unquoted N/A values (common in evaluation responses)
    cleaned = re.sub(r':\s*N/A', ': "N/A"', cleaned)
    cleaned = re.sub(r':\s*n/a', ': "n/a"', cleaned)
    
    # Fix regulatory compliance format - convert simple integers to proper score objects
    # Match patterns like "principal_reason_identification": 1 and convert to {"score": 1}
    regulatory_fields = [
        'principal_reason_identification',
        'individual_specific_content', 
        'technical_jargon_check'
    ]
    
    for field in regulatory_fields:
        # Replace simple integer values with proper score objects
        pattern = f'"({field})":\\s*([01])'
        replacement = r'"\1": {"score": \2}'
        cleaned = re.sub(pattern, replacement, cleaned)
    
    # Remove trailing commas that break JSON parsing
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    return cleaned
