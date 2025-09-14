"""
Pydantic models for structured extraction from LLM credit explanations.
This ensures type-safe, validated parsing of explanation components.
"""

from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field, field_validator
import re
import json


def _extract_with_regex(text: str) -> Optional['CreditExplanationResponse']:
    """Extract explanation data using regex patterns when JSON structure is missing."""
    
    # Extract consumer_explanation field with flexible patterns
    consumer_patterns = [
        r'"consumer_explanation"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',  # Standard JSON
        r"'consumer_explanation'\s*:\s*'([^']*(?:\\.[^']*)*)'",  # Single quotes
        r'"consumer_explanation"\s*:\s*"([^"]*)',  # Unclosed quote (truncated)
        r'consumer_explanation["\']?\s*:\s*["\']([^"\']*)',  # Flexible quotes
        r'"([^"]*declined[^"]*)"',  # Any quoted decline explanation
        r"'([^']*declined[^']*)'",  # Single quoted decline
    ]
    
    consumer_explanation = ""
    for pattern in consumer_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            consumer_explanation = match.group(1).replace('\\"', '"').replace("\\'", "'")
            break
    
    # If no consumer_explanation found, try broader patterns
    if not consumer_explanation:
        broad_patterns = [
            r'(We[^.]*declined[^.]*\.(?:[^.]*\.){0,5})',  # Decline explanation
            r'(Thank you[^.]*unable[^.]*\.(?:[^.]*\.){0,5})',  # Thank you + unable
            r'(Hello[^.]*regret[^.]*\.(?:[^.]*\.){0,5})',  # Greeting + regret
            r'(.*?declined.*?\.(?:[^.]*\.){0,3})',  # Any decline mention
            r'"([^"]{50,})"',  # Any long quoted string (likely explanation)
        ]
        
        for pattern in broad_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                consumer_explanation = match.group(1).strip()
                consumer_explanation = re.sub(r'\s+', ' ', consumer_explanation)
                break
    
    # Extract top_features array
    top_features = []
    features_patterns = [
        r'"top_features"\s*:\s*\[(.*?)\]',  # Standard array
        r'"top_features"\s*:\s*\[([^]]*)',  # Unclosed array (truncated)
    ]
    
    for pattern in features_patterns:
        features_match = re.search(pattern, text, re.DOTALL)
        if features_match:
            features_str = features_match.group(1)
            # Extract quoted strings from the array
            feature_matches = re.findall(r'"([^"]*)"', features_str)
            if not feature_matches:
                feature_matches = re.findall(r"'([^']*)'", features_str)
            top_features = feature_matches[:5]  # Limit to 5 features
            break
    
    # Create response object if we found at least a consumer explanation
    if consumer_explanation:
        manual_data = {
            "consumer_explanation": consumer_explanation,
            "analysis": {
                "top_features": top_features
            }
        }
        
        try:
            return CreditExplanationResponse(**manual_data)
        except Exception:
            pass
    
    return None


# Legacy CounterfactualChange for backward compatibility - now replaced by comprehensive evaluation


class ExplanationAnalysis(BaseModel):
    """Structured analysis section extracted from LLM explanations."""
    
    top_features: List[str] = Field(
        description="Top 5 most important features that influenced the decision, in priority order"
    )
    

class CreditExplanationResponse(BaseModel):
    """Complete structured response from LLM in JSON format."""
    
    consumer_explanation: str = Field(
        description="The main explanation text intended for the consumer/applicant"
    )
    
    analysis: ExplanationAnalysis = Field(
        description="Internal analysis containing feature priorities and counterfactual scenarios"
    )


class CEMATResult(BaseModel):
    """CEMAT evaluation result with understandability and actionability scores."""
    
    understandability_items: Dict[str, Union[int, str]] = Field(
        description="Individual scores for understandability items (0, 1, or 'N/A')"
    )

    actionability_items: Dict[str, Union[int, str]] = Field(
        description="Individual scores for actionability items (0, 1, or 'N/A')"
    )


class RegulatoryComplianceItem(BaseModel):
    """Individual regulatory compliance marker with score and reasoning."""
    
    score: int = Field(
        ge=0, le=1,
        description="Binary compliance score: 1 (compliant) or 0 (non-compliant)"
    )

class RegulatoryCompliance(BaseModel):
    """Three core regulatory compliance markers."""
    
    principal_reason_identification: RegulatoryComplianceItem = Field(
        description="Verifies that explanations clearly articulate the primary factor driving the decision"
    )
    individual_specific_content: RegulatoryComplianceItem = Field(
        description="Ensures explanations reference actual applicant data rather than generic statements"
    )
    # TEMPORARY: Made optional to analyze current batch results with missing field
    # TODO: Revert to mandatory for next batch run after fixing prompt consistency
    technical_jargon_check: Optional[RegulatoryComplianceItem] = Field(
        default_factory=lambda: RegulatoryComplianceItem(score=0),
        description="Confirms explanations avoid model-specific terminology and abstract concepts"
    )


class CounterfactualChange(BaseModel):
    """Represents a single feature change extracted from natural language explanation."""
    
    feature_name: str = Field(
        description="The exact technical name of the feature (e.g., 'RevolvingUtilizationOfUnsecuredLines', 'MonthlyIncome')"
    )
    target_value: Optional[Union[float, int]] = Field(
        default=None,
        description="The target value suggested by the LLM (baseline will be matched separately). None if LLM didn't provide a numeric value."
    )
    
    @field_validator('target_value')
    @classmethod
    def validate_target_value(cls, v):
        """Handle None values gracefully and convert strings to numbers if possible."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                # Try to convert string to number
                if '.' in v:
                    return float(v)
                else:
                    return int(v)
            except ValueError:
                # If conversion fails, return None instead of raising error
                return None
        return v


class CounterfactualExtraction(BaseModel):
    """Multiple counterfactual changes extracted from natural language explanation."""
    
    changes: List[CounterfactualChange] = Field(
        description="List of feature changes suggested by the LLM to change the prediction outcome",
        min_items=0
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result combining CEMAT, regulatory compliance, and counterfactual extraction."""
    
    cemat_evaluation: CEMATResult = Field(
        description="CEMAT understandability and actionability assessment"
    )
    regulatory_compliance: RegulatoryCompliance = Field(
        description="Three regulatory compliance markers with reasoning"
    )
    counterfactual_extraction: CounterfactualExtraction = Field(
        description="Counterfactual changes extracted from natural language"
    )


def parse_llm_json_response(json_response: str) -> Optional[CreditExplanationResponse]:
    """
    Ultra-robust JSON parsing for LLM responses that handles all edge cases.
    
    Args:
        json_response: JSON string from LLM (any format)
        
    Returns:
        Validated CreditExplanationResponse object or None if parsing fails
    """
    from pydantic import ValidationError
    
    if not json_response or not json_response.strip():
        return None
    
    # Step 1: Try direct JSON parse first (handles clean responses)
    try:
        data = json.loads(json_response.strip())
        return CreditExplanationResponse(**data)
    except (json.JSONDecodeError, ValidationError):
        pass
    
    # Step 2: Clean up the response text
    cleaned = json_response.strip()
    
    # Remove markdown code blocks (improved pattern)
    if '```' in cleaned:
        # More flexible markdown extraction
        patterns = [
            r'```(?:json)?\s*(\{.*?\})\s*```',  # Standard markdown
            r'```(?:json)?\s*(\{.*?)```',       # Missing closing newline
            r'```(\{.*?\})\s*```',              # No language specified
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
                break
    
    # Step 3: Find and extract JSON object boundaries
    start_idx = cleaned.find('{')
    if start_idx == -1:
        # No JSON structure found, try regex extraction directly
        return _extract_with_regex(cleaned)
    
    # Find matching closing brace with better handling
    brace_count = 0
    end_idx = len(cleaned)  # Default to end if no matching brace found
    in_string = False
    escape_next = False
    
    for i, char in enumerate(cleaned[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
    
    # Step 4: Try parsing extracted JSON with multiple strategies
    json_candidates = []
    
    # Primary candidate: extracted JSON
    json_str = cleaned[start_idx:end_idx]
    json_candidates.append(json_str)
    
    # Fallback candidate: clean up control characters
    clean_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    json_candidates.append(clean_str)
    
    # Emergency candidate: try to fix common JSON issues
    fixed_str = clean_str
    # Fix common issues: trailing commas, unescaped quotes in strings
    fixed_str = re.sub(r',\s*}', '}', fixed_str)  # Remove trailing commas
    fixed_str = re.sub(r',\s*]', ']', fixed_str)  # Remove trailing commas in arrays
    json_candidates.append(fixed_str)
    
    # Try each candidate
    for candidate in json_candidates:
        try:
            data = json.loads(candidate)
            return CreditExplanationResponse(**data)
        except (json.JSONDecodeError, ValidationError):
            continue
    
    # Step 5: Handle truncated JSON by attempting completion
    if brace_count > 0 and end_idx == len(cleaned):
        # JSON was truncated, try to fix by closing braces
        json_str = cleaned[start_idx:]
        missing_braces = '"}' * brace_count  # Close strings and braces
        completed_json = json_str + missing_braces
        
        try:
            data = json.loads(completed_json)
            return CreditExplanationResponse(**data)
        except (json.JSONDecodeError, ValidationError):
            pass
    
    # Step 6: Last resort - try to extract fields manually with regex
    result = _extract_with_regex(cleaned)
    if result is None:
        # If all else fails, log and return None
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"JSON parsing failed completely. Response preview: {json_response[:200]}...")
    
    return result



