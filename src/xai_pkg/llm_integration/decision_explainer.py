from typing import Dict, List, Any, Optional, Tuple
import time
import logging
import os
import re
import sys
import json
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import SecretStr

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.callbacks.manager import get_openai_callback

from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
try:
    from dotenv import load_dotenv
    # Load from project config directory
    config_path = os.path.join(os.path.dirname(__file__), '../../../config/.env')
    if os.path.exists(config_path):
        load_dotenv(config_path)
except ImportError:
    pass

config_dir = os.path.join(os.path.dirname(__file__), '../../../config')
sys.path.insert(0, os.path.dirname(config_dir))
from config.config import settings

# Import custom prompts for credit decision explanations
from .prompts import CREDIT_SYSTEM_PROMPT, CREDIT_APPLICATION_PROMPT
from .extraction_models import CreditExplanationResponse, parse_llm_json_response

# Configure logging for thesis documentation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Container for LLM explanation results with metadata for evaluation."""
    prediction_id: str
    llm_name: str
    explanation_text: str
    processing_time: float
    explanation_id: str = None
    current_explanation: int = 0
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Set default values for optional fields."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.explanation_id:
            self.explanation_id = f'{self.prediction_id}_exp_{self.current_explanation + 1}'

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExplanationResult to a dictionary for JSON serialization."""
        return {
            "prediction_id": self.prediction_id,
            "explanation_id": self.explanation_id,
            "llm_name": self.llm_name,
            "explanation_text": self.explanation_text,
            "processing_time": self.processing_time,
            "token_usage": self.token_usage,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
class CreditDecisionExplainer:
    """
    Multi-LLM explanation generator for comparative thesis analysis.
    Supports OpenAI, Ollama, and Mistral for academic evaluation.
    """

    def __init__(self, llms: Optional[Dict[str, Any]] = None):
        # Initialize prompt templates
        self.system_prompt_template = SystemMessagePromptTemplate.from_template(CREDIT_SYSTEM_PROMPT)
        self.user_prompt_template = HumanMessagePromptTemplate.from_template(CREDIT_APPLICATION_PROMPT)
        
        # Initialize Pydantic output parser
        self.output_parser = PydanticOutputParser(pydantic_object=CreditExplanationResponse)
        
        # Initialize LLMs
        if llms is not None:
            self.llms = llms
        else:
            from config.config import initialize_llms
            logger.info("Initializing LLMs from configuration...")
            self.llms = initialize_llms()

    def get_rate_limit_delay(self, llm_name: str) -> float:
        """Get appropriate request delay based on LLM provider."""
        request_delays = settings.get('REQUEST_DELAYS', {
            'openai': 2.0,
            'google': 3.0, 
            'ollama': 0.0
        })
        return request_delays.get(llm_name, 0.0)
    
    def _get_token_usage(self, llm, messages, llm_name: str) -> Tuple[Any, Optional[Dict[str, int]]]:
        """Get token usage for different LLM providers and return response."""
        if 'openai' in llm_name.lower():
            with get_openai_callback() as cb:
                response = llm.invoke(messages)
                token_usage = {
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_tokens': cb.total_tokens,
                    'total_cost': cb.total_cost
                }
                return response, token_usage
        elif 'gemini' in llm_name.lower():
            response = llm.invoke(messages)
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                    'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                    'total_tokens': getattr(usage, 'total_token_count', 0),
                    'total_cost': None
                }
                return response, token_usage
            return response, None
        else:
            response = llm.invoke(messages)
            return response, None
    
    def _parse_response(self, content: str, llm_name: str, prediction_id: str) -> str:
        """Parse LLM response using robust extraction methods."""
        try:
            parsed_response = parse_llm_json_response(content)
            if parsed_response:
                clean_explanation = parsed_response.consumer_explanation.replace('\n', ' ').replace('\r', ' ').strip()
                return json.dumps({
                    "consumer_explanation": clean_explanation,
                    "analysis": {"top_features": parsed_response.analysis.top_features}
                }, ensure_ascii=False)
        except Exception as parse_error:
            logger.warning(f"Structured parsing failed for {llm_name} prediction {prediction_id}: {parse_error}")
        
        # Fallback: return raw content
        return content if content else ""
    
    def generate_explanations_concurrent(self, 
                                       system_vars: Dict[str, Any], 
                                       user_vars: Dict[str, Any],
                                       llm_names: List[str],
                                       max_workers: int = 3) -> Dict[str, ExplanationResult]:
        """
        Generate explanations concurrently using multiple LLMs.
        
        Args:
            system_vars: Variables for system prompt
            user_vars: Variables for user prompt
            llm_names: List of LLM names to use
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dict mapping LLM names to ExplanationResults
        """
        results = {}
        available_llms = [name for name in llm_names if name in self.llms]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_llm = {
                executor.submit(self.generate_explanation, system_vars, user_vars, llm_name): llm_name
                for llm_name in available_llms
            }
            
            for future in as_completed(future_to_llm):
                llm_name = future_to_llm[future]
                try:
                    result = future.result()
                    results[llm_name] = result
                    
                    # Apply rate limiting
                    delay = self.get_rate_limit_delay(llm_name)
                    if delay > 0.1:
                        time.sleep(delay)
                        
                except Exception as e:
                    logger.error(f"Error in concurrent explanation generation for {llm_name}: {e}")
                    results[llm_name] = ExplanationResult(
                        prediction_id=user_vars.get('prediction_id', 'unknown'),
                        llm_name=llm_name,
                        explanation_text="",
                        processing_time=0,
                        error=str(e)
                    )
        
        return results
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_explanation(self, 
                           system_vars: Dict[str, Any], 
                           user_vars: Dict[str, Any],
                           llm_name: str) -> ExplanationResult:
        """
        Generate explanation using specified LLM.
        
        Args:
            system_vars: Variables for system prompt
            user_vars: Variables for user prompt  
            llm_name: Name of LLM to use
            
        Returns:
            ExplanationResult with explanation and metadata
        """
        
        if llm_name not in self.llms:
            return ExplanationResult(
                prediction_id=user_vars.get('prediction_id', 'unknown'),
                llm_name=llm_name,
                explanation_text="",
                processing_time=0,
                error=f"LLM {llm_name} not available"
            )
        
        start_time = time.time()
        prediction_id = user_vars.get('prediction_id', 'unknown')
        
        try:
            # Create and format chat prompt
            chat_prompt = ChatPromptTemplate.from_messages([
                self.system_prompt_template,
                ("human", CREDIT_APPLICATION_PROMPT + "\n\n{format_instructions}")
            ])
            
            formatted_prompt = chat_prompt.format_prompt(
                **system_vars, 
                **user_vars,
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # Generate explanation with token tracking
            llm = self.llms[llm_name]
            response, token_usage = self._get_token_usage(llm, formatted_prompt.to_messages(), llm_name)
            
            # Parse response using robust extraction
            explanation_text = self._parse_response(response.content, llm_name, prediction_id)
            
            processing_time = time.time() - start_time
            
            return ExplanationResult(
                prediction_id=prediction_id,
                llm_name=llm_name,
                explanation_text=explanation_text,
                processing_time=processing_time,
                token_usage=token_usage
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error generating explanation with {llm_name} for prediction {prediction_id}: {e}")
            
            return ExplanationResult(
                prediction_id=prediction_id,
                llm_name=llm_name,
                explanation_text="",
                processing_time=processing_time,
                error=str(e)
            )
    
    def generate_single_explanation(
        self,
        prediction_id: str, 
        prediction: Dict[str, Any],
        llm_name: str,
        regeneration_num: int,
        system_vars: Dict[str, Any],
        dynamo_client=None,
        dynamo_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a single explanation for a prediction and optionally store to DynamoDB."""
        
        from ..model_explainer.utils import format_prediction_variables
        
        explanation_id = f"{prediction_id}_exp_{regeneration_num}_{llm_name}"
        
        try:
            # Format variables and generate explanation
            user_vars = format_prediction_variables(prediction)
            user_vars['prediction_id'] = prediction_id
            result = self.generate_explanation(system_vars, user_vars, llm_name)
            
            # Build explanation record
            explanation = {
                "prediction_id": prediction_id,
                "cluster_id": prediction.get('cluster_id'),
                "timestamp": prediction.get('timestamp'),
                "regeneration_number": regeneration_num,
                "explanation_id": explanation_id,
                "llm_name": llm_name,
                "explanation_text": result.explanation_text,
                "processing_time": result.processing_time,
                "token_usage": result.token_usage,
                "error": result.error,
                "explanation_timestamp": result.timestamp
            }
            
            # Store to DynamoDB immediately if client is provided
            if dynamo_client and dynamo_settings:
                self.store_explanation(explanation, dynamo_client, dynamo_settings)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation {explanation_id}: {e}")
            error_explanation = {
                "prediction_id": prediction_id,
                "cluster_id": prediction.get('cluster_id'),
                "timestamp": prediction.get('timestamp'),
                "regeneration_number": regeneration_num,
                "explanation_id": explanation_id,
                "llm_name": llm_name,
                "explanation_text": "",
                "processing_time": 0.0,
                "token_usage": None,
                "error": str(e),
                "explanation_timestamp": datetime.now().isoformat()
            }
            
            # Store error explanation to DynamoDB if client is provided
            if dynamo_client and dynamo_settings:
                self.store_explanation(error_explanation, dynamo_client, dynamo_settings)
            
            return error_explanation
    
    def store_explanation(self, explanation: Dict[str, Any], dynamo_client, dynamo_settings: Dict[str, Any]) -> bool:
        """Store explanation to DynamoDB with error handling."""
        try:
            table_name = dynamo_settings.get("explanation_table", "llm_explanations")
            dynamo_client.store(explanation, table_name, overwrite=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to store explanation {explanation.get('explanation_id')} to DynamoDB: {e}")
            return False