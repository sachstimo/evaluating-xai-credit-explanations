"""
Unified JudgeLLM interface.
Uses Gemini API directly for Google models, falls back to existing explanation LLMs.
"""

import logging
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pytz
from config.config import settings, initialize_llms
from ..llm_integration.extraction_models import EvaluationResult
from ..llm_integration.prompts import EVALUATOR_SYSTEM_PROMPT, EVALUATOR_USER_PROMPT

# LangChain imports for cleaner prompt handling
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Google Genai imports
try:
    from google.genai import types
except ImportError:
    types = None

logger = logging.getLogger(__name__)


class JudgeLLM:
    """Unified judge that uses Gemini API directly or falls back to existing LLMs."""
    
    def __init__(self, judge_name: Optional[str] = None):
        """Initialize judge - LangChain for single evaluations, direct API for batch."""
        if judge_name is None:
            judge_name = settings.get("default_judge", "gemini-2.5-flash")
        
        self.judge_name = judge_name
        
        # Initialize LangChain components for cleaner prompt handling
        self._init_langchain_components()
        
        # Initialize specific LangChain model for ALL single evaluations
        self.llm = self._init_single_langchain_model(judge_name)
        logger.info(f"✅ Using LangChain judge: {judge_name}")
        
        # Separate batch capability (Gemini only, no fallback)
        self.supports_batch = "gemini" in judge_name.lower()
        if self.supports_batch:
            try:
                self._init_gemini_client()
                logger.info(f"✅ Batch operations enabled for: {judge_name}")
            except Exception as e:
                logger.warning(f"Batch operations disabled: {e}")
                self.supports_batch = False

    def _init_langchain_components(self):
        """Initialize LangChain prompt templates and output parser."""
        # Import feature mapping
        from ..model_explainer.utils import FEATURE_NAME_MAPPING
        
        # Safely replace the feature mapping placeholder without using .format()
        feature_names_list = str(list(FEATURE_NAME_MAPPING.keys()))
        formatted_system_prompt = EVALUATOR_SYSTEM_PROMPT.replace("{FEATURE_NAME_MAPPING}", feature_names_list)
        
        # Create prompt templates
        self.system_prompt_template = SystemMessagePromptTemplate.from_template(formatted_system_prompt)
        self.user_prompt_template = HumanMessagePromptTemplate.from_template(EVALUATOR_USER_PROMPT)
        
        # Create chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_prompt_template,
            self.user_prompt_template
        ])
        
        # Initialize Pydantic output parser
        self.output_parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    def _init_single_langchain_model(self, model_name: str):
        """Initialize a single LangChain model by name."""
        # Get the specific model config
        model_config = settings["LLMS"].get(model_name)
        if not model_config or not model_config.get("enabled", False):
            available = [name for name, cfg in settings["LLMS"].items() if cfg.get("enabled", False)]
            raise ValueError(f"Judge model '{model_name}' not available. Available: {available}")
        
        # Initialize the specific model based on provider
        if model_config["provider"] == "ollama":
            from langchain_ollama import ChatOllama
            max_tokens = model_config.get("max_tokens", settings["LLM_MAX_TOKENS"])
            return ChatOllama(
                model=model_config["model"],
                temperature=model_config.get("temperature", settings["LLM_TEMPERATURE"]),
                num_predict=max_tokens
            )
            
        elif model_config["provider"] == "openai":
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(f"No OpenAI API key for {model_name}")
            return ChatOpenAI(
                model=model_config["model"],
                temperature=model_config.get("temperature", settings["LLM_TEMPERATURE"]),
                max_completion_tokens=model_config.get("max_tokens", settings["LLM_MAX_TOKENS"]),
                api_key=SecretStr(api_key)
            )
            
        elif model_config["provider"] == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(f"No Google API key for {model_name}")
            
            # Suppress Google gRPC warnings
            os.environ['GRPC_VERBOSITY'] = 'ERROR'
            os.environ['GLOG_minloglevel'] = '2'
            
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_config["model"],
                temperature=model_config.get("temperature", settings["LLM_TEMPERATURE"]),
                max_output_tokens=model_config.get("max_tokens", settings["LLM_MAX_TOKENS"]),
                google_api_key=api_key,
                max_retries=5
            )
            
        else:
            raise ValueError(f"Unsupported provider '{model_config['provider']}' for judge model '{model_name}'")

    def _init_gemini_client(self):
        """Initialize Gemini client for API access."""
        import google.genai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        # Use a proper Gemini model name for batch operations
        self.model_name = "models/gemini-2.5-flash"

    def _extract_prediction_context(self, prediction_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize prediction context data."""
        if 'prediction' in prediction_context:
            pred_data = prediction_context['prediction']
            prediction_outcome = "DECLINED" if pred_data.get('prediction', 0) == 1 else "APPROVED"
            input_data = pred_data.get('input_data', {})
        else:
            prediction_outcome = "APPROVED" if prediction_context.get('original_prediction', 0) == 0 else "DECLINED"
            input_data = prediction_context.get('input_data', {})
        
        return {
            'prediction_outcome': prediction_outcome,
            'age': input_data.get('age', 0),
            'monthly_income': input_data.get('MonthlyIncome', 0),
            'debt_ratio': input_data.get('DebtRatio', 0),
            'revolving_utilization': input_data.get('RevolvingUtilizationOfUnsecuredLines', 0),
            'number_of_open_credit_lines': input_data.get('NumberOfOpenCreditLinesAndLoans', 0),
            'number_30_59_late': input_data.get('NumberOfTime30-59DaysPastDueNotWorse', 0),
            'number_60_89_late': input_data.get('NumberOfTime60-89DaysPastDueNotWorse', 0),
            'number_90_days_late': input_data.get('NumberOfTimes90DaysLate', 0)
        }

    def _clean_explanation_text(self, explanation_text: str) -> str:
        """Extract clean explanation text from JSON or raw format."""
        if isinstance(explanation_text, str) and explanation_text.startswith('{'):
            try:
                parsed = json.loads(explanation_text)
                return parsed.get('consumer_explanation', explanation_text)
            except json.JSONDecodeError:
                pass
        return explanation_text

    def evaluate_single(
        self, 
        explanation_text: str, 
        prediction_context: Dict[str, Any],
        judge_model: Optional[str] = None,
        prediction_id: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """Single evaluation using LangChain.
        
        Args:
            explanation_text: The explanation to evaluate
            prediction_context: Context with prediction and input data
            judge_model: Specific model name (if different from current judge)
            prediction_id: Optional prediction ID for logging
        """
        # If requesting different model than current, create new instance
        if judge_model is not None and judge_model != self.judge_name:
            temp_judge = JudgeLLM(judge_name=judge_model)
            return temp_judge.evaluate_single(explanation_text, prediction_context, None, prediction_id)
        
        # Use current judge model
        return self._evaluate_with_langchain(explanation_text, prediction_context, prediction_id)

    def _evaluate_with_langchain(
        self,
        explanation_text: str,
        prediction_context: Dict[str, Any],
        prediction_id: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """Core LangChain evaluation logic."""
        # Prepare context and prompt
        context = self._extract_prediction_context(prediction_context)
        context['consumer_explanation'] = self._clean_explanation_text(explanation_text)
        context['format_instructions'] = self.output_parser.get_format_instructions()
        
        # Format prompt using LangChain template
        formatted_prompt = self.chat_prompt.format_prompt(**context)
        
        # Run evaluation
        start_time = time.time()
        response = self.llm.invoke(formatted_prompt.to_messages())
        eval_time = time.time() - start_time
        
        logger.debug(f"Judge evaluation completed in {eval_time:.1f}s")
        
        if not response or not hasattr(response, 'content') or not response.content:
            error_msg = f"Empty response from {self.judge_name}"
            if prediction_id:
                error_msg += f" for {prediction_id}"
            logger.error(error_msg)
            return None
        
        # Parse response
        try:
            from ..model_explainer.utils import preprocess_llm_json_response
            clean_content = preprocess_llm_json_response(response.content, self.judge_name)
            json_data = json.loads(clean_content)
            result = EvaluationResult(**json_data)
            logger.debug(f"Successfully parsed evaluation result")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse response from {self.judge_name}: {e}")
            return None

    def evaluate_llm_explanation(
        self, 
        explanation_text: str, 
        prediction_context: Dict[str, Any],
        prediction_id: Optional[str] = None
    ) -> Optional[EvaluationResult]:
        """Evaluate explanation using LangChain (unified approach)."""
        return self._evaluate_with_langchain(explanation_text, prediction_context, prediction_id)

        # === Gemini Batch Operations ===
    
    def _create_explicit_cache(self) -> Optional[str]:
        """Create explicit cache for system prompt following the docs pattern."""
        try:
            # Delete any existing cache with same display name
            caches = list(self.client.caches.list())
            for cache in caches:
                if (hasattr(cache, 'display_name') and 
                    cache.display_name == "evaluation-system-prompt"):
                    self.client.caches.delete(name=cache.name)
                    logger.info(f"🗑️ Deleted old cache: {cache.name}")
            
            # Create fresh cache using the correct format from docs
            cache = self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    display_name="evaluation-system-prompt",
                    system_instruction=EVALUATOR_SYSTEM_PROMPT,
                    ttl="86400s"  # 24 hours TTL (enough for large batch jobs)
                )
            )
            
            logger.info(f"✅ Created explicit cache: {cache.name}")
            return cache.name
            
        except Exception as e:
            logger.error(f"❌ Failed to create explicit cache: {e}")
            return None

    def extend_cache_ttl(self, cache_name: str, additional_hours: int = 24) -> bool:
        """Extend cache TTL to prevent expiration during long batch jobs."""
        try:
            # Update cache with extended TTL
            self.client.caches.patch(
                name=cache_name,
                updates={'ttl': f"{additional_hours * 3600}s"}
            )
            logger.info(f"✅ Extended cache {cache_name} TTL by {additional_hours} hours")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not extend cache TTL: {e}")
            return False

    def _load_explanations_from_dynamo(self, num_explanations: Optional[int] = None) -> List[Dict]:
        """Load explanations from DynamoDB with fallback to JSON."""
        try:
            from ..storage.dynamodb_client import DynamoDBClient
            
            logger.info("📥 Attempting to load explanations from DynamoDB...")
            dynamo_client = DynamoDBClient()
            explanations = dynamo_client.get_all_explanations(limit=num_explanations)
            
            if explanations:
                logger.info(f"✅ Loaded {len(explanations)} explanations from DynamoDB")
                return explanations
            else:
                logger.warning("⚠️ No explanations found in DynamoDB, falling back to JSON")
                
        except Exception as e:
            logger.warning(f"⚠️ DynamoDB load failed: {e}, falling back to JSON")
            # Fallback to JSON
            return self._load_explanations_from_json(num_explanations)
    
    def _load_explanations_from_json(self, num_explanations: Optional[int] = None) -> List[Dict]:
        """Load explanations from JSON file."""
        explanations_file = Path(settings["OUTPUT_FILEPATH"])
        logger.info(f"📂 Loading explanations from JSON: {explanations_file}")
        
        with open(explanations_file, 'r') as f:
            explanations_data = json.load(f)
        
        explanations = explanations_data.get('explanations', explanations_data)
        if num_explanations is not None:
            explanations = explanations[:num_explanations]
            
        logger.info(f"✅ Loaded {len(explanations)} explanations from JSON")
        return explanations

    def check_existing_batch_jobs(self) -> List[Dict[str, Any]]:
        """Check for existing batch jobs and their status."""
        if not self.supports_batch:
            return []
            
        try:
            # List all batch jobs
            batch_jobs = list(self.client.batches.list())
            
            # Filter for our evaluation jobs
            evaluation_jobs = []
            for job in batch_jobs:
                if hasattr(job, 'display_name') and 'credit-explanation-evaluation' in job.display_name:
                    evaluation_jobs.append({
                        'job_id': job.name,
                        'display_name': job.display_name,
                        'state': job.state.name,
                        'create_time': str(job.create_time) if job.create_time else None
                    })
            
            return evaluation_jobs
            
        except Exception as e:
            logger.error(f"❌ Failed to check existing batch jobs: {e}")
            return []

    def submit_batch_job(
        self, 
        predictions_file: Optional[Path] = None,
        num_explanations: Optional[int] = None,
        cache_system_prompt: bool = True,
        use_dynamo: bool = True,
        force_submit: bool = False
    ) -> Dict[str, Any]:
        """Submit batch evaluation job using Gemini API."""
        
        if not self.supports_batch:
            raise NotImplementedError(f"Batch operations only available with Gemini judge, currently using: {self.judge_name}")
        
        # Check for existing batch jobs
        if not force_submit:
            existing_jobs = self.check_existing_batch_jobs()
            running_jobs = [job for job in existing_jobs if job['state'] in ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING']]
            
            if running_jobs:
                logger.info(f"📊 Found {len(running_jobs)} existing batch job(s) still running:")
                for job in running_jobs:
                    logger.info(f"   - {job['display_name']} (ID: {job['job_id']}) - State: {job['state']}")
                    if job.get('create_time'):
                        logger.info(f"     Started: {job['create_time']}")
                logger.info("💡 Use force_submit=True to submit anyway, or wait for existing jobs to complete")
                return {
                    'status': 'blocked',
                    'message': f"Batch job submission blocked - {len(running_jobs)} job(s) still running",
                    'running_jobs': running_jobs,
                    'job_id': running_jobs[0]['job_id'],  # Use first running job for monitoring
                    'using_cache': False
                }
        
        logger.info("🚀 Starting Gemini API batch job submission...")
        
        # Handle zero explanations case gracefully
        if num_explanations == 0:
            logger.info("⏭️ Skipping batch job submission (num_explanations=0)")
            return {
                'status': 'skipped',
                'message': 'Batch job skipped - num_explanations set to 0',
                'request_count': 0,
                'using_cache': False
            }
        
        # Load explanations (DynamoDB first, then JSON fallback)
        if use_dynamo:
            explanations = self._load_explanations_from_dynamo(num_explanations)
        else:
            explanations = self._load_explanations_from_json(num_explanations)
        
        # Load predictions for context (explanations don't contain input data like age, income, etc.)
        if predictions_file is None:
            # Default to sampled predictions file since explanations were generated from it
            predictions_file = Path(settings["OUTPUT_FILEPATH"]).parent.parent / "predictions" / "prediction_results_sampled.json"
        
        logger.info(f"📂 Loading predictions context from {predictions_file}")
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
            predictions_lookup = {
                pred_data['prediction']['prediction_id']: pred_data 
                for pred_data in predictions_data['predictions'].values()
            }
        else:
            predictions_lookup = {p['prediction_id']: p for p in predictions_data}
        
        logger.info(f"📊 Loaded {len(explanations)} explanations and {len(predictions_lookup)} predictions")
        
        # Create explicit cache for system prompt if requested
        cache_name = None
        if cache_system_prompt:
            cache_name = self._create_explicit_cache()
        
        # Filter explanations that have matching predictions
        processed_explanations = []
        for explanation in explanations:
            pred_id = explanation.get('prediction_id')
            if pred_id in predictions_lookup:
                processed_explanations.append(explanation)
        
        if not processed_explanations:
            raise ValueError("No explanations found with matching predictions")
        
        logger.info(f"✅ Found {len(processed_explanations)} explanations with matching predictions")
        
        # Create batch job with descriptive name
        timestamp = int(time.time())
        job_name = f"credit-explanation-evaluation-{len(processed_explanations)}-explanations-{timestamp}"
        
        try:
            # Submit batch job using input file method
            logger.info(f"📤 Submitting batch job with {len(processed_explanations)} explanations...")
            
            from .utils import prepare_batch_jsonl
            
            # Prepare JSONL file
            temp_dir = Path("temp")
            jsonl_file = prepare_batch_jsonl(
                explanations=processed_explanations,  # Pass the successfully processed explanations
                predictions_lookup=predictions_lookup,
                output_dir=temp_dir,
                filename_prefix="batch_requests",
                cache_name=cache_name
            )
            
            logger.info(f"📁 JSONL file created: {jsonl_file}")
            
            # Upload file to Gemini API
            uploaded_file = self.client.files.upload(
                file=str(jsonl_file),
                config={"display_name": f"batch-requests-{timestamp}", "mime_type": "application/jsonl"}
            )
            
            logger.info(f"📤 File uploaded: {uploaded_file.name}")
            
            # Create batch job with uploaded file
            batch_params = {
                "model": self.model_name,
                "config": {"display_name": job_name},
                "src": uploaded_file.name
            }
            
            batch_job = self.client.batches.create(**batch_params)
            
            # Clean up temp file
            jsonl_file.unlink(missing_ok=True)
 
            logger.info(f"Context caching: {'✅ Explicit' if cache_name else '❌ Disabled'}")
            logger.info(f"Batch job with {len(processed_explanations)} explanations submitted successfully with job name: {job_name}")
 
            return {
                'job_id': batch_job.name,
                'job_name': job_name,
                'cache_name': cache_name,
                'request_count': len(processed_explanations),
                'timestamp': timestamp,
                'using_cache': cache_name is not None,
                'method': 'input_file'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to submit batch job: {e}")
            # Clean up cache if created
            if cache_name:
                try:
                    self.client.caches.delete(name=cache_name)
                    logger.info(f"🧹 Cleaned up explicit cache: {cache_name}")
                except:
                    pass
            raise

    def check_batch_status(self, job_id: str) -> Dict[str, Any]:
        """Check batch job status."""
        if not self.supports_batch:
            raise NotImplementedError(f"Batch operations only available with Gemini judge, currently using: {self.judge_name}")
            
        try:
            batch_job = self.client.batches.get(name=job_id)
            
            status_info = {
                'job_id': job_id,
                'state': batch_job.state.name,
                'create_time': str(batch_job.create_time) if batch_job.create_time else None,
                'update_time': str(batch_job.update_time) if batch_job.update_time else None,
                'request_count': getattr(batch_job, 'request_count', 0)
            }
            
            logger.info(f"📊 Batch job status: {batch_job.state.name}")
            
            if hasattr(batch_job, 'completion_stats') and batch_job.completion_stats:
                stats = batch_job.completion_stats
                status_info.update({
                    'completed_requests': getattr(stats, 'completed_request_count', 0),
                    'failed_requests': getattr(stats, 'failed_request_count', 0)
                })
                
                logger.info(f"   Completed: {getattr(stats, 'completed_request_count', 0)}")
                logger.info(f"   Failed: {getattr(stats, 'failed_request_count', 0)}")
            
            return status_info
            
        except Exception as e:
            logger.error(f"❌ Failed to check batch status: {e}")
            return {'job_id': job_id, 'state': 'ERROR', 'error': str(e)}

    def cancel_batch_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a batch job."""
        if not self.supports_batch:
            raise NotImplementedError(f"Batch operations only available with Gemini judge, currently using: {self.judge_name}")
            
        try:
            # First check if job exists
            batch_job = self.client.batches.get(name=job_id)
            original_state = batch_job.state.name
            
            logger.info(f"🛑 Cancelling batch job: {job_id} (current state: {original_state})")
            
            # Cancel the job
            self.client.batches.cancel(name=job_id)
            
            # Verify cancellation
            updated_job = self.client.batches.get(name=job_id)
            new_state = updated_job.state.name
            
            logger.info(f"✅ Batch job cancelled successfully! State changed: {original_state} → {new_state}")
            
            return {
                'job_id': job_id,
                'success': True,
                'original_state': original_state,
                'new_state': new_state
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel batch job {job_id}: {e}")
            return {
                'job_id': job_id,
                'success': False,
                'error': str(e)
            }

    def get_batch_results(self, job_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get and process batch results with token usage tracking."""
        
        if not self.supports_batch:
            raise NotImplementedError(f"Batch operations only available with Gemini judge, currently using: {self.judge_name}")
        
        job_id = job_info['job_id']
        cache_name = job_info.get('cache_name')
        
        logger.info(f"📥 Retrieving batch results for: {job_id}")
        
        try:
            batch_job = self.client.batches.get(name=job_id)
            
            if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
                logger.warning(f"⚠️ Job not completed yet. State: {batch_job.state.name}")
                return []
            
            # Get results from output file (input file method)
            # Try multiple possible output file attributes
            output_file = None
            if hasattr(batch_job, 'output_uri') and batch_job.output_uri:
                output_file = batch_job.output_uri
                logger.info(f"📁 Found output_uri: {output_file}")
            elif hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'file_name'):
                output_file = batch_job.dest.file_name
                logger.info(f"📁 Found dest.file_name: {output_file}")
            elif hasattr(batch_job, 'output_file') and batch_job.output_file:
                output_file = batch_job.output_file
                logger.info(f"📁 Found output_file: {output_file}")
            
            if not output_file:
                logger.error("❌ No output file found in batch job")
                logger.info(f"🔍 Available attributes: {[attr for attr in dir(batch_job) if not attr.startswith('_')]}")
                return []
            
            # Download and parse results file
            file_content = self.client.files.download(file=output_file)
            responses = [json.loads(line) for line in file_content.decode().strip().split('\n')]
            logger.info(f"📥 Downloaded {len(responses)} responses from output file")
            processed_results = []
            
            logger.info(f"🔄 Processing {len(responses)} batch responses...")
            
            for i, response in enumerate(responses):
                try:
                    # Extract response content from the JSON structure
                    if 'response' in response and 'candidates' in response['response']:
                        candidates = response['response']['candidates']
                        if candidates and 'content' in candidates[0]:
                            content = candidates[0]['content']
                            if 'parts' in content and content['parts']:
                                response_text = content['parts'][0]['text']
                                
                                # Use robust JSON parsing with fallbacks
                                try:
                                    # First try: direct JSON parse
                                    evaluation_data = json.loads(response_text.strip())
                                except json.JSONDecodeError:
                                    # Second try: clean up common issues and retry
                                    cleaned_text = response_text.strip()
                                    
                                    # Remove markdown code blocks
                                    if '```' in cleaned_text:
                                        import re
                                        # Extract JSON from markdown
                                        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL)
                                        if match:
                                            cleaned_text = match.group(1)
                                        else:
                                            # Remove any remaining markdown
                                            cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
                                    
                                    # Remove common JSON issues
                                    import re
                                    cleaned_text = re.sub(r',\s*}', '}', cleaned_text)  # Trailing commas
                                    cleaned_text = re.sub(r',\s*]', ']', cleaned_text)   # Trailing commas in arrays
                                    
                                    try:
                                        evaluation_data = json.loads(cleaned_text)
                                    except json.JSONDecodeError:
                                        # Final attempt: use the existing robust parser
                                        from ..model_explainer.utils import preprocess_llm_json_response
                                        clean_content = preprocess_llm_json_response(response_text, "gemini-2.5-flash")
                                        evaluation_data = json.loads(clean_content)
                                
                                # Handle missing technical_jargon_check field
                                if 'regulatory_compliance' in evaluation_data:
                                    reg_compliance = evaluation_data['regulatory_compliance']
                                    if 'technical_jargon_check' not in reg_compliance:
                                        reg_compliance['technical_jargon_check'] = {'score': 0}
                                
                                evaluation_result = EvaluationResult(**evaluation_data)
                                
                                processed_results.append({
                                    'evaluation_result': evaluation_result.dict(),
                                    'metadata': response.get('custom_metadata', {}),
                                    'batch_index': i
                                })
                
                except Exception as e:
                    logger.error(f"Failed to parse batch response {i}: {e}")
                    continue
            
            logger.info(f"✅ Successfully processed {len(processed_results)} batch results")
            
            # Clean up explicit cache if it was created
            if cache_name:
                try:
                    self.client.caches.delete(name=cache_name)
                    logger.info(f"🧹 Cleaned up explicit cache: {cache_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean up explicit cache: {e}")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve batch results: {e}")
            return []

    def list_batch_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent batch jobs."""
        if not self.supports_batch:
            raise NotImplementedError(f"Batch operations only available with Gemini judge, currently using: {self.judge_name}")
            
        try:
            jobs = []
            for batch_job in self.client.batches.list():
                jobs.append({
                    'job_id': batch_job.name,
                    'display_name': getattr(batch_job, 'display_name', 'Unknown'),
                    'state': batch_job.state.name,
                    'create_time': str(batch_job.create_time) if batch_job.create_time else None
                })
            
            # Sort by create time (newest first) and limit
            jobs.sort(key=lambda x: x['create_time'] or '', reverse=True)
            return jobs[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to list batch jobs: {e}")
            return []

    # Backward compatibility methods
    def extract_cemat_scores(self, result: EvaluationResult) -> Dict[str, Any]:
        """Extract CEMAT scores for backward compatibility."""
        if not result or not result.cemat_evaluation:
            return {}
        
        cemat = result.cemat_evaluation
        return {
            'understandability_items': cemat.understandability_items,
            'actionability_items': cemat.actionability_items,
            'understandability_score': sum(v for v in cemat.understandability_items.values() if v != 'N/A'),
            'actionability_score': sum(v for v in cemat.actionability_items.values() if v != 'N/A')
        }

    def extract_regulatory_scores(self, result: EvaluationResult) -> Dict[str, Any]:
        """Extract regulatory compliance scores."""
        if not result or not result.regulatory_compliance:
            return {}
        
        reg = result.regulatory_compliance
        return {
            'principal_reason_identification': {'score': reg.principal_reason_identification.score},
            'individual_specific_content': {'score': reg.individual_specific_content.score},
            'technical_jargon_check': {'score': reg.technical_jargon_check.score}
        }

    def extract_counterfactual_changes(self, result: EvaluationResult) -> Dict[str, Any]:
        """Extract counterfactual changes for verification."""
        if not result or not result.counterfactual_extraction:
            return {}
        
        changes_dict = {}
        for change in result.counterfactual_extraction.changes:
            if change.target_value is not None:
                changes_dict[change.feature_name] = {'to': change.target_value}
        
        return {'counterfactual_changes': changes_dict}