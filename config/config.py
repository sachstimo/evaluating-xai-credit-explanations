"""Simple configuration loader for XAI Credit Decisions project."""
import os
import json
from pathlib import Path
from pydantic import SecretStr
import logging

# Try to load dotenv if available, but don't fail if it's not installed
try:
    from dotenv import load_dotenv
    # Load from config/.env file
    config_env_file = Path(__file__).parent / ".env"
    load_dotenv(config_env_file)
except ImportError:
    # dotenv not available, just use os.getenv
    pass

# Load settings from JSON config
config_file = Path(__file__).parent / "config.json"
with open(config_file) as f:
    settings = json.load(f)

# Add secrets from environment (never write these to file)
settings["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
settings["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Add computed paths
PROJECT_ROOT = Path(__file__).parent.parent
settings["PROJECT_ROOT"] = str(PROJECT_ROOT)
settings["DATA_DIR"] = str(PROJECT_ROOT / "data")
settings["OUTPUT_DIR"] = str(PROJECT_ROOT / "output")
settings["CONFIG_DIR"] = str(PROJECT_ROOT / "config")

# Convert relative paths to absolute
settings["OUTPUT_FILEPATH"] = str(PROJECT_ROOT / "output" / "llm_explanations" / "explanations.json")
settings["PREDICTIONS_PATH"] = str(PROJECT_ROOT / "output" / "predictions" / "prediction_results.json")
settings["METADATA_PATH"] = str(PROJECT_ROOT / "output" / "models" / "model_metadata.json")

# NEVER write settings back to JSON - keep secrets out of files

def get_absolute_path(relative_path: str) -> Path:
    """Convert relative path to absolute from project root."""
    return PROJECT_ROOT / relative_path


def initialize_llms():
    """Initialize LLMs from configuration."""
    
    logger = logging.getLogger(__name__)
    llms = {}
    
    for name, config in settings["LLMS"].items():
        if not config.get("enabled", False):
            continue
            
        try:
            if config["provider"] == "ollama":
                from langchain_ollama import ChatOllama
                max_tokens = config.get("max_tokens", settings["LLM_MAX_TOKENS"])
                # Special handling for gpt-oss model which seems to have token limiting issues
                if "gpt-oss" in config["model"]:
                    llms[name] = ChatOllama(
                        model=config["model"],
                        temperature=config.get("temperature", settings["LLM_TEMPERATURE"]),
                        num_predict=max_tokens
                    )
                else:
                    llms[name] = ChatOllama(
                        model=config["model"],
                        temperature=config.get("temperature", settings["LLM_TEMPERATURE"]),
                        num_predict=max_tokens
                    )
                logger.info(f"Initialized {name} via Ollama with model_kwargs num_predict={max_tokens}")
                
            elif config["provider"] == "openai":
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning(f"No OpenAI API key for {name}")
                    continue
                    
                llms[name] = ChatOpenAI(
                    model=config["model"],
                    temperature=config.get("temperature", settings["LLM_TEMPERATURE"]),
                    max_completion_tokens=config.get("max_tokens", settings["LLM_MAX_TOKENS"]),
                    api_key=SecretStr(api_key)
                )
                logger.info(f"Initialized {name} via OpenAI")
                
            elif config["provider"] == "google":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    logger.warning(f"No Google API key for {name}")
                    continue
                    
                try:
                    # Suppress Google gRPC warnings
                    os.environ['GRPC_VERBOSITY'] = 'ERROR'
                    os.environ['GLOG_minloglevel'] = '2'
                    
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    
                    llms[name] = ChatGoogleGenerativeAI(
                        model=config["model"],
                        temperature=config.get("temperature", settings["LLM_TEMPERATURE"]),
                        max_output_tokens=config.get("max_tokens", settings["LLM_MAX_TOKENS"]),
                        google_api_key=api_key,
                        max_retries=5,  # Increase retries for rate limiting
                        # Removed response_mime_type - may be causing empty responses
                    )
                    logger.info(f"Initialized {name} via Google Gemini")
                    
                except ImportError:
                    logger.warning(f"Google Generative AI dependencies not available for {name}")
                    logger.warning("Install with: uv add langchain-google-genai")
                    continue
                
            elif config["provider"] == "huggingface":
                try:
                    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
                    from transformers import pipeline
                    import torch
                    
                    # Determine device and settings based on hardware
                    if torch.backends.mps.is_available():
                        # Apple Silicon - use MPS
                        device = "mps"
                        torch_dtype = torch.float16
                        device_map = None
                        logger.info(f"Using Apple Silicon MPS for {name}")
                    elif torch.cuda.is_available():
                        # NVIDIA GPU
                        device = "auto"
                        torch_dtype = "auto"
                        device_map = "auto"
                        logger.info(f"Using CUDA GPU for {name}")
                    else:
                        # CPU fallback
                        device = "cpu"
                        torch_dtype = torch.float32
                        device_map = None
                        logger.info(f"Using CPU for {name} (will be slower)")
                    
                    # Create pipeline with appropriate settings
                    hf_pipeline = pipeline(
                        "text-generation",
                        model=config["model"],
                        torch_dtype=torch_dtype,
                        device=device,
                        device_map=device_map,
                        max_new_tokens=config.get("max_tokens", settings["LLM_MAX_TOKENS"]),
                        trust_remote_code=True  # Required for gpt-oss
                    )
                    
                    # Wrap in LangChain
                    hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)
                    llms[name] = ChatHuggingFace(llm=hf_llm)
                    logger.info(f"Initialized {name} via Hugging Face")
                    
                except ImportError:
                    logger.warning(f"Hugging Face dependencies not available for {name}")
                    logger.warning("Install with: uv add langchain-huggingface transformers torch")
                    continue
                except Exception as e:
                    if "GPU" in str(e) or "MXFP4" in str(e):
                        logger.warning(f"GPU-specific error for {name}: {e}")
                        logger.warning(f"Try disabling {name} or use a different model variant")
                    else:
                        logger.warning(f"Failed to load {name}: {e}")
                    continue
                
        except Exception as e:
            logger.warning(f"Failed to initialize {name}: {e}")
    
    return llms


