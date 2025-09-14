import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
import joblib
import json
import logging
import uuid
from datetime import datetime
from xai_pkg.model_training.preprocessing import log_transform_features, trim_extreme_values, winsorize_late_vars, impute_income, recalculate_debt_ratio
from xai_pkg.model_explainer.utils import add_original_values_to_dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditPredictor:
    """
    Basic predictor class that wraps a trained ML pipeline and provides
    a clean interface for making predictions that can be consumed by
    explainer classes and LLMs.
    """
    
    def __init__(self, 
                 model_pipeline,
                 model_metadata: dict):

        """
        Initialize the predictor with a trained model pipeline.
        
        Args:
            model_pipeline: Trained sklearn pipeline or model
            model_type: Type of model (for metadata)
            model_version: Version of the model (for tracking)
            feature_names: List of feature names (if not available in pipeline)
        """
        self.model_pipeline = model_pipeline
        self.model_preprocessor = model_pipeline.named_steps['preprocessor'] if 'preprocessor' in model_pipeline.named_steps else None
        self.model_metadata = model_metadata  # Store full metadata for access by explainers
        self.model_type = model_metadata.get('model_type', "unknown")
        self.model_version = model_metadata.get('model_version', "unknown")
        self.feature_names = model_metadata.get('used_features')  # Use correct key from metadata
        self.log_scale_features = model_metadata.get('log_scale_features', [])
        # Use training-time caps if available
        self.trimmed_features = model_metadata.get('trimmed_values', {})
        self.dropped_columns = model_metadata.get('dropped_columns', [])
        self.target = model_metadata.get('target', None)
        
        # Get training descriptive statistics for consistent preprocessing
        self.descriptive_stats = model_metadata.get('descriptive_statistics', {})
        self.income_median = self.descriptive_stats.get('MonthlyIncome', {}).get('median', 5400.0)  # Fallback value

        # Try to extract feature names from pipeline
        if self.feature_names is None:
            self.feature_names = self._extract_feature_names()
        
        logger.info(f"CreditPredictor initialized with {self.model_type} model")

    def _preprocess_input(
            self, 
            input_data: Union[pd.DataFrame, Dict, pd.Series, np.ndarray],
            keep_target: bool = False
            ) -> pd.DataFrame:
        """
        Preprocess input data to match the expected format (DataFrame row) of the model.
        This includes both manual preprocessing steps AND sklearn pipeline preprocessing.
        """
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.copy()
        elif isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            input_data = pd.DataFrame([input_data]) 
        elif isinstance(input_data, np.ndarray):
            input_data = pd.DataFrame(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Drop columns
        if self.dropped_columns:
            if keep_target:
                drop_cols = [col for col in self.dropped_columns if col != self.target]
            else:
                drop_cols = self.dropped_columns
            input_data.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Critical preprocessing steps - fail fast if any of these fail
        try:
            # Winsorize late vars
            input_data = winsorize_late_vars(input_data)
        except Exception as e:
            logger.error(f"Failed to winsorize late variables: {e}")
            raise RuntimeError(f"Critical preprocessing step failed: winsorize_late_vars - {e}")
        
        try:
            # Impute income using stored training median
            input_data = impute_income(
                input_data, 
                add_flag=True,
                treat_zero_as_missing=True,
                median_value=self.income_median
                )
        except Exception as e:
            logger.error(f"Failed to impute income: {e}")
            raise RuntimeError(f"Critical preprocessing step failed: impute_income - {e}")
        
        try:
            # Recalculate debt ratio
            input_data = recalculate_debt_ratio(input_data)
        except Exception as e:
            logger.error(f"Failed to recalculate debt ratio: {e}")
            raise RuntimeError(f"Critical preprocessing step failed: recalculate_debt_ratio - {e}")
        
        try:
            # Trim extreme values
            input_data = trim_extreme_values(input_data, trim_dict=self.trimmed_features)
        except Exception as e:
            logger.error(f"Failed to trim extreme values: {e}")
            raise RuntimeError(f"Critical preprocessing step failed: trim_extreme_values - {e}")

        try:
            # Log transform features (optional step)
            input_data = log_transform_features(input_data, self.log_scale_features, drop_features=True)
        except Exception as e:
            logger.warning(f"Failed to apply log transformations (continuing without): {e}")
            # Log transform is optional, so we continue without it

        return input_data

        
    
    def _extract_feature_names(self) -> Optional[List[str]]:
        """
        Attempt to extract feature names from the pipeline
        """
        try:
            # Try to get feature names from the pipeline
            if hasattr(self.model_pipeline, 'feature_names_in_'):
                return self.model_pipeline.feature_names_in_.tolist()
            
            # Try to get from preprocessing step if it's a pipeline
            if hasattr(self.model_pipeline, 'named_steps'):
                for step_name, step in self.model_pipeline.named_steps.items():
                    if hasattr(step, 'get_feature_names_out'):
                        return step.get_feature_names_out().tolist()
            
            return None
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            return None
    
    def predict_single(self, 
                      input_data: Union[Dict[str, Any], pd.Series, np.ndarray, pd.DataFrame],
                      verbose: bool = False) -> dict:
        """
        Make a prediction for a single instance.
        
        Args:
            input_data: Input data as dict, pandas Series, or numpy array
            prediction_id: Optional ID for tracking this prediction. If None, generates automatically.
            
        Returns:
            PredictionResult: Structured prediction result
        """
        try:
            # Generate prediction ID if not provided
            prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
            
            if verbose:
                logger.info(f"Prediction ID: {prediction_id}")
                logger.info(f"Input data: {input_data}")
                logger.info(f"Number of features: {len(self.feature_names) if self.feature_names else 'unknown'}")
                logger.info(f"Log scale features: {self.log_scale_features}")
                
            # Preprocess input data
            input_data = self._preprocess_input(input_data, keep_target= False)

            if verbose:
                logger.info(f"Successfully preprocessed input data")

            # Make prediction
            prediction = self.model_pipeline.predict(input_data)[0]
            proba = self.model_pipeline.predict_proba(input_data)[0]
            probability = float(proba[1])

            # Convert input_data to dict format for storage (includes unlogged values)
            input_dict = self._convert_input_for_results_dict(input_data)

            prediction_result = {
                "prediction_id": prediction_id,
                "prediction": int(prediction),
                "probability": probability,
                "input_data": input_dict,
                "timestamp": datetime.now().isoformat()
            }

            if verbose:
                logger.info(f"Prediction result: {prediction_result['prediction']}, Probability: {probability}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


    def predict_batch(self, 
                     input_data: pd.DataFrame
                     ) -> Dict[str, dict]:
        """
        Make predictions for a batch of instances.
        
        Args:
            input_data: Input data as pandas DataFrame
        
        Returns:
            Dict of prediction results keyed by prediction_id
        """
        results = {}
        # Preprocess all input data at once
        processed_data = self._preprocess_input(input_data, keep_target=False)
        # Predict
        predictions = self.model_pipeline.predict(processed_data)
        probabilities = self.model_pipeline.predict_proba(processed_data)[:, 1]
        # Iterate and build results
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
            input_dict = self._convert_input_for_results_dict(processed_data.iloc[i])
            results[prediction_id] = {
                "prediction_id": prediction_id,
                "prediction": int(pred),
                "probability": float(proba),
                "input_data": input_dict,
                "timestamp": datetime.now().isoformat()
            }
        return results
    
    
    def _convert_input_for_results_dict(self, input_data: Union[Dict, pd.Series, np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Convert input data to dictionary format and add original unlogged values for storage in prediction output JSON.
        """
        # First convert to basic dictionary format
        if isinstance(input_data, dict):
            result = input_data
        elif isinstance(input_data, pd.Series):
            result = input_data.to_dict()
        elif isinstance(input_data, pd.DataFrame):
            # For DataFrame, take the first row and convert to dict
            if len(input_data) > 0:
                result = input_data.iloc[0].to_dict()
            else:
                result = {}
        elif isinstance(input_data, np.ndarray):
            if self.feature_names:
                result = dict(zip(self.feature_names, input_data.flatten()))
            else:
                result = {f"feature_{i}": val for i, val in enumerate(input_data.flatten())}
        else:
            result = {"input": str(input_data)}
        
        # Convert pd.NA and np.nan to None for JSON serialization
        cleaned_result = {}
        for key, value in result.items():
            if pd.isna(value):
                cleaned_result[key] = None
            else:
                cleaned_result[key] = value
        
        return add_original_values_to_dict(
            cleaned_result, 
            self.log_scale_features, 
            keep_log_features=False
        )
    
    @classmethod
    def load_model(
        cls, 
        model_filepath: str = "../output/models/best_model.pkl",
        model_metadata_filepath: str = "../output/models/model_metadata.json"
        ) -> 'CreditPredictor':
        """
        Load a saved model from disk.
        """
        model_pipeline = joblib.load(model_filepath)
        model_metadata = json.load(open(model_metadata_filepath))
        
        predictor = cls(
            model_pipeline=model_pipeline,
            model_metadata=model_metadata
        )
        
        logger.info(f"Model loaded from {model_filepath}")
        return predictor