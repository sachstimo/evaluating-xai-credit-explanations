import pandas as pd
import numpy as np
from typing import Optional
import logging


logger = logging.getLogger(__name__)

class BaseExplainer:
    """
    Base class for model explainers. Handles standardized loading and processing of training data.

    Args:
        predictor: Fitted CreditPredictor instance
        model_training_data: Optional training data to use for explanation
        outcome_name: Optional outcome name to use for explanation
    """
    def __init__(self, predictor, model_training_data: Optional[pd.DataFrame] = None, outcome_name: Optional[str] = None):
        self.predictor = predictor
        self.model_pipeline = predictor.model_pipeline
        self.model_training_data = model_training_data if model_training_data is not None else None
        self.feature_names = getattr(predictor, 'feature_names', None)
        self.model_type = getattr(predictor, 'model_type', None)
        self.outcome_name = outcome_name


        # Extract model from pipeline
        if hasattr(self.model_pipeline, 'named_steps'):
            # It's a sklearn Pipeline - take the last step as model
            self.model = list(self.model_pipeline.named_steps.values())[-1]
        else:
            # Not a pipeline, just a model
            self.model = self.model_pipeline

        # Extract preprocessor from pipeline if available
        if hasattr(self.model_pipeline, 'named_steps'):
            self.preprocessor = self.model_pipeline.named_steps.get('preprocessor', None)
        else:
            self.preprocessor = None
        
        # Update feature names based on what the model actually expects after preprocessing
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_.tolist()
        elif self.feature_names is None:
            self.feature_names = self._extract_feature_names()
        
        logger.info(f"BaseExplainer initialized with {self.model_type} model. Features: {len(self.feature_names) if self.feature_names else 'unknown'}")

    def _extract_feature_names(self) -> Optional[list]:
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