import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union
import logging
from xai_pkg.model_explainer.base_explainer import BaseExplainer
from xai_pkg.model_explainer.utils import convert_logit_to_probability

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(
            self, 
            base_explainer: BaseExplainer,
            n_background_samples: Union[int, str] = 'full',
            model_training_data: Optional[pd.DataFrame] = None,
            baseline: Optional[float] = None,
            warning_threshold: float = 0.3,
            suppress_shap_warnings: bool = False
            ):
        
        """
        Initialize the SHAP explainer.
        Args:
            base_explainer: Shared BaseExplainer instance
            n_background_samples: Number of background samples to use for SHAP
            warning_threshold: Threshold for logging warnings if SHAP and model predictions differ significantly
            suppress_shap_warnings: Whether to suppress SHAP warnings
        """
        self.base_explainer = base_explainer
        # Import from base_explainer
        self.predictor = base_explainer.predictor
        self.model_pipeline = base_explainer.model_pipeline
        self.preprocessor = base_explainer.preprocessor
        self.model = base_explainer.model
        self.feature_names = base_explainer.feature_names
        self.model_type = base_explainer.model_type
        self.outcome_name = base_explainer.outcome_name
        # Add model training data
        self.model_training_data = model_training_data if model_training_data is not None else base_explainer.model_training_data

        # Initialize SHAP explainer parameters
        self.n_background_samples = n_background_samples
        self.suppress_shap_warnings = suppress_shap_warnings
        self.warning_threshold = warning_threshold
        self.baseline = baseline

        # Preprocess background data to match model expectations
        if self.model_training_data is not None:
            background_df = self.model_training_data.copy()
            
            # If target column is present, drop it for background data
            target_col = getattr(self.predictor, 'target', None)
            if target_col and target_col in background_df.columns:
                background_df = background_df.drop(columns=[target_col])


            processed_background = self.predictor._preprocess_input(
                background_df,
                keep_target=False
            )

            # Apply sklearn preprocessing (imputation, scaling) to match model input format
            processed_background = self.preprocessor.transform(processed_background)
            
            # Sample if needed
            if self.n_background_samples != 'full':
                n_samples = int(self.n_background_samples)
                if len(processed_background) > n_samples:
                    self.background_data = processed_background.sample(n=n_samples, random_state=42)
                else:
                    self.background_data = processed_background
            else:
                self.background_data = processed_background
        else:
            self.background_data = None

        # Initialize SHAP explainer
        self.explainer = None
        self._initialize_shap_explainer()


    def _initialize_shap_explainer(self):
        """
        Initialize the SHAP explainer.
        """
        try:
            def model_predict_proba(X_train):
                """Expects preprocessed data, returns probabilities"""
                
                # Ensure column order matches what the model expects
                if hasattr(self.model, 'feature_names_in_'):
                    expected_features = self.model.feature_names_in_
                    X_train = X_train.reindex(columns=expected_features)

                # Use the model directly since data is already preprocessed
                # This avoids double preprocessing issues
                proba = self.model.predict_proba(X_train)
                return proba[:, 1]
            
            self.explainer = shap.Explainer(
                model_predict_proba,
                self.background_data,
                link=shap.links.logit
            )
            
            logger.info(f"SHAP Explainer initialized successfully with logit link")
                    
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            import traceback
            traceback.print_exc()
            self.explainer = None
    
    def explain_single(self, 
                      instance_data: pd.DataFrame,
                      prediction_result: Optional[Dict] = None,
                      verbose = False
                      ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single instance.
        
        Args:
            instance_data: RAW input data (will be preprocessed internally)
            prediction_result: Optional prediction result from predictor
            
        Returns:
            Dict containing waterfall explanation data
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer was not properly initialized.")
        
        # Apply manual preprocessing first
        processed_instance = self.predictor._preprocess_input(instance_data, keep_target=False)
        
        # Apply sklearn preprocessing (imputation, scaling) to match model input format
        processed_instance = self.preprocessor.transform(processed_instance)

        # Ensure column order matches what the model expects
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
            processed_instance = processed_instance.reindex(columns=expected_features)
        
        # Get SHAP explanation (WITH logit link, values are in log-odds space)
        shap_explanation = self.explainer(processed_instance)
        shap_values = np.array(shap_explanation.values[0]).flatten()
        baseline_logit = shap_explanation.base_values[0]
        
        # Store for visualization
        self._last_shap_explanation = shap_explanation
        
        # Convert to contributions dictionary
        contributions = {}
        if self.feature_names:
            for i, feature_name in enumerate(self.feature_names):
                contributions[feature_name] = float(shap_values[i])
        else:
            for i in range(len(shap_values)):
                contributions[f"feature_{i}"] = float(shap_values[i])
        
        # Calculate predicted probability
        predicted_logit = baseline_logit + sum(contributions.values())
        predicted_proba = convert_logit_to_probability(predicted_logit)

        if verbose:
            logger.info(f"SHAP explanation for instance: {contributions}")
        
        # Get original model prediction
        actual_prediction = prediction_result.get('probability', 0.0) if prediction_result else 0.0
        
        explanation = {
            'baseline': float(baseline_logit),
            'contributions': contributions,
            'model_probability': float(actual_prediction),
            'shap_sum_logit': float(predicted_logit),
            'shap_probability': float(predicted_proba),
            'difference': float(abs(predicted_proba - actual_prediction))
        }
        
        # Log warning if large difference
        if explanation['difference'] > self.warning_threshold and not self.suppress_shap_warnings:
            logger.warning(
                f"Difference between SHAP ({predicted_proba:.4f}) and model ({actual_prediction:.4f}): "
                f"{explanation['difference']:.4f}"
            )
        
        return explanation


    def explain_batch(self, 
                     batch_data: pd.DataFrame,
                     prediction_results: Optional[Dict[str, Dict]] = None,
                     verbose=False
                     ) -> Dict[str, Dict[str, Any]]:
        """
        Generate SHAP explanations for a batch of instances.
        
        Args:
            batch_data: RAW input data (will be preprocessed internally)
            prediction_results: Optional dict of prediction results keyed by prediction_id
            
        Returns:
            Dict of explanations keyed by prediction_id
        """

        if self.explainer is None:
            raise RuntimeError("SHAP explainer was not properly initialized.")
        
        # Apply manual preprocessing first
        processed_batch = self.predictor._preprocess_input(batch_data, keep_target=False)
        
        # Apply sklearn preprocessing (imputation, scaling) to match model input format
        processed_batch = self.preprocessor.transform(processed_batch)

        # Ensure column order matches what the model expects
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
            processed_batch = processed_batch.reindex(columns=expected_features)

        shap_explanations = self.explainer(processed_batch)
        explanations = {}
        
        # Get prediction IDs in the same order as they were generated
        prediction_ids_ordered = list(prediction_results.keys()) if prediction_results else None

        for i, shap_exp in enumerate(shap_explanations):
            shap_values = np.array(shap_exp.values).flatten()
            baseline_logit = shap_exp.base_values

            # Use feature names if available
            contributions = {}
            if self.feature_names:
                for j, feature_name in enumerate(self.feature_names):
                    contributions[feature_name] = float(shap_values[j])
            else:
                for j in range(len(shap_values)):
                    contributions[f"feature_{j}"] = float(shap_values[j])

            predicted_logit = baseline_logit + sum(contributions.values())
            predicted_proba = convert_logit_to_probability(predicted_logit)

            # Get prediction_id if available
            prediction_id = None
            actual_prediction = 0.0
            if prediction_results and prediction_ids_ordered and i < len(prediction_ids_ordered):
                prediction_id = prediction_ids_ordered[i]
                actual_prediction = prediction_results[prediction_id].get('probability', 0.0)
            else:
                prediction_id = str(i)

            explanation = {
                'baseline': float(baseline_logit),
                'contributions': contributions,
                'model_probability': float(actual_prediction),
                'shap_sum_logit': float(predicted_logit),
                'shap_probability': float(predicted_proba),
                'difference': float(abs(predicted_proba - actual_prediction))
            }

            if explanation['difference'] > self.warning_threshold and not self.suppress_shap_warnings:
                logger.warning(
                    f"Difference between SHAP ({predicted_proba:.4f}) and model ({actual_prediction:.4f}): "
                    f"{explanation['difference']:.4f}"
                )

            explanations[prediction_id] = explanation

        return explanations

    
    def get_top_features(self, explanation: Dict[str, Any], n_features: int = 5) -> Dict[str, Any]:
        """Get the top N features by absolute contribution."""
        contributions = explanation['contributions']
        
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        positive_contributors = [(k, v) for k, v in sorted_contributions if v > 0]
        negative_contributors = [(k, v) for k, v in sorted_contributions if v < 0]
        
        return {
            'top_positive': positive_contributors[:n_features],
            'top_negative': negative_contributors[:n_features],
            'top_overall': sorted_contributions[:n_features]
        }
    
    def print_waterfall(self, explanation: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Create and display a visual SHAP waterfall plot."""
        if hasattr(self, '_last_shap_explanation'):
            shap.plots.waterfall(
                self._last_shap_explanation[0] if hasattr(self._last_shap_explanation, '__getitem__') else self._last_shap_explanation,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Waterfall plot saved to: {save_path}")
            
            plt.show()
        else:
            logger.error("No SHAP explanation available for visualization")