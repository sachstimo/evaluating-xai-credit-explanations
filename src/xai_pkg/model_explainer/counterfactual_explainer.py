import dice_ml
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple
from scipy import stats
import io
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from xai_pkg.model_explainer.base_explainer import BaseExplainer
from xai_pkg.model_explainer.utils import unlog

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CounterfactualExplainer:
    def __init__(
            self, 
            base_explainer: BaseExplainer,
            train_df_with_target: pd.DataFrame,
            total_CFs: int = 1,
            immutable_features: Optional[list[str]] = ['age'],
            desired_class: str = "opposite"
            ):
        
        """
        Initialize the Counterfactual explainer.
        Args:
            base_explainer: Shared BaseExplainer instance
            train_df_with_target: Training data including target column (required by DiCE)
            total_CFs: Number of counterfactuals to generate per instance
            desired_class: Desired class for counterfactuals ("opposite" or 0/1)
        """
        self.base_explainer = base_explainer
        self.predictor = base_explainer.predictor
        self.model_pipeline = base_explainer.model_pipeline
        self.model_training_data = train_df_with_target
        self.feature_names = base_explainer.feature_names if base_explainer.feature_names else train_df_with_target.columns.tolist()
        self.model_type = base_explainer.model_type
        self.outcome_name = base_explainer.outcome_name
        self.preprocessor = base_explainer.preprocessor
        self.immutable_features = immutable_features or []
        
        if immutable_features:
            self.features_to_vary = [f for f in self.feature_names if f not in immutable_features and f != self.outcome_name]
        else:
            self.features_to_vary = [f for f in self.feature_names if f != self.outcome_name]

        self.total_CFs = max(1, total_CFs)
        self.desired_class = desired_class

        # Check if model_training_data is provided
        if self.model_training_data is None:
            raise ValueError("model_training_data is required for CounterfactualExplainer")
        
        # The training data has already been manually preprocessed
        # Apply the sklearn preprocessing (from the pipeline) to prepare it for DiCE
        self.model_training_data = train_df_with_target.copy()
        
        # Extract features (without target) for preprocessing
        features_df = self.model_training_data.drop(columns=[self.outcome_name])
        target_series = self.model_training_data[self.outcome_name]
        
        # Apply preprocessing using the preprocessor from base_explainer
        if self.preprocessor is not None:
            preprocessed_features = self.preprocessor.transform(features_df)
            # Add target back
            preprocessed_features[self.outcome_name] = target_series.values
            self.model_training_data = preprocessed_features
        
        # Convert boolean columns to integers for DiCE compatibility
        for col in self.model_training_data.columns:
            if self.model_training_data[col].dtype == bool:
                self.model_training_data[col] = self.model_training_data[col].astype(int)

        # Identify feature types - all continuous for simplicity
        self.continuous_features = [col for col in self.model_training_data.columns if col != self.outcome_name]
        
        # Setup DiCE with simplified configuration
        try:
            self.dice_data = dice_ml.Data(
                dataframe=self.model_training_data,
                continuous_features=self.continuous_features,
                outcome_name=self.outcome_name
            )

            # Pass only the classifier (not the full pipeline) to DiCE since data is already preprocessed
            classifier = base_explainer.model  # This is extracted by BaseExplainer
            self.dice_model = dice_ml.Model(model=classifier, backend="sklearn")
            
            # Use genetic method for reliable and reasonably fast counterfactuals
            try:
                self.dice = dice_ml.Dice(self.dice_data, self.dice_model, method="genetic")
                self.method = "genetic"
            except Exception:
                logger.warning("Genetic method failed, falling back to random method")
                self.dice = dice_ml.Dice(self.dice_data, self.dice_model, method="random")
                self.method = "random"
                    
        except Exception as e:
            logger.error(f"Failed to initialize DiCE: {e}")
            raise

        # Calculate feature percentiles from training data for distance calculations
        self._calculate_feature_percentiles()
        
        logger.info(f"CounterfactualExplainer initialized with {self.total_CFs} CFs, method: {self.method}")

    def _calculate_feature_percentiles(self) -> None:
        """Calculate percentiles for each feature in the training data for distance calculations."""
        self.feature_percentiles = {}
        
        for feature in self.continuous_features:
            if feature in self.model_training_data.columns:
                feature_values = self.model_training_data[feature].dropna()
                if len(feature_values) > 0:
                    self.feature_percentiles[feature] = feature_values
    
    
    def _calculate_percentile_distance(self, from_value: float, to_value: float, feature: str) -> tuple:
        """Calculate percentile distance for a feature change."""
        if feature not in self.feature_percentiles:
            return 50.0, 50.0
        
        feature_values = self.feature_percentiles[feature]
        from_percentile = stats.percentileofscore(feature_values, from_value, kind='rank')
        to_percentile = stats.percentileofscore(feature_values, to_value, kind='rank')
        
        return from_percentile, to_percentile
    
    def _calculate_cf_distance_score(self, cf_dict: dict, original_values: dict) -> dict:
        """Calculate distance score for a counterfactual."""
        percentile_distances = []
        
        for feature, cf_value in cf_dict.items():
            if (feature != self.outcome_name and 
                feature in original_values and 
                cf_value != original_values[feature]):
                
                original_value = original_values[feature]
                from_pct, to_pct = self._calculate_percentile_distance(original_value, cf_value, feature)
                
                # Calculate the percentile change magnitude
                percentile_change = abs(to_pct - from_pct) if from_pct is not None and to_pct is not None else 0
                percentile_distances.append(percentile_change)
        
        # Overall distance metric - lower is better (closer to typical values)
        avg_percentile_change = np.mean(percentile_distances) if percentile_distances else 50.0
        
        return {
            'avg_percentile_change': avg_percentile_change,
            'num_features_changed': len(percentile_distances)
        }

    def _convert_preprocessed_to_raw(self, feature: str, preprocessed_value: float, 
                                   original_preprocessed: float, original_raw: float) -> float:
        """
        Convert DiCE's preprocessed counterfactual value back to raw data space.
        
        This handles the two-step preprocessing:
        1. Manual preprocessing (imputation, log transforms, trimming)
        2. Pipeline preprocessing (scaling/normalization)
        
        Args:
            feature: Feature name
            preprocessed_value: DiCE's counterfactual value (fully preprocessed)
            original_preprocessed: Original preprocessed value for this feature
            original_raw: Original raw value for this feature
            
        Returns:
            Reasonable raw value that approximates the counterfactual
        """
        try:            
            # Use pipeline inverse transform if available
            if (hasattr(self.base_explainer, 'preprocessor') and 
                self.base_explainer.preprocessor and 
                hasattr(self.base_explainer.preprocessor, 'inverse_transform')):
                
                try:
                    # Create a copy of original preprocessed values
                    inverse_input = np.array([[original_preprocessed] for _ in self.feature_names])
                    # Replace the target feature with counterfactual value
                    feature_idx = self.feature_names.index(feature) if feature in self.feature_names else None
                    if feature_idx is not None:
                        inverse_input[feature_idx, 0] = preprocessed_value
                    
                    # Apply inverse transform to get back to "manually preprocessed" space
                    inverse_result = self.base_explainer.preprocessor.inverse_transform(inverse_input.T)
                    raw_value = inverse_result[0, feature_idx] if feature_idx is not None else preprocessed_value
                    
                    # Handle special cases where manual preprocessing significantly changes values
                    if feature == 'imputed_income':
                        # This is a flag feature, should be 0 or 1
                        return max(0, min(1, round(raw_value)))
                    
                    elif 'log' in feature.lower() or feature in getattr(self.predictor, 'log_scale_features', []):
                        # Unlog
                        return unlog(raw_value)

                    return raw_value
                    

                except Exception as e:
                    logger.debug(f"Pipeline inverse transform failed for {feature}: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to convert preprocessed value to raw for {feature}: {e}")
            return original_raw

    def explain_single(
        self, 
        instance_data: Union[Dict[str, Any], pd.DataFrame],
        verbose: bool = False,
        show_only_changes: bool = False
        ) -> Optional[list[Dict[str, Any]]]:
    
        """
        Generate counterfactual explanation for a single instance.
        Args:
            instance_data: Input data for the instance
            verbose: Whether to show verbose logging
            show_only_changes: Whether to return only features that changed
            suppress_output: Whether to suppress DiCE output (set to False for multithreading)
        Returns:
            List of counterfactuals or None if no counterfactuals could be generated
        """
        try:
            # Convert to DataFrame if needed
            instance_df = instance_data if isinstance(instance_data, pd.DataFrame) else pd.DataFrame([instance_data])
            
            # Store the original raw values for comparison later
            original_raw_values = instance_df.iloc[0].to_dict()
            
            # Get column order from dice_data to ensure alignment
            dice_features = self.continuous_features
                
            # Apply manual preprocessing first
            instance_df = self.predictor._preprocess_input(instance_df)
            
            # Convert boolean columns to integers for DiCE compatibility
            bool_cols = instance_df.select_dtypes(include='bool').columns
            instance_df[bool_cols] = instance_df[bool_cols].astype(int)
            
            # Apply the sklearn pipeline preprocessing to match training data format
            if self.preprocessor is not None:
                if verbose:
                    logger.info(f"Applying pipeline preprocessing to instance data")
                instance_processed_df = self.preprocessor.transform(instance_df)
            else:
                instance_processed_df = instance_df

            # Ensure it's a DataFrame with correct column names
            if not isinstance(instance_processed_df, pd.DataFrame):
                instance_processed_df = pd.DataFrame(instance_processed_df, columns=dice_features)
            
            # Generate counterfactuals with suppressed output and actionability-focused parameters
            import io
            import contextlib
            
            # Suppress all DiCE output including progress bars
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                # Standard parameters - DiCE validates flips internally
                cf_params = {
                    'total_CFs': self.total_CFs,
                    'desired_class': self.desired_class,
                    'features_to_vary': [str(f) for f in self.features_to_vary],
                    'verbose': False
                }
                
                # Enforce immutable features using permitted_range (more reliable than features_to_vary alone)
                # Note: We need to use the preprocessed values since DiCE works with preprocessed data
                if self.immutable_features:
                    permitted_range = {}
                    current_values = instance_processed_df.iloc[0].to_dict()
                    for feature in self.immutable_features:
                        if feature in current_values:
                            # Lock the feature to its current preprocessed value by setting range to [value, value]
                            current_val = current_values[feature]
                            permitted_range[feature] = [current_val, current_val]
                    
                    if permitted_range:
                        cf_params['permitted_range'] = permitted_range
                        if verbose:
                            logger.info(f"Applied permitted_range constraints (preprocessed values): {permitted_range}")
                
                # Debug logging to see what's being passed to DiCE
                if verbose:
                    logger.info(f"Immutable features: {self.immutable_features}")
                    logger.info(f"Features to vary being passed to DiCE: {cf_params['features_to_vary']}")
                    logger.info(f"Features NOT allowed to vary: {[f for f in self.continuous_features if f not in self.features_to_vary]}")
                
                # Add method-specific parameters optimized for both generation and flip success
                if self.method == "genetic":
                    # Prioritize flip success while maintaining generation rate
                    cf_params.update({
                        'proximity_weight': 0.2,  # Allow dramatic changes for reliable flips
                        'sparsity_weight': 0.3,   # Low sparsity weight - flip success priority
                        'diversity_weight': 1.0,  # High diversity to find working solutions
                        'categorical_penalty': 0.05,  # Low penalty to allow necessary changes
                        'yloss_type': 'hinge_loss',  # Use hinge loss for better convergence
                        'posthoc_sparsity_param': 0.2,  # Very low post-hoc sparsity
                        'posthoc_sparsity_algorithm': 'linear',  # Faster than binary
                    })
                elif self.method == "random":
                    # For random method: prioritize flip success
                    cf_params.update({
                        'proximity_weight': 0.2,  # Allow dramatic changes
                        'posthoc_sparsity_param': 0.2,  # Very low sparsity constraint
                    })
                
                cf = self.dice.generate_counterfactuals(
                    instance_processed_df, 
                    **cf_params
                )

            # Check for valid counterfactuals
            if not cf.cf_examples_list or len(cf.cf_examples_list) == 0:
                if verbose:
                    logger.warning("No counterfactuals generated")
                return None
            
            cf_examples = cf.cf_examples_list[0].final_cfs_df
            
            if cf_examples is None or len(cf_examples) == 0:
                if verbose:
                    logger.warning("Generated counterfactuals are empty")
                return None
            
            if verbose:
                logger.info(f"Generated {len(cf_examples)} counterfactuals")

            # Convert to dictionary format
            result = cf_examples.to_dict(orient="records")
            
            # Sort by distance (closer to typical values first)
            if len(result) > 0:
                original_processed_values = instance_processed_df.iloc[0].to_dict()
                
                # Calculate distance for each counterfactual and sort
                cf_with_distances = []
                for cf_dict in result:
                    distance_info = self._calculate_cf_distance_score(cf_dict, original_processed_values)
                    cf_with_distances.append((cf_dict, distance_info['avg_percentile_change']))
                
                # Sort by distance (ascending = closer to typical values first)
                cf_with_distances.sort(key=lambda x: x[1])
                result = [cf_dict for cf_dict, _ in cf_with_distances]
            
            # If true, filter to show only changed features
            if show_only_changes and len(result) > 0:
                # Use preprocessed values to determine which features changed
                original_processed_values = instance_processed_df.iloc[0].to_dict()
                filtered_result = []
                
                for cf_dict in result:
                    # Only include features that actually need to change (exclude unchanged and outcome)
                    changes_needed = {}
                    
                    for feature, cf_value in cf_dict.items():
                        if (feature != self.outcome_name and 
                            feature in original_processed_values):
                            
                            # Check if feature should be treated as immutable
                            is_immutable = feature in self.immutable_features
                            
                            # For immutable features, check if value changed (with floating point tolerance)
                            if is_immutable:
                                original_cf_value = original_processed_values[feature]
                                value_changed = abs(cf_value - original_cf_value) > 1e-10
                            else:
                                value_changed = cf_value != original_processed_values[feature]
                            
                            # Only include if the feature actually changed
                            if value_changed:
                                # Original and counterfactual values in their respective spaces
                                original_raw = original_raw_values.get(feature, None)
                                original_processed = original_processed_values[feature]
                                cf_processed = cf_value
                                
                                # Calculate percentiles FIRST (in preprocessed space for consistency)
                                from_pct, to_pct = self._calculate_percentile_distance(original_processed, cf_processed, feature)
                                percentile_change = abs(to_pct - from_pct) if from_pct is not None and to_pct is not None else None
                                
                                # Convert counterfactual to display format (raw space)
                                if is_immutable:
                                    cf_display = original_raw  # Keep original for immutable features
                                else:
                                    # Convert from preprocessed space to raw space for display
                                    if abs(cf_processed - original_processed) < 1e-6:
                                        cf_display = original_raw  # No meaningful change
                                    else:
                                        # Apply proportional change
                                        if abs(original_processed) > 1e-10:
                                            ratio = cf_processed / original_processed
                                            cf_display = original_raw * max(0.1, min(10.0, ratio))
                                        else:
                                            # Handle near-zero original values
                                            cf_display = original_raw + (cf_processed - original_processed)
                                        
                                        # Apply feature-specific bounds
                                        if feature == 'age':
                                            cf_display = max(18, min(100, cf_display))
                                        elif feature == 'MonthlyIncome':
                                            cf_display = max(0, cf_display)
                                        elif feature.startswith('Number'):
                                            cf_display = max(0, round(cf_display))
                                        elif feature in ['DebtRatio', 'RevolvingUtilizationOfUnsecuredLines']:
                                            cf_display = max(0, cf_display)
                                
                                # Apply training data bounds if available
                                if hasattr(self.predictor, 'model_metadata') and self.predictor.model_metadata:
                                    trimmed_values = self.predictor.model_metadata.get('trimmed_values', {})
                                    if feature in trimmed_values:
                                        min_val, max_val = trimmed_values[feature]
                                        cf_display = max(min_val, min(max_val, cf_display))
                                
                                # Calculate difference in display space
                                diff = cf_display - original_raw if isinstance(cf_display, (int, float)) and isinstance(original_raw, (int, float)) else None
                                
                                # Skip if the values are effectively the same (no real change)
                                if isinstance(diff, (int, float)) and abs(diff) < 1e-10:
                                    continue
                                
                                changes_needed[feature] = {
                                    "from": original_raw,
                                    "to": cf_display,
                                    "diff": diff,
                                    "from_percentile": from_pct,
                                    "to_percentile": to_pct,
                                    "percentile_change": percentile_change
                                }
                    
                    if changes_needed:
                        filtered_result.append(changes_needed)
                
                return filtered_result if filtered_result else result
            
            return result
            
        except Exception as e:
            if verbose:
                logger.error(f"Error generating counterfactuals: {e}")
            return None

    def _explain_single_with_timeout(self, instance_data, verbose=False, show_only_changes=False, timeout=30):
        """Generate counterfactual with timeout protection."""
        def target():
            return self.explain_single(instance_data, verbose=verbose, show_only_changes=show_only_changes)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(target)
            try:
                return future.result(timeout=timeout)
            except Exception:
                return None

    def explain_batch(
        self, 
        instances_data: pd.DataFrame,
        prediction_ids: list = None,
        verbose: bool = False,
        show_only_changes: bool = False,
        batch_size: int = 100,
        timeout_per_batch: int = 30,
        show_progress: bool = True
        ) -> dict:
        """Generate counterfactual explanations using individual processing with timeout protection."""
        
        if prediction_ids is None:
            prediction_ids = [f"pred_{i}" for i in range(len(instances_data))]
        
        results = {}
        timeout_per_instance = max(5, timeout_per_batch // 10)  # At least 5 seconds per instance
        
        logger.info(f"Processing {len(instances_data)} instances individually with {timeout_per_instance}s timeout per instance")
        
        # Process each instance individually with optional progress bar
        if show_progress:
            with tqdm(total=len(instances_data), desc="CF Generation") as pbar:
                for i, pred_id in enumerate(prediction_ids):
                    instance = instances_data.iloc[i:i+1].copy()
                    
                    try:
                        cf_result = self._explain_single_with_timeout(
                            instance, 
                            verbose=verbose, 
                            show_only_changes=show_only_changes,
                            timeout=timeout_per_instance
                        )
                        results[pred_id] = cf_result
                        
                        if cf_result is not None:
                            pbar.set_postfix({"Success": f"{sum(1 for r in results.values() if r is not None)}/{len(results)}"})
                        else:
                            if verbose:
                                logger.warning(f"CF generation failed/timeout for {pred_id}")
                            
                    except Exception as e:
                        if verbose:
                            logger.warning(f"CF generation error for {pred_id}: {e}")
                        results[pred_id] = None
                    
                    pbar.update(1)
        else:
            # Process without progress bar
            for i, pred_id in enumerate(prediction_ids):
                instance = instances_data.iloc[i:i+1].copy()
                
                try:
                    cf_result = self._explain_single_with_timeout(
                        instance, 
                        verbose=verbose, 
                        show_only_changes=show_only_changes,
                        timeout=timeout_per_instance
                    )
                    results[pred_id] = cf_result
                    
                    if cf_result is None and verbose:
                        logger.warning(f"CF generation failed/timeout for {pred_id}")
                        
                except Exception as e:
                    if verbose:
                        logger.warning(f"CF generation error for {pred_id}: {e}")
                    results[pred_id] = None
        
        # Log final completion
        total_successful = sum(1 for result in results.values() if result is not None)
        logger.info(f"🏁 CF generation complete: {total_successful}/{len(instances_data)} instances successful")
        
        return results
