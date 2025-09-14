
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


def get_preprocessor(
    X_train: pd.DataFrame, 
    verbose: bool = False,
    impute_strategy: tuple = ('constant', 0),
    ) -> Pipeline:
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    remaining_features = X_train.columns.difference(numeric_features).tolist()
    
    if verbose:
        print(f"Numeric features identified: {numeric_features}")
        print(f"Remaining features (not numeric): {remaining_features if remaining_features else 'None'}")
    
    if not numeric_features:
        raise ValueError("No numeric features found in the training data.")
    
    steps = [
        ('imputer', SimpleImputer(strategy=impute_strategy[0], fill_value=impute_strategy[1])),
        ('scaler', RobustScaler()), 
    ]   

    numeric_transformer = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_regular', numeric_transformer, numeric_features),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    preprocessor.set_output(transform='pandas')

    pre_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    return pre_pipeline

def trim_extreme_values(data: pd.DataFrame, trim_dict: dict) -> pd.DataFrame:
    """Clean extreme values before other preprocessing"""
    df_clean = data.copy()

    # Replace infinities with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # Cap features based on trim_dict
    for col, (lower, upper) in dict(trim_dict).items():
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    return df_clean

def winsorize_late_vars(df, late_vars=None):
    """
    Replace 96/98 values in late payment variables with the median of valid values.
    Args:
        df (pd.DataFrame): DataFrame to process.
        late_vars (list): List of column names to process.
    Returns:
        pd.DataFrame: DataFrame with winsorized late variables.
    """
    if late_vars is None:
        late_vars = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse"
        ]
    df = df.copy()
    for col in late_vars:
        if col in df.columns:
            valid_mask = ~df[col].isin([96, 98])
            median_val = df.loc[valid_mask, col].median()
            df.loc[df[col].isin([96, 98]), col] = median_val
    return df

def log_transform_features(df: pd.DataFrame, features: list, drop_features: bool = False) -> pd.DataFrame:
    """
    Apply log transformation - set invalid values to NaN, let imputer handle them.

    Parameters:
    df (pd.DataFrame): DataFrame containing the features to be transformed.
    features (list): List of feature names to apply log transformation.
    drop_features (bool): If True, drop the original features after transformation.

    Returns:
    pd.DataFrame: DataFrame with log-transformed features.
    """
    df_copy = df.copy()
    
    for col in features:
        if col in df_copy.columns:
            # Replace zeros and negative values with NaN before log transform
            temp_values = df_copy[col].copy()
            temp_values[temp_values <= 0] = np.nan
            
            # Apply log transform
            df_copy[col + "_log"] = np.log1p(temp_values)
            
            if drop_features:
                df_copy.drop(columns=[col], inplace=True)
    
    return df_copy


def impute_income(
    df: pd.DataFrame,
    income_col: str = 'MonthlyIncome',
    add_flag: bool = False,
    treat_zero_as_missing: bool = True,
    median_value: float = None,
    ) -> pd.DataFrame:
    """
    Impute income based on the median of the income column.
    
    Args:
        df: DataFrame to process
        income_col: Name of the income column
        add_flag: Whether to add an imputation flag
        treat_zero_as_missing: Whether to treat zero income as missing data
    """
    df_copy = df.copy()
    df_copy = df_copy.replace(["NA", "N/A", "n/a", "N/A", "n/a"], np.nan)
    
    # Treat zero income as missing if specified
    if treat_zero_as_missing:
        df_copy.loc[df_copy[income_col] == 0, income_col] = np.nan
    
    # Add float flag for imputed income BEFORE imputation (convert bool to float for SHAP compatibility)
    if add_flag:
        df_copy["imputed_income"] = df_copy[income_col].isna().astype(float)
    
    # Use provided median or calculate from data (with fallback for empty data)
    if median_value is not None:
        impute_value = median_value
    else:
        # Calculate median, with fallback for empty/all-NaN cases
        non_null_values = df_copy[income_col].dropna()
        if len(non_null_values) > 0:
            impute_value = non_null_values.median()
        else:
            # Fallback value for cases where all values are NaN
            impute_value = 3000.0  # Reasonable default income
    
    df_copy[income_col] = df_copy[income_col].fillna(impute_value)
    return df_copy

def recalculate_debt_ratio(
    df: pd.DataFrame,
    debt_ratio_col: str = 'DebtRatio',
    income_col: str = 'MonthlyIncome',
    ) -> pd.DataFrame:
    """
    Recalculate debt ratio based on the imputed income only for rows with imputed income.
    """
    df_copy = df.copy()

    if "imputed_income" in df_copy.columns:
        # Use np.where for efficient vectorized operation
        imputed_mask = df_copy["imputed_income"] == True
        # Only recalculate for imputed rows, else keep original
        df_copy[debt_ratio_col] = np.where(
            imputed_mask,
            df_copy[debt_ratio_col] / df_copy[income_col],
            df_copy[debt_ratio_col]
        )

    # Clean any infinite values that might have been created
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

    return df_copy