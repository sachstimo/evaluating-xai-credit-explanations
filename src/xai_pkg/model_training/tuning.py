import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def tune_hyperparameters(
        X_train, 
        y_train, 
        preprocessor : Pipeline, 
        models: list = [], 
        n_trials=20, 
        cv_folds=5, 
        scoring_metric: str = 'f1', 
        obj_direction: str = 'maximize',
        random_state: int = 42, 
        verbose = True):
    
    # Suppress Optuna warnings
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='optuna')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    
    """
    Tune hyperparameters using Optuna for loan dataset.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
    
    Returns:
        Best parameters and trained model
    """

    if models is None:
        models = [
            'LogisticRegression', 
            'RandomForest', 
            'LightGBM'
        ]

    def objective(trial):
        # Define model and hyperparameters based on model type
        classifier_name = trial.suggest_categorical("classifier", models)

        # Generate a trial-specific random seed
        trial_seed = random_state + trial.number

        if classifier_name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'max_iter': trial.suggest_int('max_iter', 1000, 2000),
                'random_state': trial_seed,
                'tol': trial.suggest_float('tol', 1e-8, 1e-4, log=True),
                'fit_intercept': True,
                'class_weight': 'balanced',  
                'n_jobs': 1 
            }

            if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                params['solver'] = 'liblinear'
            elif params['penalty'] == 'elasticnet':
                params['solver'] = 'saga'
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
            
            model = LogisticRegression(**params)

        elif classifier_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': True,
                'class_weight': 'balanced', 
                'random_state': trial_seed,
                'oob_score': True, 
                'n_jobs': -1,
                'max_samples': trial.suggest_float('max_samples', 0.8, 0.95),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            }
            model = RandomForestClassifier(**params)

        elif classifier_name == 'LightGBM':
            params = {
                'objective': 'binary',
                'num_leaves': trial.suggest_int('num_leaves', 8, 24),
                'max_depth': trial.suggest_int('max_depth', 4, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.8, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.95),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'class_weight': 'balanced',
                'random_state': trial_seed,
                'verbosity': -1,
                'force_col_wise': True,
                'boosting_type': 'gbdt'  # Remove DART to reduce complexity
            }
            model = LGBMClassifier(**params)

        elif classifier_name == 'XGBoost':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'subsample': trial.suggest_float('subsample', 0.8, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.95),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.8, 0.95),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
                'random_state': trial_seed,
                'verbosity': 0,
                'tree_method': 'hist',
                'grow_policy': 'depthwise'  # Remove lossguide to reduce complexity
            }
            model = XGBClassifier(**params)

        else:
            raise ValueError(f"Unsupported classifier: {classifier_name}")
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=trial_seed)
        
        try:
            score = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv,
                scoring=scoring_metric,
                n_jobs=1, 
                verbose=False
            )
            return score.mean()
        
        except Exception as e:
            # Log the error and return a poor score to prune this trial
            print(f"Trial {trial.number} failed with error: {e}")
            return float('-inf') if obj_direction == 'maximize' else float('inf')
    
    # Create study with optimized sampler and pruner
    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    
    study = optuna.create_study(
        direction=obj_direction,
        sampler=sampler,
        pruner=pruner
    )
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=verbose,
        callbacks=[lambda study, trial: study.set_user_attr("random_state", random_state)]
    )
        
    # Train final model with best parameters
    best_params = study.best_params
    best_trial = study.best_trial

    # Remove the 'classifier' key before passing to model constructor
    model_params = {k: v for k, v in best_params.items() if k != 'classifier'}
    
    # Set consistent random state for the final model
    final_random_state = random_state + 999  # Use a different seed than trials

    if 'random_state' in model_params:
        model_params['random_state'] = final_random_state
        
    print(f"Best trial: #{best_trial.number} with value: {best_trial.value}")
    print(f"Best classifier: {best_params['classifier']}")
    print(f"Best parameters: {model_params}")

    if best_params['classifier'] == 'LogisticRegression':
        best_model = LogisticRegression(**model_params)
    elif best_params['classifier'] == 'RandomForest':
        best_model = RandomForestClassifier(**model_params)
    elif best_params['classifier'] == 'LightGBM':
        best_model = LGBMClassifier(**model_params)
    elif best_params['classifier'] == 'XGBoost':
        best_model = XGBClassifier(**model_params)
    else:
        best_model = None
        raise ValueError(f"Unsupported classifier: {best_params['classifier']}")

    # Create the full pipeline with preprocessing
    best_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', best_model)
    ])

    # Fit the complete pipeline
    best_pipe.fit(X_train, y_train)

    return best_params, best_pipe, study