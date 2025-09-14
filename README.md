# Explainable AI in Credit Underwriting: Evaluating LLM-generated Decision Explanations

A comprehensive framework for explainable AI in credit decision making, combining traditional ML models with multiple explanation techniques and LLM integration.

## Workflow Overview

The system follows this sequence:

1. **Model Training** (`notebooks/model_training.ipynb`)
   - Data preprocessing and feature engineering
   - Hyperparameter tuning with Optuna
   - Model selection and evaluation
   - Outputs: trained model, preprocessing pipeline, performance metrics

2. **Batch Predictions & Explanations** (`scripts/predict_batch.py`)
   - Generate predictions for test dataset
   - Create SHAP explanations for feature importance
   - Generate counterfactual explanations using DiCE-ML
   - Outputs: predictions, SHAP values, counterfactual scenarios

3. **Data Sampling** (`notebooks/sampling.ipynb`)
   - Stratified sampling from prediction results
   - Balance representation across prediction probabilities
   - Outputs: sampled dataset for LLM evaluation

4. **LLM Explanation Generation** (`scripts/explain_predictions.py`)
   - Generate natural language explanations using multiple LLMs
   - Support for Ollama, OpenAI, Google Vertex AI
   - Caching system for cost efficiency
   - Outputs: LLM explanations stored in DynamoDB

5. **Evaluation** (Choose one approach):
   - **Option A**: Sequential evaluation (`scripts/evaluate_llm_responses.py`)
     - Evaluate all explanations locally using chosen LangChain LLM from `config.json`
     - Compute CEMAT, regulatory compliance, counterfactual verification
   - **Option B**: Batch processing with Google Cloud
     - Submit batch job via `notebooks/judgeLLM_batch_evaluation.ipynb`
     - Process results and compute local metrics via `scripts/process_batch_evaluation_results.py`
   - Outputs: evaluation scores and metrics

6. **Analysis & Visualization** (`notebooks/explanation_evaluation.ipynb`)
   - Comparative analysis across LLMs
   - Performance metrics and processing times
   - Generate publication-ready visualizations
   - Outputs: analysis plots and summary statistics

## Quick Start

### Prerequisites
- Python 3.10
- uv package manager
- Ollama (for local LLM inference)
- AWS credentials (for DynamoDB storage)
- Google Cloud credentials (for batch evaluation option)

### Installation
```bash
git clone https://github.com/yourusername/xai-credit-decisions.git
cd xai-credit-decisions
uv sync
```

## Configuration

Key settings in `config/config.json`:
- LLM provider configurations
- Evaluation criteria weights
- Processing parameters

Create a `.env` file in `config/` by copying the template:
```bash
cp config/env.template config/.env
```

Then edit `config/.env` with your actual API keys:
- AWS credentials (if DynamoDB storage is desired)
- Google API Key (for Gemini models)
- OpenAI API Key (for GPT models)


## Key Components

- **Model Training**: Automated pipeline with Optuna hyperparameter optimization
- **Explanations**: SHAP values, counterfactuals, and LLM-generated natural language
- **Evaluation**: Multi-criteria assessment of explanation quality
- **Storage**: DynamoDB for scalable evaluation data management
- **Visualization**: Academic-style plots for thesis publication

## Dataset

Uses the "Give Me Some Credit" dataset with 150,000 credit applications and 11 features including age, income, debt ratios, and credit history.

## Output Structure

```
output/
├── models/              # Trained models and metadata
├── predictions/         # Batch prediction results
├── llm_explanations/   # LLM-generated explanations
├── evaluations/         # Evaluation results and metrics
└── plots/              # Generated visualizations
```



## Research Context

This implementation supports research into:
- Comparative analysis of explanation methods
- Human-AI interaction in financial decision making
- Regulatory compliance for AI explainability
- Trust and adoption of AI systems in finance

---

*Master's Thesis: Explainable AI in Credit Underwriting: Evaluating LLM-generated Decision Explanations*

## Usage Examples

### Quick Start
```bash
# 1. Train the model
uv run jupyter notebook notebooks/model_training.ipynb

# 2. Generate predictions and explanations
uv run python scripts/predict_batch.py

# 3. Sample data for evaluation
uv run jupyter notebook notebooks/sampling.ipynb

# 4. Generate LLM explanations
uv run python scripts/explain_predictions.py

# 5. Evaluate explanations
uv run python scripts/evaluate_llm_responses.py

# 6. Analyze results
uv run jupyter notebook notebooks/explanation_evaluation.ipynb
```

### Batch Evaluation with Google Cloud
For large-scale evaluation, use the batch processing approach:
```bash
# Submit batch job
uv run jupyter notebook notebooks/judgeLLM_batch_evaluation.ipynb

# Process results
uv run python scripts/process_batch_evaluation_results.py
```

## Research Methodology

This implementation follows a systematic approach to explainable AI evaluation:

1. **Model Development**: LightGBM classifier with hyperparameter optimization
2. **Explanation Generation**: SHAP values, counterfactual scenarios, and LLM-generated natural language
3. **Evaluation Framework**: Multi-criteria assessment including CEMAT, regulatory compliance, and counterfactual verification
4. **Comparative Analysis**: Performance metrics across different LLM providers and explanation methods

## Key Findings

- **LLM Performance**: Gemini 2.5 Flash shows superior performance in explanation quality
- **Explanation Methods**: Combination of SHAP and counterfactuals provides comprehensive insights
- **Regulatory Compliance**: Generated explanations meet regulatory requirements for financial AI systems
- **Processing Efficiency**: Local models (Ollama) provide cost-effective alternatives to cloud APIs

## File Structure

```
xai-credit-decisions/
├── config/                 # Configuration files
├── data/                   # Dataset and processed data
├── notebooks/              # Jupyter notebooks for analysis
├── output/                 # Generated results and visualizations
├── scripts/                # Python scripts for automation
├── src/xai_pkg/           # Main package source code
└── README.md              # This file
```

## Dependencies

Key dependencies include:
- **Machine Learning**: scikit-learn, LightGBM, XGBoost, SHAP, DiCE-ML
- **LLM Integration**: LangChain, OpenAI, Google Vertex AI, Ollama
- **Evaluation**: Custom evaluation metrics and judge LLM framework
- **Visualization**: Matplotlib, Seaborn for publication-ready plots

## License

This project is part of a master's thesis research. Please cite appropriately if used in academic work.

## Acknowledgements

* Prof. Jose Rodriguez as thesis supervisor
* [Fredes & Vitria (2024)](https://arxiv.org/abs/2408.15133) on the combination of Counterfactuals and LLMs
* [Slack et al. (2022)](https://arxiv.org/abs/2207.04154) for an interactive approach on XAI
* [Marten et al. (2023)](https://arxiv.org/abs/2309.17057) for the inspiration on combining SHAP and CF to give natural language explanations