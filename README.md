# Financial Python: Quantitative Risk & Applied ML Knowledge Base

This repository is a personal, hands-on knowledge base of 40+ Jupyter notebooks built while studying **quantitative market risk management** and **applied machine learning**, ranging from first-principles mathematics to production MLOps tooling. It is organized as a reference library rather than a single application: material is grouped by subject area, and within each area notebooks are (mainly) numbered to be read in sequence. 

## Highlights for Reviewers

Reviewersare encouraged to start with the notebooks below. The top three ( Black-Scholes/Brownian motion, CVaR optimization, and VaR backtesting ) were implemented from scratch from theory.

| Notebook | Why it merits review                                                                                                                                                                                                                                                                   |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Black-Scholes: Brownian Motion to Monte Carlo Pricing](Market%20Risk%20Management/09_black_scholes_brownian_motion_simulation.ipynb) | Builds the stochastic pricing stack from first principles: Bernoulli trials, random walks, Brownian motion, a no-arbitrage/replication derivation of risk-neutral pricing, and the Black-Scholes PDE/closed form. Implements a Monte Carlo pricer (GBM paths, 95% confidence interval, convergence check) validated against the closed form. I also experimented with the JAX framework instead of NumPy/SciPy here. |
| [Rockafellar-Uryasev CVaR Optimization](Market%20Risk%20Management/04_rockafellar_uryasev_optimization.ipynb) | notebook reformulates portfolio optimization as a linear program to minimize CVaR directly, rather than using variance/std dev as a proxy for risk.                                                                                                                                    |
| [VaR Backtesting: Kupiec, Christoffersen, Traffic Light](Market%20Risk%20Management/08_backtest_trafficlight_kupiec_christoffersen.ipynb) | implementation of the statistical tests used by risk desks and regulators (Kupiec proportion-of-failures, Christoffersen independence, and Traffic Light) to validate a VaR model against realized losses.                                                                             |
| [ARCH/GARCH Volatility Modeling](Market%20Risk%20Management/07_arch_garch.ipynb) | A complete conditional-heteroskedasticity model build-out, including residual diagnostics and mean-equation assumptions.                                                                                                                                                               |
| [Custom Group-Based Imputer](ML/scikit-learn/data_cleaning/01_missing_values.ipynb) | A hand-written, scikit-learn-compatible transformer (`GroupImputer`) built on `BaseEstimator`/`TransformerMixin` that imputes missing values using per-group statistics, with a fallback to a global statistic for groups unseen at fit time. Covered by a `pytest` unit-test suite.   |
| [Lazy-Loading Parquet Dataset for PyTorch](ML/pytorch/06_OOP_weight_initialization_normalization.ipynb) | A custom `torch.utils.data.Dataset` (`HomeLazyDataset`) that reads Parquet files row-group by row-group through PyArrow instead of loading them fully into memory, with row-group offset indexing and single-group caching to avoid redundant disk reads under random-access sampling. |
| [MLflow: Tracking → Registry → Serving](ML/MLFlow) | A complete MLOps loop covering experiment tracking, model registry/staging and theroretically serving                                                                                                                                                                                  |
| [Maximum Likelihood for ARCH/GARCH](Market%20Risk%20Management/06_maximum_likelihood_arch_garch.ipynb) | Studies maximum likelihood estimation from first principles, contrasting closed-form OLS with numerical MLE and working through log-likelihood by hand, to build the theory needed before fitting ARCH/GARCH volatility models with a library implementation.                          |


## Repository Structure

```
.
├── Market Risk Management/     # VaR/CVaR, volatility, and derivatives-pricing notebooks (numbered, cumulative)
├── ML/
│   ├── scikit-learn/            # Classical ML: classification, regression, clustering, metrics
│   │   └── data_cleaning/       # Missing values, encoding, scaling, pipelines
│   ├── pytorch/                 # Tensors, custom nn.Module/Dataset classes, RNNs, torchmetrics
│   ├── MLFlow/                  # Experiment tracking, model registry, model serving
│   ├── utils/                   # Reference notebooks on supporting tooling (Parquet/PyArrow, Plotly)
│   ├── loss_functions.md        # Reference notes on loss functions by task
│   └── optimizers.md            # Reference notes on optimizer theory
├── Structural Breaks.ipynb      # Stationarity and structural breaks in a risk-management context
├── data_financialcrisis/        # Historical price and financial-crisis datasets
├── data_general/                # Time-series and credit-risk datasets, with supplementary reference notebooks
└── data_nonfinancial/           # General-purpose dataset used for ML practice unrelated to finance
```

## Contents

### Market Risk Management

The core VaR/CVaR and volatility-modeling sequence, intended to be read in order:

1. [Markowitz Optimization](Market%20Risk%20Management/01_markowitz_optimization.ipynb): Efficient frontier construction using PyPortfolioOpt and cvxpy.
2. [Monte Carlo Simulation](Market%20Risk%20Management/02_monte_carlo.ipynb): Distribution fitting for basic random variables, extended to simulating equity price paths.
3. [VaR & CVaR](Market%20Risk%20Management/03_var_cvar.ipynb): Historical and parametric Value-at-Risk/Conditional VaR (normal and Student-t).
4. [Rockafellar-Uryasev CVaR Optimization](Market%20Risk%20Management/04_rockafellar_uryasev_optimization.ipynb): Linear-programming reformulation of CVaR minimization, implemented from scratch from independently derived theory.
5. [Volatility & Implied Volatility](Market%20Risk%20Management/05_volatility_implied_volatility.ipynb): Rolling/annualized volatility, implied volatility, VIX, and the variance risk premium.
6. [Maximum Likelihood ARCH/GARCH](Market%20Risk%20Management/06_maximum_likelihood_arch_garch.ipynb): Maximum likelihood estimation fundamentals (OLS vs. MLE, log-likelihood) studied to build the theory behind volatility fitting; the ARCH/GARCH models themselves are fit with a library implementation.
7. [ARCH/GARCH](Market%20Risk%20Management/07_arch_garch.ipynb): Conditional heteroskedasticity models.
8. [Backtesting (Kupiec, Christoffersen, Traffic Light)](Market%20Risk%20Management/08_backtest_trafficlight_kupiec_christoffersen.ipynb): VaR model validation; the tests are implemented from scratch from independently derived theory.
9. [Black-Scholes](Market%20Risk%20Management/09_black_scholes_brownian_motion_simulation.ipynb): Builds up from first principles — Bernoulli trials, random walks, and Brownian motion (Wiener process properties, discretization, three generator implementations) — through a no-arbitrage derivation of risk-neutral pricing and the Black-Scholes PDE/closed form, including the Greeks. Also implements a Monte Carlo pricer that simulates GBM paths and validates against the closed form with a 95% confidence interval and convergence plot.

### Machine Learning (`ML/`)

**[scikit-learn](ML/scikit-learn)**. Classical ML fundamentals: [classification](ML/scikit-learn/01_classificiation.ipynb) (k-Nearest Neighbors), [regression](ML/scikit-learn/02_regression.ipynb) (linear and regularized), [accuracy metrics](ML/scikit-learn/03_basic_accuracy_metrics.ipynb) (confusion matrix, precision/recall, ROC), [hyperparameter tuning](ML/scikit-learn/04_hyperparameter%20tuning.ipynb) (grid search), and [clustering](ML/scikit-learn/05_clustering.ipynb) (k-means).

**[data_cleaning](ML/scikit-learn/data_cleaning)**. [Missing values and imputation](ML/scikit-learn/data_cleaning/01_missing_values.ipynb) (including the hand-written `GroupImputer` described above), [categorical encoding](ML/scikit-learn/data_cleaning/02_category_encoding.ipynb), [scaling and skewness](ML/scikit-learn/data_cleaning/03_scaling.ipynb), and [pipelines/ColumnTransformer](ML/scikit-learn/data_cleaning/0x_pipelines.ipynb) for leakage-safe preprocessing.

**[pytorch](ML/pytorch)**. [tensor operations](ML/pytorch/01_tensor_operations.ipynb), an [introduction to activation functions](ML/pytorch/02_introduction_activation_functions.ipynb), [data loading](ML/pytorch/03_loading_data.ipynb) (`Dataset`/`DataLoader`, train/val/test loops), a [deeper look at activation functions](ML/pytorch/04_activation_functions.ipynb), [optimizers](ML/pytorch/05_optimizers.ipynb) (SGD with momentum, learning-rate effects), [custom `nn.Module` design and the lazy Parquet-loading `Dataset`](ML/pytorch/06_OOP_weight_initialization_normalization.ipynb) described above, [evaluation with torchmetrics](ML/pytorch/07_metrics.ipynb), and [RNNs for sequential data](ML/pytorch/09_RNN.ipynb).

**[MLFlow](ML/MLFlow)**. [tracking fundamentals](ML/MLFlow/00_mlflow_intro_random_code_snippets.ipynb), [models and autologging](ML/MLFlow/01_mlflow_models.ipynb), the [model registry](ML/MLFlow/02_mlflow_model_registry.ipynb), and [model serving](ML/MLFlow/03_serving_models.ipynb) from local disk, a run ID, or S3.

**[utils](ML/utils)**. Supplementary reference notebooks on supporting tooling: [Parquet and PyArrow](ML/utils/01_parquet_pyarrow.ipynb) and [Plotly](ML/utils/02_plotly.ipynb).



## Stack

Python, pandas/NumPy, scikit-learn, PyTorch, torchmetrics, MLflow, statsmodels/`arch` (GARCH), cvxpy, PyPortfolioOpt, PyArrow, Plotly, Jupyter.
