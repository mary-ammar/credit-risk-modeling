# Credit Risk Modeling

A machine learning project to predict credit default risk using the German Credit Dataset.
The goal is to build and compare three classification models that can identify applicants
likely to default on a loan, while explaining the reasoning behind each prediction.

## Dataset

This project uses the **German Credit Dataset** originally created by
Prof. Hans Hofmann, University of Hamburg (1994).
Available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

The dataset contains 1000 applicants described by 20 features covering
checking account status, credit history, loan purpose, savings, employment,
and personal information. The target variable indicates whether the applicant
is a good (0) or bad (1) credit risk.

## Project Structure

```
credit-risk-modeling/
│
├── notebooks/
│   ├── 01_problem_understanding.md       # Project scope and goals
│   ├── 02_data_understanding.md          # Dataset description
│   ├── 03_data_exploration.ipynb         # Exploratory data analysis
│   ├── 04_data_preprocessing.ipynb       # Cleaning, encoding, scaling
│   ├── 05_logistic_regression.ipynb      # Baseline model
│   ├── 06_random_forest.ipynb            # Ensemble model
│   ├── 07_gradient_boosting.ipynb        # Sequential ensemble model
│   ├── 08_model_evaluation.ipynb         # Side-by-side comparison
│   ├── 09_model_interpretation.ipynb     # Feature importance analysis
│   └── 10_shap_analysis.ipynb            # SHAP explainability
│
├── data/                                  # Generated files (not tracked by git)
├── models/                                # Saved models (not tracked by git)
├── src/                                   # Helper scripts
├── Dockerfile                             # Docker setup
├── requirements.txt                       # Python dependencies
└── .github/workflows/ci.yml              # GitHub Actions CI pipeline
```

## How to Run

### Option 1 — Local

```bash
# Clone the repo
git clone https://github.com/mary-ammar/credit-risk-modeling.git
cd credit-risk-modeling

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/
```

> **Important:** Always run `04_data_preprocessing.ipynb` first.
> It generates the `.npy` files in `data/` that all model notebooks depend on.
> These files are not tracked by git because they are reproducible from code.

### Option 2 — Docker

```bash
# Build the image
docker build -t credit-risk .

# Run the container
docker run -p 8888:8888 credit-risk
```

Then open `http://localhost:8888` in your browser.

## Models Trained

Three models were trained and compared on the same train/test split:

| Model | Accuracy | ROC-AUC | Recall (Bad) | Precision (Bad) |
|---|---|---|---|---|
| Logistic Regression | 72.5% | 0.782 | 0.767 | 0.529 |
| Random Forest | 73.5% | 0.792 | 0.667 | 0.548 |
| Gradient Boosting | 72.0% | 0.781 | 0.717 | 0.524 |

**Recommended model: Random Forest** — best ROC-AUC and precision while
maintaining solid recall on the bad credit class.

## Key Findings

All three models consistently identified the same top features as the
strongest predictors of credit default risk:

- **Checking account status** — applicants with no checking account
  are associated with lower default risk
- **Credit amount** — larger loans carry higher default risk
- **Loan duration** — longer loans are associated with higher risk
- **Age** — shows non-linear relationship with credit risk
- **Credit history** — critical credit history is a strong risk signal

## Preprocessing Steps

The following transformations were applied to prepare the data for modeling:

- IQR-based outlier clipping on `credit_amount`, `duration_months`, and `age`
- Log transformation on `credit_amount` to correct right skew
- Rare category grouping (categories below 5% frequency replaced with `Other`)
- RobustScaler on numerical features
- OneHotEncoder on categorical features
- All transformations fitted on training data only to prevent data leakage

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
joblib
jupyter
```

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
