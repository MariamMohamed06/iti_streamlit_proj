# Job Change Predictor — Streamlit App

Built from `ITI_Project_V2_.ipynb`. Serves live predictions from **every**
model in the notebook:

| Model | Features used | Notes |
|---|---|---|
| KNN | plain encoded | SMOTE + PCA + RandomizedSearchCV + tuned decision threshold |
| LR + SMOTE | plain encoded | Logistic Regression on SMOTE-balanced data |
| FE LR + SMOTE | feature-engineered | adds `experience_to_training_ratio`, `has_relevant_degree` |
| Baseline Random Forest | feature-engineered | `class_weight="balanced"` |
| Tuned Random Forest | feature-engineered | RandomizedSearchCV over RF hyperparameters |
| XGBoost + scale_pos_weight | sanitized encoded | Approach A, early stopping |
| XGBoost + SMOTE | sanitized encoded | Approach B, early stopping |

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Train & export every model

Put the original Kaggle **`aug_train.csv`** ("HR Analytics: Job Change of
Data Scientists") next to `train_all_models.py`, then:

```bash
python train_all_models.py
```

Add `--fast` for a quicker smoke-test with smaller hyperparameter search
budgets (lower quality models, seconds instead of minutes):

```bash
python train_all_models.py --fast
```

This creates an `artifacts/` folder containing:
- `encoder.pkl` — the shared `OneHotEncoder`
- one `.pkl` file per model
- `metadata.json` — feature lists/order, per-model threshold + test metrics

## 3. Run the app

```bash
streamlit run app.py
```

The app reads dropdown options (cities, education levels, etc.) directly
from the encoder, so it always matches whatever categories were present in
your training data — nothing is hardcoded.

## Features

- **Single model mode** — pick any one of the 7 models, see a colorful
  gauge of the "will change jobs" probability plus its accuracy/precision/
  recall/F1 on the held-out test set.
- **Compare all models mode** — enter one candidate profile and see every
  model's probability side-by-side on a horizontal bar chart, an average
  probability, and a simple majority-vote consensus.
- Sidebar leaderboard ranking all 7 models by F1-score.
