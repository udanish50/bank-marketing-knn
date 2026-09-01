# Bank Marketing Subscription Prediction

A machine learning repository for the UCI Bank Marketing dataset. The project keeps the core classifier intentionally manual: a NumPy implementation of k-nearest neighbors (KNN), wrapped in clean data-loading, preprocessing, evaluation, and reporting code.

The classification task is to predict whether a bank client subscribes to a term deposit (`y = yes/no`) after a direct marketing campaign.

## Why This Repo Exists

Many beginner notebooks stop after `accuracy_score`. This dataset is imbalanced, so accuracy alone can be misleading. This repo teaches the full workflow:

- extract the original UCI archive safely
- encode categorical and binary variables
- standardize numeric variables using train-only statistics
- train a manual KNN classifier
- evaluate with many classification metrics
- save reproducible reports for GitHub review

## Dataset

Dataset: **Bank Marketing**, UCI Machine Learning Repository  
Creators: S. Moro, P. Rita, P. Cortez  
DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306)  
License: Creative Commons Attribution 4.0 International

The repo expects the original `bank+marketing.zip` archive. Put it here:

```text
data/raw/bank+marketing.zip
```

The loader also works with the nested UCI layout inside the archive.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp /path/to/bank+marketing.zip data/raw/
python scripts/train_manual_knn.py --k 5
```

Outputs are written to:

```text
reports/metrics.json
reports/confusion_matrix.csv
reports/figures/confusion_matrix.png
```

## Metrics Included

The report includes more than 10 metrics:

- accuracy
- balanced accuracy
- precision
- recall / sensitivity
- specificity
- F1 score
- F-beta score (`beta=2`)
- ROC AUC
- PR AUC / average precision
- log loss
- Brier score
- Matthews correlation coefficient
- Cohen's kappa
- negative predictive value
- false positive rate
- false negative rate
- top-10-percent lift

## Project Structure

```text
.
├── data/
│   ├── raw/                 # place bank+marketing.zip here
│   └── processed/           # generated artifacts, ignored by git
├── reports/
│   └── figures/             # generated evaluation plots
├── scripts/
│   └── train_manual_knn.py  # end-to-end CLI
├── src/
│   └── bank_marketing_knn/
│       ├── data.py          # archive extraction and dataset loading
│       ├── knn.py           # manual NumPy KNN classifier
│       ├── metrics.py       # rich binary-classification metrics
│       └── preprocessing.py # train/test-safe feature preparation
└── tests/
```

## Teaching Notes

### Why manual KNN?

KNN is easy to understand: for each test row, find the `k` closest training rows and let them vote. The implementation here is deliberately transparent:

1. distances are Euclidean
2. neighbor labels vote by majority
3. predicted probability is the fraction of positive neighbors
4. ties are handled deterministically

### Why standardize numeric columns?

KNN is distance-based. Without scaling, a variable such as `balance` can dominate smaller-scale variables such as `campaign` or `previous`. This repo computes scaling parameters from the training set only, then applies them to both train and test sets to avoid leakage.

### Why so many metrics?

Only about 11.5% of the sampled `bank.csv` rows are positive. A model can score high accuracy while missing most subscribers. Precision, recall, specificity, PR AUC, lift, and MCC give a more honest picture.

## Example

```bash
python scripts/train_manual_knn.py \
  --dataset-zip data/raw/bank+marketing.zip \
  --k 7 \
  --test-size 0.30 \
  --random-state 42
```

## Responsible Use

This is an educational model. Marketing decisions can affect real people, and demographic or financial variables can encode historical bias. Before using any model like this operationally, audit subgroup performance, calibrate decision thresholds, and involve domain experts.

## Citation

```bibtex
@misc{moro_rita_cortez_2014_bank_marketing,
  author = {Moro, S. and Rita, P. and Cortez, P.},
  title = {Bank Marketing},
  year = {2014},
  publisher = {UCI Machine Learning Repository},
  doi = {10.24432/C5K306}
}
```
