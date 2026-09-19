# Identifying Fraud Rings Through Clustering

Unsupervised clustering (K-Means, DBSCAN, Hierarchical) applied to the IEEE-CIS Fraud Detection dataset to test whether transactions can be grouped into fraud-ring-like clusters — without using the fraud label to fit the clustering models — and whether that grouping actually concentrates real fraud above baseline.

This is a capstone project for the AIM Postgraduate Diploma in AI & ML (Pillar 5), building on a similar clustering-based fraud-ring detection approach deployed in 2025 by Shopee PH's BI & Fraud teams.

## Business Problem

E-commerce and payments platforms lose significant revenue to fraud rings — groups of linked accounts working together — while overly aggressive fraud models create friction by declining legitimate customers. Manually deciding whether a group of accounts belongs to the same fraud ring consumes a large share of a fraud agent's review time and doesn't scale.

**Question:** Can transactions be grouped into clusters that correspond to the same underlying client — without using the fraud label in the clustering distance calculations or model fitting — such that the resulting clusters concentrate known fraud well above the platform's base rate?

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) (Kaggle competition, Vesta Corporation real-world e-commerce transactions).

- `train_transaction.csv`: 590,540 rows × 394 columns
- `train_identity.csv`: 144,233 rows × 41 columns (joined on `TransactionID`)
- Baseline fraud rate: ~3.5%

The raw data is **not included in this repo** (see `data/README.md` for download instructions) — it's too large for GitHub and is publicly available directly from Kaggle.

## Approach

1. **Problem Framing** — scoped as unsupervised clustering; the `isFraud` label was excluded from clustering distance calculations and model fitting. It was used separately for feature-selection analysis, post-hoc candidate comparison, and validation.
2. **Data Understanding** — profiled shape, missingness, and outliers across 434 raw fields.
3. **Preprocessing & Feature Engineering** — missingness treated as a signal (not noise), 38 engineered features, filter-based feature selection (413 → 271 features), PCA to 102 components (90% variance retained). Cluster tendency confirmed via Hopkins statistic (0.981, stable across 5 random seeds).
4. **Modeling** — K-Means, DBSCAN, and Hierarchical clustering, each tuned independently, then validated by checking fraud concentration in the highest-risk cluster(s) against baseline. K-Means and DBSCAN were additionally stress-tested on an 80/20 holdout split.
5. **Critical Thinking** — cluster explainability (feature profiling, since SHAP/LIME don't apply to unsupervised models), documented limitations (class imbalance, temporal leakage risk, overfitting), and a bias & fairness audit across billing regions.

## Key Results

| Model | In-sample fraud concentration | Holdout-validated | Notes |
|---|---|---|---|
| K-Means (k=5) | 15.3x baseline (54.61%) | Dropped to 3.2x | Result was largely sample-specific |
| DBSCAN | 8.45x baseline (29.41%) | Held at 9.94x | Fraud concentrated in its noise points, not its named clusters |
| Hierarchical (k=3) | 2.9x baseline (10.68%) | Not holdout-tested | Smaller sample size; served as a sanity check |

**Headline finding:** all three methods found fraud-concentrated groups, supporting the core premise. In the holdout-oriented robustness check, DBSCAN showed the most stable result, while K-Means's stronger-looking in-sample result dropped substantially on unseen data. Hierarchical clustering was not holdout-tested.

**Bias finding:** the highest-risk cluster is defined primarily by product category and device-data completeness, not fraud behavior directly — which caused two of the highest-actual-fraud billing regions to be almost entirely missed by flagging.

**Recommendation:** a two-stage funnel (K-Means as a high-recall net, DBSCAN as a precision filter within that narrowed group), combined with per-subgroup flagging-threshold recalibration to address the bias finding before any deployment.

Full detail, methodology justification, and the bias & fairness audit are in [`Charmagne Cruz_Capstone Final Report_20260924.pdf`](./Charmagne%20Cruz_Capstone%20Final%20Report_20260924.pdf).

## Repository Structure

```
├── notebooks/          # Analysis notebooks (Steps 1-5), run in Google Colab
├── src/                # Reusable helper functions extracted from the notebooks
├── models/             # Model-generation notes; fitted models can be regenerated from the notebooks
├── data/               # Not included — see data/README.md for download instructions
├── slides/             # Business-facing and technical presentation decks
├── Charmagne Cruz_Capstone Final Report_20260924.pdf
├── requirements.txt
└── README.md
```

## Reproducing This

1. Download `train_transaction.csv` and `train_identity.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) (see `data/README.md`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run `notebooks/Charmagne Cruz_Capstone_Part 1_Step 1 & 2_20260924.ipynb` first (data collection & understanding), then `notebooks/Charmagne Cruz_Capstone Part 2_Step 3 to 5_20260924.ipynb` (preprocessing through modeling & critical thinking). Both were built and run in Google Colab, with data read from Google Drive — update the `base_path` variable near the top of each notebook to point to your own data location.

## Author

Ma. Charmagne Cruz — AIM Postgraduate Diploma in AI & ML, Pillar 5 Capstone Project.
