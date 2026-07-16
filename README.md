# Predicting Adverse Authorisation Outcomes for UK FCA-Regulated Firms Using Machine Learning

**MSc Data Science (FinTech) Dissertation Project — University of Greenwich**

**Author:** Fannana Fahreen Aanan
**Supervisor:** Konstantinos Skindilias

---

## Project Overview

This project develops a **SupTech (Supervisory Technology)** machine learning tool that predicts whether a UK FCA-regulated firm will experience an **adverse authorisation outcome** — revocation of authorisation or a formal disciplinary action — within a **24-month horizon**. The goal is to help the Financial Conduct Authority (FCA) allocate limited supervisory and inspection resources toward the firms most likely to require intervention.

This is explicitly framed as a **regulatory enforcement prediction problem**, distinct from traditional financial distress or bankruptcy prediction. A firm can be financially healthy and still face enforcement (e.g. large banks fined for conduct failures), while a financially struggling firm may never be formally disciplined. The literature has extensively studied bankruptcy prediction; regulatory enforcement prediction for non-bank financial firms remains comparatively underexplored.

### Research Question

> Does an XGBoost classifier trained on FCA Financial Services Register characteristics and Companies House filing behaviour achieve materially higher precision-at-top-decile than an L2-regularised logistic regression baseline, when predicting adverse authorisation outcomes within 24 months?

---

## Dataset

A **self-constructed firm-year panel** built entirely from two free UK government APIs:

- **FCA Financial Services Register API** — 30,160 firm records collected via a targeted FRN sweep
- **Companies House API** — 22,110 matched company records (director, filing, and governance data)

**Final panel:** 82,900 firm-year observations (2016–2024), covering 26,141 unique firms, with 307 positive cases (0.37% adverse-outcome rate).

*Note: raw JSON data collection files are not included in this repository due to size (~52,000 files) and are stored locally. Data collection notebooks and API credentials are excluded for security.*

---

## Methodology

1. **Exploratory Data Analysis** (`01_Data_Cleaning.ipynb`) — including a major **data leakage investigation** that identified and corrected three separate point-in-time leakage issues in the FCA and Companies House data (`fca_status`, `company_status`, and a partially-unresolvable issue in `n_permissions`)
2. **Feature Engineering** (`02_Feature_Engineering.ipynb`) — log transformation of skewed features, one-hot encoding, missing-value treatment with domain-informed strategies
3. **Modelling** (`03_Modelling.ipynb`) — three models compared: a naive baseline, L2-regularised logistic regression, and XGBoost (tuned via Bayesian optimisation with Optuna), evaluated across a rigorous **5-round expanding-window backtest** (2020–2024)

### Key Analyses

- Bootstrap confidence intervals on model performance differences
- SHAP interpretability analysis across all backtesting rounds
- A **factorial decomposition experiment** isolating whether XGBoost's advantage stems from training data volume or regulatory regime relevance (finding: ~84% regime relevance, ~16% volume)
- Static vs. expanding-window model comparison (demonstrating the operational necessity of model retraining)
- Rank stability analysis (Spearman correlation of risk rankings across years)
- Three sensitivity analyses: exclusion of the partially-leaky `n_permissions` feature, alternative labelling of voluntary cancellations, and alternative class-imbalance handling methods (SMOTE, undersampling)
- Calibration analysis (reliability diagrams)

---

## Repository Structure
