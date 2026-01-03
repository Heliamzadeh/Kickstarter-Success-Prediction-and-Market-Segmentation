# Kickstarter-Success-Prediction-and-Market-Segmentation
Supervised ML with leakage control / Unsupervised clustering + validation / Business decision framing / Production-aware analytics /Strategic thinking (not just modeling)



---

## Overview
This project develops an **end-to-end, production-aware machine learning framework** to predict Kickstarter project success *before launch* and to uncover **latent creator market segments** using unsupervised learning. The pipeline integrates **rigorous data leakage prevention**, **supervised classification**, **cluster validation**, and **decision-oriented analytics** to support both creators and platform-level strategic planning.

The analysis is performed on **262,412 historical Kickstarter projects** across multiple categories and regions, following **industry best practices** in feature engineering, validation, and interpretability.

---

## Problem Definition
Crowdfunding success depends on multiple interacting factors (goal structure, content quality, timing, category effects), yet:

- Many predictive models suffer from **severe post-launch data leakage**
- Heterogeneous creator profiles invalidate one-size-fits-all strategies
- Platforms lack **segment-aware intervention policies**

This project addresses three core questions:

1. Can project success be reliably predicted using **only pre-launch information**?
2. Are there **structurally distinct creator segments** with different success dynamics?
3. How can ML outputs be translated into **actionable, decision-driven strategies**?

---

## Data & Feature Engineering

- **Dataset size:** 262,412 Kickstarter projects  
- **Target variable:** Binary project success  
- **Features:** 17 strictly pre-launch attributes  

### Key Feature Groups
- Funding structure: raw goal, log-scaled goal  
- Content quality: description presence, text-length proxies  
- Media signals: video presence  
- Contextual effects: category, launch month  

### Data Leakage Prevention (Critical)
The following **post-launch variables were explicitly excluded**:
- `pledged`
- `backers_count`
- `usd_pledged`

All transformations were performed using **training-only statistics**, ensuring valid out-of-sample evaluation.

---

## Data Preparation Pipeline
The preprocessing pipeline follows **production-grade ML standards**:

1. Temporal train–test split  
2. `StandardScaler` fit **only on training data**  
3. Categorical encoders fit **only on training data**  
4. Index alignment verification  
5. Missing value integrity checks  
6. Final feature–label consistency validation  

This guarantees **zero information leakage** and realistic generalization performance.

---

## Task 1 — Supervised Learning: Success Prediction

### Models Evaluated
- **Logistic Regression** (interpretable baseline)
- **Random Forest** (non-linear ensemble model)

### Model Performance Comparison

| Metric | Logistic Regression | Random Forest |
|------|---------------------|---------------|
| Accuracy | 61.97% | **68.83%** |
| Recall | 80.10% | **87.99%** |
| F1-Score | 0.724 | **0.779** |
| AUC-ROC | 0.587 | **0.685** |

**Selected Model:** Random Forest  
The ensemble model consistently outperforms the baseline across **accuracy, recall, ranking power, and robustness**, making it suitable for **decision-support deployment**.

### Key Predictive Drivers (Random Forest)
1. Video presence (~17.9%)
2. Goal amount (raw + log-scaled, ~35%)
3. Description quality indicators
4. Category-specific effects
5. Seasonal launch timing

### Interpretability (Logistic Regression)
Coefficient analysis confirms:
- Higher funding goals significantly reduce success probability
- Video and description presence materially increase odds of success
- Launch timing introduces meaningful seasonal effects

---

## Task 2 — Unsupervised Learning: Market Segmentation

### Methodology
- **K-Means clustering**
- Optimal cluster selection using:
  - Silhouette score
  - Davies–Bouldin index
  - Elbow method

### Optimal Number of Clusters: **K = 3**

| Cluster | Share | Profile | Test Success Rate |
|------|------|--------|------------------|
| Cluster 0 | 64.3% | Standard professional projects | ~60% |
| Cluster 1 | 28.9% | Lean, experienced creators | ~61% |
| Cluster 2 | 6.7% | Niche micro-projects | **~70%** |

### Key Insight
The **smallest and least-resourced cluster achieves the highest success rate**, driven by:
- Realistic funding goals
- Strong niche alignment
- High creator commitment
- Reduced over-optimism bias

This challenges the assumption that higher marketing intensity always leads to better outcomes.

---

## Strategic & Business Impact
By combining **predictive modeling** with **segment-aware analytics**, the framework enables:

- Segment-specific creator guidance
- Smarter platform resource allocation
- Higher ROI compared to generic interventions

A **2–3% platform-wide improvement** translates to **2,000–3,000 additional successful projects annually** at platform scale.

---

## Technical Stack

### Languages & Tools
- Python  
- Jupyter Notebook  
- Pandas, NumPy  
- Matplotlib, Seaborn  

### Machine Learning
- Logistic Regression
- Random Forest
- K-Means clustering
- Feature importance analysis
- Model interpretability techniques

### Analytics & Decision Science
- Data leakage prevention
- Generalization gap analysis
- Segment-driven strategy design
- ML-to-business translation

---

## Why This Project Matters
This project demonstrates:
- **Production-aware ML thinking**
- Strong understanding of **classification vs segmentation**
- Rigorous **validation and leakage control**
- Ability to convert ML outputs into **strategic decisions**
- Clear technical communication for non-technical stakeholders

It reflects the skillset expected from **data scientists, ML engineers, and analytics consultants** working on real-world decision systems.

---

## Repository Contents
- `kickstarter_ml_pipeline.ipynb` — End-to-end ML and clustering pipeline  
- `kickstarter_project_report.pdf` — Full technical and business analysis  


