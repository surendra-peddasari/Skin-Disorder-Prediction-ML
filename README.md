# Skin Disorder Prediction using Machine Learning

## Project Overview

This project develops a machine learning pipeline for the classification of six erythemato-squamous skin disorders using clinical and histopathological attributes.

The project is based on the **Skin Disorder** project brief. The brief requires:

1. A complete exploratory data analysis report.
2. A machine learning model to predict the skin-disease class.
3. Practical suggestions to help doctors identify these disorders earlier.
4. A comparison of multiple machine learning models and a production recommendation.
5. A report of challenges encountered and the techniques used to address them.
6. All major tasks to be completed in a single Jupyter notebook.

The source project brief describes a dermatology dataset with 34 attributes: 11 clinical attributes, 22 histopathological attributes, and age. The target contains six erythemato-squamous disease classes.

> **Medical-use disclaimer:** This is an educational machine-learning project. Model predictions should not be treated as a medical diagnosis or used as a substitute for a qualified dermatologist, biopsy, laboratory testing, or clinical judgment.

---

## Objectives

- Understand the distribution and characteristics of the dermatology dataset.
- Explore relationships between patient age, clinical findings, histopathological findings, and disease class.
- Handle missing observations appropriately.
- Build and compare several multiclass classification algorithms.
- Evaluate models using stratified cross-validation and a held-out test set.
- Identify the most discriminative features using tree-based feature importance.
- Produce an interpretable analysis that can support research into differential diagnosis.

---

## Dataset

The dataset contains **366 patient records** and **34 input/target columns** in the notebook workflow:

- 33 predictive attributes
- 1 target attribute: `disease_class`

The project brief describes 11 clinical attributes, 22 histopathological attributes, and `Age`.

### Clinical attributes

The clinical features include:

- erythema
- scaling
- definite borders
- itching
- Koebner phenomenon
- polygonal papules
- follicular papules
- oral mucosal involvement
- knee and elbow involvement
- scalp involvement
- family history

### Histopathological attributes

The histopathological features include:

- melanin incontinence
- eosinophils in infiltrate
- PNL infiltrate
- fibrosis of papillary dermis
- exocytosis
- acanthosis
- hyperkeratosis
- parakeratosis
- clubbing of rete ridges
- elongation of rete ridges
- thinning of suprapapillary epidermis
- spongiform pustule
- Munro microabcess
- focal hypergranulosis
- disappearance of granular layer
- vacuolisation and damage of basal layer
- spongiosis
- saw-tooth appearance of retes
- follicular horn plug
- perifollicular parakeratosis
- inflammatory mononuclear infiltrate
- band-like infiltrate

`Age` is treated as a numeric feature.

The project brief states that most feature values are scored from 0 to 3, while family history is binary and age is continuous.

### Target classes

The six classes are:

1. Psoriasis
2. Seborrheic Dermatitis
3. Lichen Planus
4. Pityriasis Rosea
5. Chronic Dermatitis
6. Pityriasis Rubra Pilaris

---

## Repository Structure

```text
Skin-Disorder-Prediction-ML/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Skin_disorder_prediction.ipynb
│
├── Results/
│   ├── eda_age_distribution.png
│   ├── eda_class_distribution.png
│   ├── feature_importance.png
│   └── best_model_confusion_matrix.png
│
├── data/
│   └── README.md
│
└── reports/
    ├── MODEL_COMPARISON.md
    └── CHALLENGES_AND_SOLUTIONS.md
```

The raw `dermatology.data` file is not committed to the repository. The notebook first attempts to load it locally and otherwise downloads it from the UCI dataset URL.

---

## Project Workflow

The notebook follows this pipeline:

```text
Raw Dermatology Dataset
        │
        ▼
Data Loading & Column Definition
        │
        ▼
Missing-Value Identification
        │
        ▼
Exploratory Data Analysis
        │
        ├── Disease Class Distribution
        └── Age Distribution by Disease
        │
        ▼
KNN Imputation
        │
        ▼
Stratified 80/20 Train-Test Split
        │
        ▼
Standard Scaling
        │
        ▼
Multiple Classification Models
        │
        ├── Random Forest
        ├── Extra Trees
        ├── XGBoost
        ├── LightGBM
        └── SVM
        │
        ▼
5-Fold Stratified Cross-Validation
        │
        ▼
Held-Out Test Evaluation
        │
        ▼
Confusion Matrix + Classification Report
        │
        ▼
Feature Importance Analysis
        │
        ▼
Clinical/Research Interpretation
```

---

## Exploratory Data Analysis

### Disease-class distribution

The notebook identifies an imbalanced target distribution. The largest class is **Psoriasis (112 records)**, while **Pityriasis Rubra Pilaris (20 records)** is the smallest.

![Disease class distribution](images/eda_class_distribution.png)

### Age distribution

The notebook also compares age distributions across the six disease classes.

![Age distribution](images/eda_age_distribution.png)

This analysis is descriptive and should not be interpreted as a clinical rule for diagnosis.

---

## Data Preprocessing

### Missing values

The notebook identifies **8 missing values in the `age` column**. Missing values are represented by `?` in the original data and are converted to `NaN` during loading.

### KNN imputation

The notebook uses:

```python
KNNImputer(n_neighbors=5)
```

to estimate missing values based on the similarity of observations.

### Train-test split

The dataset is divided using:

- 80% training data
- 20% test data
- `random_state=42`
- stratification by disease class

The resulting split is:

- Training: **292 records**
- Test: **74 records**

### Feature scaling

`StandardScaler` is fitted on the training data and then applied to both training and test sets.

---

## Machine Learning Models

Five classifiers are compared:

1. **Random Forest**
2. **Extra Trees**
3. **XGBoost**
4. **LightGBM**
5. **Support Vector Machine (SVM)**

Model performance is assessed using:

- Stratified 5-fold cross-validation accuracy
- Cross-validation standard deviation
- Held-out test accuracy
- Weighted F1-score

---

## Model Comparison

The results stored in the notebook are:

| Model | 5-Fold CV Accuracy | Test Accuracy | Weighted F1 |
|---|---:|---:|---:|
| Random Forest | 0.9725 ± 0.0139 | 0.9595 | 0.9595 |
| Extra Trees | 0.9760 ± 0.0085 | 0.9595 | 0.9595 |
| XGBoost | 0.9656 ± 0.0189 | 0.9189 | 0.9186 |
| LightGBM | 0.9656 ± 0.0189 | 0.9189 | 0.9157 |
| **SVM** | **0.9760 ± 0.0085** | **0.9730** | **0.9730** |

### Interpretation

The **SVM has the highest recorded held-out test accuracy and weighted F1-score (0.9730)** in the notebook.

Extra Trees and SVM have the same recorded mean 5-fold cross-validation accuracy (0.9760), with SVM achieving the stronger held-out test result.

The notebook nevertheless uses **Extra Trees as the model for the detailed confusion matrix and feature-importance analysis**. Therefore, the repository distinguishes between:

- **Best held-out predictive result:** SVM
- **Interpretability model used in the notebook:** Extra Trees

This distinction is important when interpreting the project results.

---

## Extra Trees Confusion Matrix

The notebook's Extra Trees evaluation produced **0.9595 test accuracy** on 74 test observations.

![Extra Trees confusion matrix](images/best_model_confusion_matrix.png)

The matrix shows strong class-level performance overall, with the main observed confusion involving:

- Seborrheic Dermatitis ↔ Pityriasis Rosea

The smaller Pityriasis Rubra Pilaris test group contains only four observations, so its perfect test classification should be interpreted cautiously.

---

## Feature Importance

The Extra Trees model was also used to calculate Gini-based feature importance.

The top features recorded by the notebook include:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `clubbing_of_rete_ridges` | 0.113419 |
| 2 | `fibrosis_of_papillary_dermis` | 0.073288 |
| 3 | `elongation_of_rete_ridges` | 0.061121 |
| 4 | `spongiosis` | 0.059772 |
| 5 | `koebner_phenomenon` | 0.056319 |
| 6 | `thinning_of_suprapapillary_epidermis` | 0.054459 |
| 7 | `vacuolisation_and_damage_of_basal_layer` | 0.049769 |
| 8 | `saw_tooth_appearance_of_retes` | 0.044977 |
| 9 | `PNL_infiltrate` | 0.036015 |
| 10 | `melanin_incontinence` | 0.032103 |

![Feature importance](images/feature_importance.png)

These are **model-derived importance scores**, not clinical causal effects. They should be treated as signals for further clinical investigation rather than independent diagnostic criteria.

---

## Suggestions for Early Identification

The notebook proposes using high-discriminant clinical and histopathological features to assist differential diagnosis.

Examples identified by the analysis include:

- clubbing of rete ridges
- fibrosis of papillary dermis
- elongation of rete ridges
- spongiosis
- Koebner phenomenon
- thinning of suprapapillary epidermis
- saw-tooth appearance of retes

A practical research workflow is to combine clinical observations with relevant histopathological findings and use the model as a decision-support aid.

The project should **not** be used to make autonomous medical decisions.

---

## Challenges and Solutions

### 1. Similar clinical presentation

The six target diseases share overlapping manifestations such as erythema and scaling.

**Approach:** Compare nonlinear ensemble models and SVM to capture relationships between multiple clinical and histopathological attributes.

### 2. Missing age observations

Eight age values are missing.

**Approach:** Use KNN imputation with five neighbors instead of replacing missing values with a simple global statistic.

### 3. Class imbalance

The class counts range from 112 Psoriasis records to 20 Pityriasis Rubra Pilaris records.

**Approach:** Use stratified train-test splitting and Stratified 5-Fold Cross-Validation so that class proportions are preserved during evaluation.

### 4. Model-selection ambiguity

Different models perform differently across cross-validation and the held-out test set.

**Approach:** Report both cross-validation performance and held-out test performance rather than selecting a model from a single metric.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/surendra-peddasari/Skin-Disorder-Prediction-ML.git
cd Skin-Disorder-Prediction-ML
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook notebooks/Skin_disorder_prediction.ipynb
```

Run the notebook from top to bottom.

The notebook attempts to read:

```text
dermatology.data
```

from the working directory. If the local file is unavailable, it falls back to the UCI dataset URL included in the notebook.

---

## Requirements

Main libraries used by the notebook:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Jupyter Notebook

See [`requirements.txt`](requirements.txt) for the project dependency list.

---

## Reproducibility

The notebook uses fixed random states in the main train/test split, cross-validation procedure, and model configurations.

However, exact results can still vary with:

- library versions
- operating system
- CPU/threading behavior
- changes to the source dataset
- changes to preprocessing or model parameters

For a production system, the preprocessing pipeline, model, dependency versions, and evaluation dataset should be versioned together.

---

## Limitations

- The dataset contains only 366 observations.
- The target distribution is imbalanced.
- The dataset focuses on six erythemato-squamous diseases.
- Feature values are structured clinical/histopathological scores rather than raw images.
- The model has not been externally validated on an independent clinical population.
- Feature importance does not establish causation.
- A high test score on this dataset does not guarantee real-world clinical performance.

---

## Future Improvements

Potential next steps include:

- Hyperparameter optimization for every classifier.
- Nested cross-validation for less biased model selection.
- Class-weighted or resampling approaches for imbalance.
- Calibration of predicted probabilities.
- SHAP-based explanations for individual predictions.
- External validation on an independent dataset.
- Pipeline serialization with `joblib`.
- A small clinician-facing web application for research demonstration.
- Monitoring for data drift if the model is deployed.
- Prospective clinical validation before any real-world use.

---

## Project Deliverables

The repository contains:

- Complete Jupyter notebook
- EDA visualizations
- Model comparison
- Confusion matrix
- Feature importance analysis
- Challenges and solutions report
- Project requirements documentation
- Installation instructions

---

## Reference

The project brief identifies the work as **PRCP-1027 - Skin Disorder** and classifies it under the healthcare domain. It specifies the six erythemato-squamous disease classes, the 34-attribute dataset structure, model comparison, challenges report, and single-notebook submission requirement.

---

## Author

**Surendra Peddasari**

GitHub: [surendra-peddasari](https://github.com/surendra-peddasari)

---

## Disclaimer

This repository is intended for **education, machine-learning experimentation, and research**. It is not a medical device and must not be used as a substitute for professional medical diagnosis or treatment.
