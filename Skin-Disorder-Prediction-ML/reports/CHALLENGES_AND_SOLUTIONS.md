# Challenges and Solutions

## Shared clinical presentation

Erythemato-squamous disorders can share clinical manifestations such as erythema and scaling.

**Technique:** Compare nonlinear ensemble models and SVM to capture interactions across clinical and histopathological features.

## Missing age values

The notebook finds eight missing age observations.

**Technique:** KNN imputation with five neighbors.

## Class imbalance

The class distribution is uneven, ranging from 112 Psoriasis records to 20 Pityriasis Rubra Pilaris records.

**Technique:** Stratified train/test splitting and Stratified 5-Fold Cross-Validation.

## Model evaluation

Cross-validation and held-out test performance are not identical.

**Technique:** Report both rather than relying on one score.
