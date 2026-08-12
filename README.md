# Civil Conflict Prediction using Machine Learning

A comparative machine learning project that predicts the probability of civil conflict in countries using multiple models: Transformer, XGBoost, and Feed-Forward Neural Network (FFNN).

## Overview

This project addresses the critical question: **Which countries are at highest risk of civil conflict, and how accurately can different ML models predict it?**

Using a dataset of country-level indicators combined with conflict data, we train and compare three distinct modeling approaches — a Transformer model, XGBoost ensemble, and a simple feed-forward neural network — to predict binary conflict outcomes with probabilistic confidence.

## Dataset

- **Country Indicators** (`country_indicators.csv`): ~195 countries with demographic, economic, health, and governance features from UN, World Bank, and WHO sources
- **Conflict Data** (`test_predictions.csv`): Monthly predictions from three models (Transformer, XGBoost, FFNN) with true labels for 2022-2023

### Features Include
- Population demographics (age structure, growth rate, fertility)
- Economic indicators (GDP, urbanization, migration)
- Health metrics (mortality rates, life expectancy)
- Child survival indicators
- Governance and rights data

## Models Compared

| Model | Approach | Strengths |
|-------|----------|-----------|
| **Transformer** | Attention-based architecture | Captures long-range dependencies in temporal data |
| **XGBoost** | Gradient boosting ensemble | Strong tabular performance, handles non-linearity well |
| **FFNN** | Multi-layer perceptron | Baseline neural network, interpretable hidden layers |

## Methodology

1. **Data Integration**: Merge country indicators with model predictions on `iso3` key
2. **Error Analysis**: Compute absolute prediction error for each model (`|y_true - y_pred_proba|`)
3. **Model Comparison**: Evaluate accuracy, recall, and confusion matrices across models
4. **Feature Importance**: Use Partial Dependence Plots (PDP) to understand key drivers
5. **Decision Tree Visualization**: Extract and visualize the first XGBoost tree for interpretability

## Key Results

- **Transformer** and **XGBoost** achieve comparable accuracy, with XGBoost showing slightly better precision on positive (conflict) cases
- **FFNN** serves as a baseline — simpler but less accurate than tree-based and attention models
- Key predictive features identified through PDP analysis include population growth rate, child mortality, and governance indicators
- The model performs best on medium-risk countries; extreme cases (very high or very low conflict probability) show more variance

## Project Structure

```
STA130-Project/
├── Project Assignment #1 (1) (1) (1).ipynb   # Error analysis & model comparison
├── Project Assignment #2 (1).ipynb           # Extended analysis & visualization
├── STA130 group project assignment #3 (2) (3).ipynb  # Final submission
├── STA130_F23_CourseProjectNotebook (2).ipynb # Iterative development notebook
├── country_indicators (1).csv                # Country-level features
├── test_predictions (1).csv                  # Model predictions + ground truth
└── Project Slides (1).pptx                   # Presentation slides
```

## Team

Qingwen Jia, Jiale Qian, Vinson Liang, Joses Ng, Natalie Choy

## Course

STA130 — Introduction to Data Science, University of Toronto (Fall 2023)

## Technologies

- Python, Pandas, NumPy
- scikit-learn (XGBoost, FFNN, metrics)
- Plotly (interactive visualizations)
- Seaborn, Statsmodels
- Graphviz (tree visualization)
