# 📧 Email Spam Classifier

A machine learning project that classifies email messages as spam or not spam using text preprocessing and Naive Bayes.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green)

## Overview

Builds an end to end spam detection pipeline from raw text messages to a deployable trained model using the UCI SMS Spam Collection dataset.

**Best model:** Multinomial Naive Bayes with TF-IDF vectorisation  
**Key metric:** Precision (prioritised over accuracy due to class imbalance)

## Pipeline
1. **Data Cleaning** — removed duplicates, dropped unused columns, label-encoded target
2. **EDA** — distribution plots, word clouds, correlation heatmap
3. **Text Preprocessing** — lowercase → tokenise → remove stopwords → stem
4. **Vectorisation** — TF-IDF with 3000 features
5. **Model Comparison** — 11 classifiers evaluated
6. **Export** — `vectorizer.pkl` + `model.pkl` saved for inference

## Results

| Model | Accuracy | Precision |
|-------|----------|-----------|
| **Multinomial NB** | **~97%** | **~100%** |
| SVC (sigmoid) | ~97% | ~98% |
| Extra Trees | ~97% | ~97% |
| Logistic Regression | ~95% | ~96% |


## Future Work

- [ ] Streamlit web app for live prediction
- [ ] Add F1 and ROC-AUC metrics
- [ ] Cross-validation for robust evaluation
