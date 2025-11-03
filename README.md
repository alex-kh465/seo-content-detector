# SEO Content Quality & Duplicate Detector

A machine learning-powered tool for analyzing SEO content quality and detecting near-duplicate pages. Built with Python, Streamlit, and scikit-learn.

## Overview

This project provides an end-to-end pipeline for analyzing web content quality through NLP features (readability, word count, sentence structure) and detecting duplicate or near-duplicate content using semantic embeddings. The Streamlit interface enables real-time URL analysis with instant quality predictions and similarity detection.

## Features

### Core Analysis
- **HTML Parsing**: Extract clean text from HTML content
- **NLP Analysis**: Compute readability scores, keyword extraction, and semantic embeddings
- **Duplicate Detection**: Identify near-duplicate content using cosine similarity (threshold: 0.80)
- **Quality Classification**: RandomForest model trained on synthetic labels (High/Medium/Low)
- **Real-Time Analysis**: Live URL scraping and instant quality assessment
- **Interactive Dashboard**: Clean Streamlit UI with metrics, flags, and similarity tables

### Advanced NLP Features
- **Sentiment Analysis**: VADER-based sentiment scoring with positive/negative/neutral classification
- **Named Entity Recognition**: Extract and categorize people, organizations, locations, and other entities
- **Topic Modeling**: LDA-based unsupervised topic discovery across content
- **Enhanced Keyword Extraction**: TF-IDF with bigrams for improved keyword identification
- **Comprehensive Visualizations**: Word clouds, similarity heatmaps, feature distributions, confusion matrices

## Setup Instructions

```bash
git https://github.com/alex-kh465/seo-content-detector
cd seo-content-detector
uv pip install -r requirements.txt
```

**Run the pipeline:**
```bash
# Parse HTML content
python streamlit_app/utils/parser.py

# Extract features
python streamlit_app/utils/features.py

# Detect duplicates
python streamlit_app/utils/similarity.py

# Train quality model
python streamlit_app/utils/scorer.py
```

**Launch Streamlit app:**
```bash
streamlit run streamlit_app/app.py
```

**Jupyter Notebook:**
Open `notebooks/seo_pipeline.ipynb` to run the complete pipeline interactively.

## Quick Start

1. Place your dataset (`url`, `html_content` columns) in `data/data.csv`
2. Run the pipeline scripts in order (parser → features → similarity → scorer)
3. Launch the Streamlit app to analyze new URLs

## Key Decisions

**Why 0.80 similarity threshold?** Balances precision and recall for duplicate detection. Lower thresholds (0.70) capture more near-duplicates but increase false positives.

**Why RandomForest?** Robust to overfitting, handles non-linear relationships between features, and provides interpretable feature importance. Compared against rule-based baseline and achieved +5-10% accuracy improvement.

**Quality Labels**: Synthetic labels based on domain knowledge (High: 1500+ words with 50-70 readability; Low: <500 words or <30 readability). Enables supervised learning without manual annotation.

**Embeddings Model**: SentenceTransformer `all-MiniLM-L6-v2` provides fast, high-quality semantic embeddings (384 dimensions) suitable for similarity comparison.

## Results Summary

### Model Performance (81 pages, 70/30 split)
- **Dataset Distribution**: 49 Low, 18 Medium, 14 High quality pages
- **RandomForest Accuracy**: 92.0% on test set
- **F1 Score**: 0.91 (weighted average)
- **Baseline Accuracy**: 68.0% (rule-based predictor)
- **Improvement**: +24.0% accuracy, +23.1% F1 score over baseline

### Feature Importance
- **Flesch Reading Ease**: 40.2% (most important)
- **Word Count**: 30.5%
- **Sentence Count**: 29.3%

### Per-Class Performance
- **High Quality**: Precision 1.00, Recall 0.50, F1 0.67
- **Medium Quality**: Precision 0.75, Recall 1.00, F1 0.86
- **Low Quality**: Precision 1.00, Recall 1.00, F1 1.00

### Duplicate Detection
- Threshold 0.80 identifies high-similarity pairs
- Cosine similarity computed on 384-dimensional embeddings
- Thin content detection: <500 words flagged

## Limitations

- Synthetic labels may not reflect actual content quality in all domains
- HTML parsing may miss content from JavaScript-heavy sites
- Embeddings require GPU for large-scale processing (CPU fallback is slower)
- Similarity detection assumes semantic meaning correlates with duplication
- No multi-language support (English only via NLTK stopwords)
- Model trained on static dataset—quality criteria may vary by industry

## Project Structure

```
seo-content-detector/
├── data/                    # CSV outputs
├── notebooks/               # Jupyter analysis notebook
├── streamlit_app/
│   ├── app.py              # Main Streamlit UI
│   ├── utils/              # Core modules
│   └── models/             # Trained model
├── requirements.txt
└── README.md
```
