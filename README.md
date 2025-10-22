# Machine Learning-Based Web Page Categorization

**Project Creator:** Shakiba Zakeri Nejad

## Overview

This project focuses on categorizing web pages based on their textual content (title, description, h1, h2 tags) using the YektANet dataset. A machine learning pipeline involving text preprocessing, feature extraction, and classification was developed to predict the category of a given web page.

## Key Results

* **F1-Score (Test Set):** 97% (weighted average)

## Methodology

The project followed these main steps:
1.  **Data Loading & Preparation:** The Yektanet training dataset (`yektanet_train.csv`) was loaded. Irrelevant columns (`id`, `domain`, `url`, `text_content`) were dropped.
2.  **Handling Class Imbalance:** The dataset showed significant class imbalance. `RandomOverSampler` from `imblearn` was used on the training data to create a balanced dataset for model training.
3.  **Text Preprocessing:**
    * Persian text normalization and tokenization were performed using the `hazm` library (`Normalizer`, `word_tokenize`).
    * Punctuation was removed.
    * Persian stopwords (from `stopwords_guilannlp`) were removed.
    * Digits were removed.
4.  **Feature Extraction:** A pipeline combined `CountVectorizer` (with the custom preprocessor and tokenizer, using unigrams and bigrams, `min_df=5`) and `TfidfTransformer` (with `sublinear_tf=True`) to convert the concatenated text features (`title`, `h1`, `description`, `h2`) into TF-IDF vectors.
5.  **Model Selection & Training:** Several classification algorithms were evaluated (LinearSVC, RandomForest, GradientBoosting, LogisticRegression, KNeighbors, MultinomialNB, DecisionTree, MLPClassifier). The `MLPClassifier` achieved the highest weighted F1-score.
6.  **Evaluation:** The best-performing model (`MLPClassifier` within the pipeline) was evaluated on a held-out test set (10% of the oversampled data) using the weighted F1-score.

## Dataset

* **Source:** Yektanet competition dataset (`yektanet_train.csv`).
* **Target Variable:** `category` (22 distinct classes).
* **Features Used:** `title`, `h1`, `description`, `h2`.

## Installation

Clone the repository and install the necessary dependencies using the `requirements.txt` file:

```bash
git clone [URL-of-your-repository]
cd [repository-directory]
pip install -r requirements.txt

Results
The final pipeline using MLPClassifier achieved a weighted F1-score of approximately 0.97 on the test set, indicating high effectiveness in categorizing the web pages based on the selected textual features.
