# SMS Shield — Spam Message Detection using NLP + Machine Learning

## 1. Problem Identification
SMS spam wastes users' time, can contain scams, and may expose users to malicious links. The objective is to build a binary text-classification system that predicts whether an SMS is **spam** or **ham (legitimate)**.

## 2. Dataset
This project uses the **UCI SMS Spam Collection**, containing 5,574 labeled SMS messages. It is a public dataset intended for spam-filtering research and is licensed CC BY 4.0.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms

Run `python download_dataset.py` to fetch the dataset into `data/`.

## 3. Preprocessing
- Remove duplicates and missing records
- Normalize labels
- Keep the original message text for TF-IDF
- TF-IDF word n-grams (1–2)
- TF-IDF character n-grams (3–5)
- Train/test split with stratification

## 4. Algorithms
### Baseline
Multinomial Naive Bayes + unigram TF-IDF.

### Improved model
Logistic Regression + FeatureUnion of word and character TF-IDF features, with `class_weight="balanced"`, bigrams, character n-grams and sublinear TF scaling.

## 5. Evaluation
Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The training script automatically compares the baseline and improved model and selects the model with the highest F1 score.

## 6. EDA
Run:
```bash
python eda.py
```
Charts are written to `reports/`.

## 7. Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_dataset.py
python train.py
python eda.py
streamlit run app.py
```

## 8. Deployment — Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload this project.
3. Ensure `requirements.txt`, `app.py`, `download_dataset.py`, `train.py`, `src/`, `data/` and `models/` are committed.
4. If you do not commit the binary dataset/model, run training before deployment or modify the app to train/download on first launch.
5. In Streamlit Community Cloud, choose the repository and `app.py` as the entry point.

For a simple classroom submission, commit the trained `models/spam_classifier.joblib` and the dataset after checking your repository size/licensing requirements.

## 9. Suggested Git commands
```bash
git init
git add .
git commit -m "Initial SMS spam detection ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sms-spam-ml.git
git push -u origin main
```

## 10. Viva points
- **Why NLP?** The input is unstructured text.
- **Why TF-IDF?** It converts text into numerical features while down-weighting very common terms.
- **Why character n-grams?** They capture misspellings, obfuscated words, URLs and SMS-specific patterns.
- **Why Logistic Regression?** It is a strong, interpretable baseline for high-dimensional sparse text features.
- **Why F1?** Spam data is imbalanced, so accuracy alone can hide poor minority-class performance.
- **Why train/test split?** To estimate performance on unseen messages.
- **What is model improvement?** Moving from unigram Naive Bayes to combined word+character TF-IDF and Logistic Regression.

## Academic citation
Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. DOI: 10.24432/C5CC84.
