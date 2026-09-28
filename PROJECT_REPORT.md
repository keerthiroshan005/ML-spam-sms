# ML Assessment-1 Project Report
## SMS Spam Detection using NLP and Classification

### Abstract
This project develops an NLP-based machine-learning system that classifies SMS messages as spam or legitimate. The system uses the UCI SMS Spam Collection, TF-IDF feature engineering, and Logistic Regression. A Naive Bayes model is implemented as a baseline and an improved word+character TF-IDF model is compared against it. A Streamlit interface provides real-time predictions.

### Problem Identification — 10 marks
Spam SMS can contain advertisements, fraudulent offers, or unwanted messages. Manual filtering does not scale. The proposed system automatically classifies incoming messages into two classes: spam and ham.

### Dataset & Preprocessing — 15 marks
The UCI SMS Spam Collection contains 5,574 labeled SMS messages. The dataset has no reported missing values. Duplicate rows are removed, labels are normalized, and text is transformed with TF-IDF. A stratified 80/20 train-test split is used.

### EDA & Visualization — 10 marks
The project generates class-distribution, message-length, and word-count visualizations. These reveal class imbalance and differences in text characteristics between spam and legitimate messages.

### ML Algorithm Implementation — 20 marks
Two classifiers are implemented. The baseline is Multinomial Naive Bayes with unigram TF-IDF. The improved model uses Logistic Regression with a FeatureUnion containing word TF-IDF (unigrams/bigrams) and character TF-IDF (3–5 character n-grams).

### Model Evaluation — 10 marks
Accuracy, precision, recall, F1-score, and confusion matrix are calculated on the held-out test set. F1 is used as the main selection criterion because spam classification is imbalanced.

### Model Improvement — 10 marks
The improved pipeline adds richer n-grams, character-level features, sublinear TF scaling, class balancing, and Logistic Regression regularization. The training script automatically records both models' metrics and saves the best model.

### Application/UI — 10 marks
A Streamlit web application accepts a message and displays the predicted class and spam probability.

### GitHub Repository — 5 marks
The repository contains source code, requirements, dataset download script, model training, EDA, UI, documentation and deployment files.

### Deployment — 5 marks
The application can be deployed on Streamlit Community Cloud or run in Docker.

### Presentation & Viva — 5 marks
Recommended presentation flow: problem → dataset → preprocessing → EDA → baseline → improved model → evaluation → UI → deployment → conclusion.

### Conclusion
The project demonstrates an end-to-end machine-learning workflow for a practical NLP classification problem, from dataset acquisition and preprocessing through model comparison, evaluation and deployment.
