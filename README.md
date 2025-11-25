NLP-Assignment-2-Sentiment-Analysis
📖 Overview

This project implements a Sentiment Analysis Model using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The goal is to classify text as Positive or Negative using a real, large dataset to ensure high accuracy.

This project includes:

Text Preprocessing

Tokenization

Stopword Removal

TF-IDF Vectorization

Building and Training the ML Model

Evaluating the Model

Predicting sentiment of custom text

🎯 Objective

To build an end-to-end sentiment analysis pipeline using a large, real-world dataset and achieve good accuracy using modern ML techniques.

📂 Dataset Used

We used the NLTK Movie Reviews Dataset, which contains:

Label Count Positive 1000 Negative 1000

Total 2000 labeled reviews, making it balanced and suitable for ML.

This dataset ensures: ✔️ High accuracy ✔️ Reliable training ✔️ Real-world text

⚙️ Steps Performed 1️⃣ Load Dataset

We imported the dataset from NLTK and combined the reviews with their labels.

2️⃣ Data Cleaning

Lowercasing

Removing punctuation

Removing unwanted characters

3️⃣ Feature Extraction (TF-IDF)

We converted text into numerical features using TfidfVectorizer for better model performance.

4️⃣ Training the Model

We used Logistic Regression, which works extremely well for text classification.

5️⃣ Evaluating the Model

We calculated:

Accuracy

Confusion Matrix

Predictions on new inputs

Expected accuracy: 80–92%

6️⃣ Predicting Custom Sentences

The model can analyze sentiment of any user-typed sentence.

📊 Technologies Used

Python

NLP (NLTK)

Scikit-learn

Machine Learning

TF-IDF Vectorization

Logistic Regression

Google Colab / Jupyter Notebook

▶️ How to Run This Project Option A — Google Colab

Open the Colab link

Run all cells

Upload custom sentences to get predictions

Option B — Local Machine pip install nltk scikit-learn

Then run the Python notebook.

📁 Repository Structure Project-2-Sentiment-Analysis/ │ ├── sentiment_analysis.ipynb # Google Colab Notebook ├── README.md # Documentation └── dataset/ (Not required — dataset is loaded from NLTK)
