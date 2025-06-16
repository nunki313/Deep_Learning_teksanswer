# -*- coding: utf-8 -*-
"""
PROYEK SUB CPMK 3: TEXT CLASSIFICATION SAJA
"""

# 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# 2. DOWNLOAD NLTK RESOURCE
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'], quiet=True)

# 3. LOAD DATASET CSV TANPA UNDERSCORE
file_path = '/content/Question-ReferenceAnswer-TestAnswer-TestScore.csv'
df = pd.read_csv(file_path)
df.columns = ['Question', 'Reference Answer', 'Test Answer', 'Test Score']

# 4. PREPROCESSING TEXT
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def preprocess_text(text):
    tokens = tokenizer.tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

df['Processed Test Answer'] = df['Test Answer'].apply(preprocess_text)

# 5. BUAT LABEL KATEGORI UNTUK KLASIFIKASI
def categorize(score):
    if score >= 8:
        return 'Excellent'
    elif score >= 7:
        return 'Good'
    else:
        return 'Fair'

df['Category'] = df['Test Score'].apply(categorize)

# 6. TF-IDF VECTORIZER
tfidf = TfidfVectorizer(max_features=200)
X = tfidf.fit_transform(df['Processed Test Answer'])
y = df['Category']

# 7. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 8. MODEL DAN EVALUASI
print("## TEXT CLASSIFICATION (Kategori: Fair, Good, Excellent) ##")

models = {
    'Linear SVM': LinearSVC(random_state=42, max_iter=10000),
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))

# 9. SIMPAN HASIL
df.to_csv('hasil_klasifikasi_text.csv', index=False)
print("\nFile hasil disimpan: hasil_klasifikasi_text.csv")
