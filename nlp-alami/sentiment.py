# 1. IMPORT LIBRARY
import nltk
nltk.download('punkt_tab', quiet=True)

import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 2. DOWNLOAD NLTK DATA (jika belum)
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'], quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 3. LOAD DATASET
file_path = '/content/Question-ReferenceAnswer-TestAnswer-TestScore.csv'
df = pd.read_csv(file_path)
df.columns = ['Question', 'Reference Answer', 'Test Answer', 'Test Score']

# 4. BUAT LABEL SENTIMEN DARI SKOR
def score_to_sentiment(score):
    if score >= 8:
        return 'positive'
    elif score >= 6:
        return 'neutral'
    else:
        return 'negative'

df['Sentiment'] = df['Test Score'].apply(score_to_sentiment)

# 5. PREPROCESSING TEXT
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

df['Processed Answer'] = df['Test Answer'].apply(preprocess_text)

# 6. TF-IDF VEKTORISASI
vectorizer = TfidfVectorizer(max_features=200)
X = vectorizer.fit_transform(df['Processed Answer'])
y = df['Sentiment']

# 7. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 8. TRAIN MODEL
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 9. EVALUASI
print("=== HASIL SENTIMENT ANALYSIS ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. VISUALISASI DISTRIBUSI SENTIMEN
plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df, palette='Set2')
plt.title("Distribusi Sentimen Berdasarkan Skor Jawaban")
plt.show()

# 11. SIMPAN FILE HASIL
df.to_csv("hasil_sentiment.csv", index=False)
print("\nHasil disimpan ke: hasil_sentiment.csv")
