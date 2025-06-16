# 1. Import library
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# 2. Download resource
nltk.download('punkt')
nltk.download('stopwords')

# 3. Load dataset
df = pd.read_csv('/content/Question-ReferenceAnswer-TestAnswer-TestScore.csv')
df.columns = ['Question', 'Reference Answer', 'Test Answer', 'Test Score']

# 4. Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

df['Processed Answer'] = df['Test Answer'].apply(preprocess)

# 5. Vectorization
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Processed Answer'])

# 6. Topic Modeling using LDA
num_topics = 3  # ubah jumlah topik jika perlu
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# 7. Menampilkan Topik
def display_topics(model, feature_names, num_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopik #{topic_idx + 1}:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names)

# 8. Menampilkan distribusi topik pada jawaban
topic_values = lda.transform(X)
df['Dominant Topic'] = topic_values.argmax(axis=1)

print("\nDistribusi topik dalam jawaban:")
print(df['Dominant Topic'].value_counts())

# 9. (Opsional) Simpan hasil
df[['Test Answer', 'Processed Answer', 'Dominant Topic']].to_csv('topic_modeling_output.csv', index=False)
print("\nFile topic_modeling_output.csv berhasil disimpan.")
