# !pip install sumy
# !pip install nltk
# !pip install transformers sentencepiece

from transformers import pipeline
import pandas as pd

# Load dataset
df = pd.read_csv("Question-ReferenceAnswer-TestAnswer-TestScore.csv")
df.columns = ['Question', 'Reference Answer', 'Test Answer', 'Test Score']

# Load summarizer
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Ringkasan hanya 10 pertama untuk efisiensi
summarized_results = []
for answer in df['Test Answer'][:10]:
    input_text = "summarize: " + str(answer)
    summary = summarizer(input_text, max_length=30, min_length=5, do_sample=False)[0]['summary_text']
    summarized_results.append(summary)

# Tambah ke DataFrame
df['Summary'] = summarized_results + [''] * (len(df) - len(summarized_results))

# Simpan hasil
df.to_csv("abstractive_summarized_answers.csv", index=False)
