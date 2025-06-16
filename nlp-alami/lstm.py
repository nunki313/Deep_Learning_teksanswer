# 1. Import Library
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 2. Load Dataset
df = pd.read_csv('hasil_klasifikasi_text.csv')  # File yang dihasilkan dari preprocessing sebelumnya
df = df[['Processed Test Answer', 'Category']].dropna()

# 3. Encode Label
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])

# 4. Tokenisasi & Padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Processed Test Answer'])
sequences = tokenizer.texts_to_sequences(df['Processed Test Answer'])
max_len = max([len(x) for x in sequences])
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    padded, df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 6. Bangun Model LSTM
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 kelas: Excellent, Good, Fair
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 7. Train Model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32,
    verbose=2
)

# 8. Evaluasi Model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# 9. Plot Akurasi
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
