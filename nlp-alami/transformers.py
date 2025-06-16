# 1. Import Library
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
# InputExample and InputFeatures are not strictly needed for this approach
# from transformers import InputExample, InputFeatures

# 2. Load Dataset
df = pd.read_csv('hasil_klasifikasi_text.csv')
df = df[['Processed Test Answer', 'Category']].dropna()

# 3. Encode Label
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])

# 4. Split Data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Processed Test Answer'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# 5. Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 6. Tokenisasi Data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# 7. Convert ke TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    list(train_labels)
)).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    list(val_labels)
)).batch(16)

# 8. Load Model BERT
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=50)

# 9. Compile Model
model.compile(
    # Pass the optimizer name as a string and configure its parameters
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# You can still configure the learning rate like this if needed:
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# model.compile(
#     optimizer=optimizer,
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy']
# )


# 10. Train Model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# 11. Evaluasi Model
val_pred = model.predict(val_dataset)
y_pred = np.argmax(val_pred.logits, axis=1)

print("\nClassification Report:")
print(classification_report(val_labels, y_pred, target_names=label_encoder.classes_))