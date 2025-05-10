import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load text data from dataset.txt
with open('food_dataset.txt', 'r', encoding='utf-8') as file:
    text_data = file.readlines()
text_data = [line.strip() for line in text_data if line.strip()]

# Tokenization and preparation
tokenizer = Tokenizer(oov_token="<OOV>")  # Added <OOV> token for unseen words
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

# Save the tokenizer for later use
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Create input sequences and the corresponding next words
input_sequences = []
for line in text_data:
    tokenized_line = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokenized_line)):
        input_sequences.append(tokenized_line[:i+1])

# Pad sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Separate X (inputs) and Y (outputs)
X, Y = input_sequences[:, :-1], input_sequences[:, -1]
Y = tf.keras.utils.to_categorical(Y, num_classes=total_words)

# Define an improved LSTM model
model = Sequential([
    Embedding(input_dim=total_words, output_dim=150, input_length=X.shape[1]),
    Bidirectional(LSTM(256, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(256),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(total_words, activation='softmax')
])

# Compile the model with an adaptive learning rate
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.002), metrics=['accuracy'])

# Implement adaptive learning rate scheduling
lr_scheduler = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, verbose=1)

# Train the model with an increased number of epochs
model.fit(X, Y, epochs=100, batch_size=64, callbacks=[lr_scheduler])

# Save the trained model
model.save('text_generation_model.keras')
