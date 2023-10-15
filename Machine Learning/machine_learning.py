import pandas as pd
import numpy as np
import gensim.downloader as api
import gensim
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import string
import os

# Check if the Word2Vec embeddings file exists
word2vec_file = 'word2vec_google_news_300.bin'

if not os.path.exists(word2vec_file):
    # Download and save the Word2Vec embeddings
    print("Step 0: Downloading Word2Vec embeddings...")
    word2vec_model = api.load("word2vec-google-news-300")
    word2vec_model.save(word2vec_file)
else:
    # Load the saved Word2Vec embeddings
    print("Step 0: Loading Word2Vec embeddings...")
    word2vec_model = gensim.models.KeyedVectors.load(word2vec_file)

# Define the text_to_embedding function
def text_to_embedding(text, embedding_model, embedding_dim):
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()

    words = text.split()
    embeddings = np.zeros((len(words), embedding_dim))

    for i, word in enumerate(words):
        if word in embedding_model:
            embeddings[i, :] = embedding_model[word]
        else:
            embeddings[i, :] = np.zeros(embedding_dim)  # Use a zero vector for out-of-vocabulary words

    if len(words) > 0:
        text_embedding = np.mean(embeddings, axis=0)
    else:
        text_embedding = np.zeros(embedding_dim)  # Handle the case of an empty text

    return text_embedding

# Load the dataset
print("Step 1: Loading the dataset...")
data = pd.read_csv('selected_data.csv')

# Data Preprocessing
print("Step 2: Data Preprocessing...")
# Fill missing values in numerical columns with zeros
data['vlak_min'].fillna(0, inplace=True)
data['vlak_max'].fillna(0, inplace=True)
data['begin_dat'].fillna(0, inplace=True)
data['eind_dat'].fillna(0, inplace=True)

# Fill missing values in text columns with a special token
data['subcategorie'].fillna('<UNK>', inplace=True)
data['object'].fillna('<UNK>', inplace=True)
data['objectdeel'].fillna('<UNK>', inplace=True)

# Text-to-Vector Conversion
print("Step 3: Text-to-Vector Conversion...")


embedding_dim = 300  # Assuming we are using 300-dimensional Word2Vec embeddings
data['subcategorie_emb'] = data['subcategorie'].apply(lambda x: text_to_embedding(x, word2vec_model, embedding_dim))
data['object_emb'] = data['object'].apply(lambda x: text_to_embedding(x, word2vec_model, embedding_dim))
data['objectdeel_emb'] = data['objectdeel'].apply(lambda x: text_to_embedding(x, word2vec_model, embedding_dim))

# Handling "niveau1" Values
print("Step 4: Handling 'niveau1' Values...")
# Split data into rows with non-"Not classified", non-blank "niveau1" values and rows with blank or "Not classified"
labeled_data = data[data['niveau1'] != 'Not classified']
unlabeled_data = data[data['niveau1'].isnull() | (data['niveau1'] == 'Not classified')]

# Data Splitting
print("Step 5: Data Splitting...")
X_train, X_temp, y_train, y_temp = train_test_split(
    labeled_data[['subcategorie_emb', 'object_emb', 'objectdeel_emb', 'vlak_min', 'vlak_max', 'begin_dat', 'eind_dat']],
    labeled_data['niveau1'],
    test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42)

# Model Building
print("Step 6: Model Building...")
# Calculate the maximum length of text embeddings in X_train
max_len = max([len(x) for x in X_train['subcategorie_emb']])

input_dim = max_len * embedding_dim  # Corrected input dimension

# Corrected input shape
model = keras.Sequential([
    keras.layers.Input(shape=(max_len, embedding_dim)),  # Corrected input shape
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(12, activation='softmax')  # 12 possible "niveau1" values
])

# Model Compilation
print("Step 7: Model Compilation...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
print("Step 8: Model Training...")
# Encode the "niveau1" labels into one-hot vectors
encoder = LabelEncoder()
encoder.fit(y_train)
y_train_encoded = encoder.transform(y_train)
y_val_encoded = encoder.transform(y_val)
y_test_encoded = encoder.transform(y_test)

y_train_onehot = keras.utils.to_categorical(y_train_encoded - 1, num_classes=12)  # 12 output neurons
y_val_onehot = keras.utils.to_categorical(y_val_encoded - 1, num_classes=12)
y_test_onehot = keras.utils.to_categorical(y_test_encoded - 1, num_classes=12)

# Convert text embeddings to a consistent shape (pad with zeros if necessary)
def pad_embeddings(embeddings, max_len, embedding_dim):
    if embeddings.shape[0] >= max_len:
        return embeddings[:max_len]
    else:
        padded_embeddings = np.zeros((max_len, embedding_dim))
        padded_embeddings[:embeddings.shape[0]] = embeddings
        return padded_embeddings

X_train_pad = np.array([pad_embeddings(x, max_len, embedding_dim) for x in X_train['subcategorie_emb']])
X_val_pad = np.array([pad_embeddings(x, max_len, embedding_dim) for x in X_val['subcategorie_emb']])
X_test_pad = np.array([pad_embeddings(x, max_len, embedding_dim) for x in X_test['subcategorie_emb']])

# Convert NumPy arrays to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train_pad, dtype=tf.float32)
y_train_onehot_tensor = tf.convert_to_tensor(y_train_onehot, dtype=tf.float32)
X_val_tensor = tf.convert_to_tensor(X_val_pad, dtype=tf.float32)
y_val_onehot_tensor = tf.convert_to_tensor(y_val_onehot, dtype=tf.float32)

# Print the shape of X_train_tensor and y_train_onehot_tensor
print(f"X_train_tensor.shape is {X_train_tensor.shape}")
print(f"y_train_onehot_tensor.shape is {y_train_onehot_tensor.shape}")

# Model Training
print("Step 9: Model Training...")
history = model.fit(X_train_tensor, y_train_onehot_tensor, epochs=50, batch_size=32, validation_data=(X_val_tensor, y_val_onehot_tensor))

# Model Testing
print("Step 10: Model Testing...")
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_onehot)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Prediction for Blank or "Not classified" "niveau1"
print("Step 11: Prediction for Blank or 'Not classified' 'niveau1'...")
unlabeled_inputs = unlabeled_data[['subcategorie_emb', 'object_emb', 'objectdeel_emb', 'vlak_min', 'vlak_max', 'begin_dat', 'eind_dat', 'vondstnummer']]
unlabeled_inputs_pad = np.array([pad_embeddings(x, max_len, embedding_dim) for x in unlabeled_inputs['subcategorie_emb']])
predictions = model.predict(unlabeled_inputs_pad)

# Assign the predicted labels to unlabeled_data['niveau1']
predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1) + 1)
unlabeled_data['niveau1'] = predicted_labels

# Create a DataFrame with 'niveau1' and 'vondstnummer' columns
output_data = unlabeled_data[['niveau1', 'vondstnummer']]

# Save the updated dataset with predicted "niveau1" values
print("Step 12: Saving the updated dataset...")
output_data.to_csv('predicted_dataset.csv', index=False, columns=['niveau1', 'vondstnummer'])
