import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()
for key in word_index:
    word_index[key] += 3
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# max_val = np.max(np.concatenate((
#     [len(entry) for entry in train_data], 
#     [len(entry) for entry in test_data]
# )))
max_val = 250

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=max_val)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=max_val)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])

def decode_reviews(texts):
    return [decode_review(text) for text in texts]


