import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
import numpy as np
import pickle
from utils import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dropout

# read data
with open('train_data.pkl', 'rb') as file:
    train_data_json = pickle.load(file)

with open('test_data.pkl', 'rb') as file:
    test_data_json = pickle.load(file)

# functions
def embedding_validation(emb):
    return isinstance(emb, list) and len(emb) == 768


# model
bert_embedding_dim = len(train_data_json[0]["embedding"])
target_embedding_dim = len(train_data_json[0]["target_embedding"])

bert_input = Input(shape=(bert_embedding_dim,), name='bert_input')
target_input = Input(shape=(target_embedding_dim,), name='target_input')
combined = Concatenate()([bert_input, target_input])
hidden1 = Dense(128, activation='relu')(combined)
hidden1 = Dropout(0.5)(hidden1)
hidden2 = Dense(64, activation='relu')(hidden1)
hidden1 = Dropout(0.5)(hidden1)
hidden3 = Dense(16, activation='relu')(hidden2)
hidden1 = Dropout(0.5)(hidden1)
hidden4 = Dense(8, activation='relu')(hidden3)
hidden1 = Dropout(0.5)(hidden1)
output = Dense(3, activation='softmax')(hidden4)
model = Model(inputs=[bert_input, target_input], outputs=output)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# train data preparation
bert_embeddings = np.array([d['embedding'] for d in train_data_json])
target_embeddings = np.array([d['target_embedding'] for d in train_data_json])
labels = np.array([d['Label'] for d in train_data_json])

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# fit model
history = model.fit(
    [bert_embeddings, target_embeddings],
    labels,
    epochs=100,
    batch_size=8
)

# test data preparation
test_labels, test_bert_embeddings, test_target_embeddings = [], [], []
for row in test_data_json:
    embedding_fields = ["text_embedding"]

    bert_embeddings = []
    for field in embedding_fields:
        if embedding_validation(row[field]):
            bert_embeddings.append(row[field])

    if bert_embeddings:
        test_bert_embeddings.append(np.array(average_vectors(bert_embeddings)))
        test_target_embeddings.append(np.array(row['target_embedding']))
        test_labels.append(row['Label'])

test_labels_encoded = label_encoder.transform(test_labels)

# predict model
predictions = model.predict([np.array(test_bert_embeddings), np.array(test_target_embeddings)])
predicted_classes = np.argmax(predictions, axis=1)

# calculate F1-Measure
f1 = f1_score(test_labels_encoded, predicted_classes, average='weighted')
print(f"F1-Measure: {f1}")
