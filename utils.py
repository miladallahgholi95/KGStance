import numpy as np

def average_vectors(vectors):
    vectors_array = np.array(vectors)
    average_vector = np.mean(vectors_array, axis=0)
    return average_vector.tolist()
