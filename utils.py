import numpy as np

def average_vectors(vectors):

    if len(vectors) == 1:
        return vectors[0]

    vectors_array = np.array(vectors)
    average_vector = np.mean(vectors_array, axis=0)
    return average_vector.tolist()
