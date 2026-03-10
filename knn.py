import numpy as np
from collections import Counter


# -------- Distance Functions -------- #

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)

    if normA == 0 or normB == 0:   # avoid division by zero
        return 0

    return dot / (normA * normB)


def hamming_distance(a, b):
    return np.sum(a != b)


def minkowski_distance(a, b, p=3):
    return np.power(np.sum(np.abs(a - b) ** p), 1 / p)




def knn(X_train, y_train, x_test, k, distance_func, similarity=False):
    
    values = []

    for i in range(len(X_train)):
        val = distance_func(X_train[i], x_test)
        values.append((val, y_train[i]))

    # similarity → larger value is better
    if similarity:
        values.sort(reverse=True)
    else:
        values.sort()

    neighbors = [values[i][1] for i in range(k)]

    return Counter(neighbors).most_common(1)[0][0]




X_train = np.array([[1,2],[2,3],[3,4],[6,7]])
y_train = ['A','A','A','B']
x_test = np.array([2,2])
k = 3


# -------- Running KNN with different measures -------- #

print("Euclidean Distance:", knn(X_train, y_train, x_test, k, euclidean_distance))

print("Manhattan Distance:", knn(X_train, y_train, x_test, k, manhattan_distance))

print("Cosine Similarity:", knn(X_train, y_train, x_test, k, cosine_similarity, similarity=True))

print("Hamming Distance:", knn(X_train, y_train, x_test, k, hamming_distance))

print("Minkowski Distance:", knn(X_train, y_train, x_test, k, lambda a,b: minkowski_distance(a,b,3)))