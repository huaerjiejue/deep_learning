import numpy as np
from src import L2W1

def test_dictionary_to_vector():
    w1 = np.array([[1, 2, 3], [4, 5, 6]])
    b1 = np.array([[1], [2]])
    w2 = np.array([[7, 8], [9, 10], [11, 12]])
    b2 = np.array([[3], [4], [5]])
    parameters = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    expected = np.array([1, 2, 3, 4, 5, 6, 1, 2, 7, 8, 9, 10, 11, 12, 3, 4, 5]).reshape((17, 1))
    actual = L2W1.dictionary_to_vector(parameters)
    assert np.allclose(expected, actual)


def test_vector_to_dictionary():
    vector = np.array([[1, 2, 3, 4, 5, 6, 1, 2, 7, 8, 9, 10, 11, 12, 3, 4, 5]]).reshape((17, 1))
    expected = {'W1': np.array([[1, 2, 3], [4, 5, 6]]), 'b1': np.array([[1], [2]]),
                'W2': np.array([[7, 8], [9, 10], [11, 12]]), 'b2': np.array([[3], [4], [5]])
                }
    actual = L2W1.vector_to_dictionary(vector, [3, 2, 3])
    assert np.allclose(expected['W1'], actual['W1'])
    assert np.allclose(expected['b1'], actual['b1'])
    assert np.allclose(expected['W2'], actual['W2'])
    assert np.allclose(expected['b2'], actual['b2'])






