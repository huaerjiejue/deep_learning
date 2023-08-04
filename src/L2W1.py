import numpy as np


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    L = len(layer_dims) - 1  # number of layers in the network
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(AL + 1e-5) + (1 - Y) * np.log(1 - AL + 1e-5))
    return cost


def compute_cost_with_regularization(AL, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    parameters -- python dictionary containing parameters of the model
    lambd -- regularization hyperparameter, scalar

    Returns:
    cost - value of the regularized loss function
    """
    m = Y.shape[1]
    L = len(parameters) // 2
    cross_entropy_cost = compute_cost(AL, Y)
    L2_regularization_cost = 0
    for l in range(1, L + 1):
        L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
    L2_regularization_cost = lambd / (2 * m) * L2_regularization_cost
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL"
                  Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                  bl -- bias vector of shape (layer_dims[l], 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() with 'D' instead of 'A' where applicable,
                the cache of linear_sigmoid_forward() with 'D' instead of 'A'
    """
    np.random.seed(1)
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = np.maximum(0, Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < keep_prob
        A = np.multiply(A, D)
        A /= keep_prob
        cache = (A_prev, W, b, Z, D)
        caches.append(cache)
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z = np.dot(W, A) + b
    AL = 1 / (1 + np.exp(-Z))
    cache = (A, W, b, Z)
    caches.append(cache)
    return AL, caches


def backward_propagation_with_dropout(X, Y, cache, keep_prob=0.5):
    """
        Implements the backward propagation of our baseline model to which we added dropout.

        Arguments:

        X -- input dataset, of shape (2, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation_with_dropout()
        keep_prob - probability of keeping a neuron active during drop-out, scalar

        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1 * D1  # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    s = np.maximum(0, x)
    return s


def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL".

    Returns:
    theta -- vector containing all the parameters.
    """
    L = len(parameters) // 2
    theta = np.zeros((0, 1))
    for l in range(1, L + 1):
        theta = np.concatenate((theta, parameters['W' + str(l)].reshape(-1, 1)), axis=0)
        theta = np.concatenate((theta, parameters['b' + str(l)].reshape(-1, 1)), axis=0)
    return theta


def vector_to_dictionary(theta, layer_dims):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.

    Arguments:
    theta -- vector containing all the parameters.
    layer_dims -- python list containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL".
    """
    parameters = {}
    L = len(layer_dims) - 1
    start = 0
    for l in range(1, L + 1):
        parameters['W' + str(l)] = theta[start:start + layer_dims[l] * layer_dims[l - 1]].reshape(
            (layer_dims[l], layer_dims[l - 1]))
        start += layer_dims[l] * layer_dims[l - 1]
        parameters['b' + str(l)] = theta[start:start + layer_dims[l]].reshape((layer_dims[l], 1))
        start += layer_dims[l]
    return parameters


def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.

    Arguments:
    gradients -- python dictionary containing your gradients "dW1", "db1", ..., "dWL", "dbL".

    Returns:
    theta -- vector containing all the parameters.
    """
    L = len(gradients) // 3
    theta = np.zeros((0, 1))
    for l in range(1, L + 1):
        theta = np.concatenate((theta, gradients['dW' + str(l)].reshape(-1, 1)), axis=0)
        theta = np.concatenate((theta, gradients['db' + str(l)].reshape(-1, 1)), axis=0)
    return theta



