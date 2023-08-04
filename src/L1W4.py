import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a, x


def ReLU(x):
    a = np.maximum(0, x)
    return a, x


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    np.random.seed(666)
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters["W" + str(l)] = (
                np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        )
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        assert parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layer_dims[l], 1)
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = ReLU(Z)
    # 抛出异常
    else:
        raise Exception("Non-supported activation function")
    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward()
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1).
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            activation="relu",
        )
        caches.append(cache)
    # Implement LINEAR -> SIGMOID.
    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
    )
    caches.append(cache)
    assert AL.shape == (1, X.shape[1])
    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector, shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    sum_cost = np.dot(Y, np.log(AL + 1e-5).T) + np.dot((1 - Y), np.log(1 - AL + 1e-5).T)
    cost = -sum_cost / m
    cost = np.squeeze(cost)
    assert cost.shape == ()
    return cost


def sigmoid_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    activation_cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    assert dA.shape == activation_cache.shape
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def ReLU_backward(dA, activation_cache):
    """
    Implement the backward propagation for a single ReLU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    activation_cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    assert dA.shape == activation_cache.shape
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),
                same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)
    assert dA_prev.shape == A_prev.shape
    dW = 1 / m * np.dot(dZ, A_prev.T)
    assert dW.shape == W.shape
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    assert db.shape == b.shape
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache)
             stored for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),
                same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    assert dA.shape == cache[1].shape  # cache[1] is z
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = ReLU_backward(dA, activation_cache)
    # 抛出异常
    else:
        raise Exception("Non-supported activation function")
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu"
                the cache of linear_activation_forward() with "sigmoid"

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # Lth layer (SIGMOID -> LINEAR) gradients.
    current_cache = caches[
        L - 1
        ]  # the last layer, which its activation function is sigmoid
    (
        grads["dA" + str(L)],
        grads["dW" + str(L)],
        grads["db" + str(L)],
    ) = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 2)], current_cache, activation="relu"
        )
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate=0.05):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing parameters
    grads -- python dictionary containing gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2  # the number of layers
    # Update rule for each parameter
    for l in range(L):
        name_w = "W" + str(l + 1)
        name_b = "b" + str(l + 1)
        parameters[name_w] = parameters[name_w] - learning_rate * grads["d" + name_w]
        parameters[name_b] = parameters[name_b] - learning_rate * grads["d" + name_b]
    return parameters


def L_layer_model(
        X,
        Y,
        layers_dims,
        learning_rate=0.0075,
        num_iterations=3000,
        print_cost=False,
        print_every=1000,
):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    Y -- true "label" vector
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    print_every -- print the cost every print_every iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    # Loop (gradient descent)
    for i in range(1, num_iterations + 1):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        # Compute cost
        cost = compute_cost(AL, Y)
        assert cost.shape == ()
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % print_every == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if i % print_every == 0:
            costs.append(cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


def score(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network and returns the accuracy.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    accuracy -- accuracy of the model on the given set X
    """
    m = X.shape[1]
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    # convert probas to 0/1 predictions
    p = np.zeros((1, m))
    p[probas > 0.5] = 1
    # print results
    # print("Accuracy: " + str(np.sum(p == y) / m))
    accuracy = np.sum(p == y) / m
    return accuracy


def predict(X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network and returns the accuracy.

    Arguments:
    X -- data set of examples you would like to label, X.shape[0] is the number of features.
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given set X
    """
    if X.shape[0] != parameters["W1"].shape[1] & X.shape[1] != parameters["W1"].shape[1]:
        raise ValueError(
            "The number of features of X must be equal to the number of rows of W1, which is 13 in this case."
        )
    if X.shape[1] == parameters["W1"].shape[1]:
        X = X.T
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    # convert probas to 0/1 predictions
    p = np.zeros((1, X.shape[1]))
    p[probas > 0.5] = 1
    return p
