import numpy as np
from swimnetworks import Dense

'''
Collection of predefined activation functions and their derivatives used by Ansatz classes
'''

def tanh_activation(x):
    return np.tanh(x)

def tanh_x(x, r=0.05):
    """First derivative of tanh.
    """
    x = np.clip(x, -10, 10)
    return 1/np.cosh(x)**2

def tanh_xx(x):
    """Second derivative of tanh.
    """
    x = np.clip(x, -10, 10)
    return -2*np.sinh(x) / np.cosh(x)**3

def tanh_xxx(x):
    """third derivative of tanh.
    """
    x = np.clip(x, -10, 10)
    return 4 * np.tanh(x) ** 2 / np.cosh(x)**2 - 2 * np.cosh(x)**4

def tanh_xxxx(x):
    """Fourth derivative of tanh.
    """
    x = np.clip(x, -10, 10)
    return 16. * np.sinh(x) / np.cosh(x)**5 - 8. * (np.sinh(x)**3) / np.cosh(x)**5

def sin_activation(x):
    return np.sin(x)

def sin_x(x):
    return np.cos(x)

def sin_xx(x):
    return -np.sin(x)

def cos_activation(x):
    return np.cos(x)

def cos_x(x):
    return -np.sin(x)

def cos_xx(x):
    return -np.cos(x)

def relu_activation(x):
    return np.maximum(x, 0)

def relu_x(x):
    return np.where(x < 0., 0., 1.)

def relu_xx(x):
    return np.zeros_like(x)


# define dicts for known activation functions and their derivatives etc
# to simplify processing of input arguments
activations = {
    "tanh": tanh_activation,
    "relu": relu_activation,
    "sin": sin_activation,
    "cos": cos_activation,
}

activations_x = {
    "tanh": tanh_x,
    "relu": relu_x,
    "sin": sin_x,
    "cos": cos_x,
}

activations_xx = {
    "tanh": tanh_xx,
    "relu": relu_xx,
    "sin": sin_xx,
    "cos": cos_xx,
}

activations_xxx = {
    "tanh": tanh_xxx,
}

activations_xxxx = {
    "tanh": tanh_xxxx,
}

# match activation functions to parameter samples. 
# This is necessary because there is (currently) not a sampler in the swimnetworks package for every activation function here.
parameter_samplers = {
    "tanh": "tanh",
    "relu": "relu",
    "sin": "tanh",
    "cos": "tanh",
}