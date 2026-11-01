from dataclasses import dataclass
from typing import Callable, Union
from abc import ABC, abstractmethod

from .ansatz import Ansatz
# from swimpde.utils import activations, activations_x, activations_xx, parameter_samplers
from swimpde.utils import activations, activations_x, activations_xx, activations_xxx, activations_xxxx, parameter_samplers
from swimnetworks import Dense

from sklearn.pipeline import Pipeline
import numpy as np

@dataclass
class BasicAnsatz(Ansatz): 
    '''
    Ansatz representing a simple neural network with a linear transformation followed by a single dense layer.

    The basis functions all have the form psi(wx + b), where psi is any activation function.

    Attributes:
    ----------
    activation: Union[str, Callable]
        scalar function to use as activation function. 
        If the function is not a predefined one (passed as string), the derivatives relevant to the solver and a parameter sampler need to be provided as callable, 
        otherwise they can be automatically deduced
    activation_x: Callable
        first derivative of the activation function
    activation_xx: Callable
        second derivative of the activation function
    parameter_sampler: Union[str, Callable]
        parameter sampler to use in the SWIM algorithm (see the SWIM package for possible options)
    n_neurons: int
        number of basis functions/neurons
    random_state: int
        random state to use in the parameter sampling to allow reproducability

    '''

    activation: Union[str, Callable]
    activation_x: Callable = None
    activation_xx: Callable = None
    activation_xxx: Union[Callable, str] = None
    activation_xxxx: Union[Callable, str] = None
    parameter_sampler: Union[str, Callable] = None
    
    n_neurons: int = 1024

    random_state: int = 1
    regularization_scale: float = 1e-13
    
    def __post_init__(self):
        # deduce the activation derivatives and sampler in case the activation function is a predefined one
        if isinstance(self.activation, str):
            try:
                if self.activation_x is None:
                    self.activation_x = activations_x[self.activation]
                if self.activation_xx is None:
                    self.activation_xx = activations_xx[self.activation]
                if self.activation_xxx is None:
                    self.activation_xxx = activations_xxx[self.activation]
                if self.activation_xxxx is None:
                    self.activation_xxxx = activations_xxxx[self.activation]

                if self.parameter_sampler is None:
                    self.parameter_sampler = parameter_samplers[self.activation]
                
                self.activation = activations[self.activation]
            except KeyError:
                raise ValueError(f"Unknown activation {self.activation}.")
            
        # try to look up names from known activation functions, if the passed parameters are not callables
        if not isinstance(self.activation, Callable):
            try:
                self.activation = activations[self.activation]
            except KeyError:
                raise ValueError(f"Unknown activation {self.activation}.")
        if not isinstance(self.activation_x, Callable):
            try:
                self.activation_x = activations_x[self.activation_x]
            except KeyError:
                raise ValueError(f"Unknown activation_x {self.activation_x}.")
        if not isinstance(self.activation_xx, Callable):
            try:
                self.activation_xx = activations_xx[self.activation_xx]
            except KeyError:
                raise ValueError(f"Unknown activation_xx {self.activation_xx}.")

        if not isinstance(self.activation_xxx, Callable):
            try:
                self.activation_xxx = activations_xxx[self.activation_xxx]
            except KeyError:
                raise ValueError(f"Unknown activation_xxx {self.activation_xxx}.")

        if not isinstance(self.activation_xxxx, Callable):
            try:
                self.activation_xxxx = activations_xxxx[self.activation_xxxx]
            except KeyError:
                raise ValueError(f"Unknown activation_xxxx {self.activation_xxxx}.")
            
        # internal model
        self._model = None

    
    def init_model(self, domain, boundary_condition = None, initial_condition = None, init_cond_intetior_only=True,
                   all_points=False):
        '''
        Build the model and initialize weights.
        '''
        layers = []
        if initial_condition is None:
            sample_uniformly = True # Sample unfiromly is used only when the param sampler uses the prob distr
        else:
            sample_uniformly = False
        layers.append((
                "basis",
                Dense(
                    layer_width=self.n_neurons,
                    activation=self.activation,
                    parameter_sampler=self.parameter_sampler,
                    random_seed=self.random_state,
                    sample_uniformly=sample_uniformly,
                ),
            )
        )

        self._model = Pipeline(steps=layers, verbose=False)
        
        # initialize all internals
        # careful to use ALL points here, otherwise the weights are not initialized over the entire domain.
        if initial_condition is None:
            self._model.fit(
                domain.interior_points, 
                np.zeros((domain.interior_points.shape[0], self.n_neurons))
            )
        else:
            if all_points:
                self._model.fit(
                    domain.all_points, 
                    initial_condition
                )
            else:
                self._model.fit(
                    domain.interior_points, 
                    initial_condition
                )

        


    def evaluate_model(self, x):
        '''
        Evaluate the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons)
        '''
        self._model.steps[0][1].activation = self.activation 
        return self._model.transform(x)


    def evaluate_model_gradient(self, x):
        '''
        Evaluate the gradient of the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons, d)
        '''

        self._model.steps[0][1].activation = self.activation_x
        return np.stack([
            self._model.transform(x)
                * self._model.steps[0][1].weights[d, :] for d in range(x.shape[1])
            ], axis = -1)


    def evaluate_model_laplace(self, x, parameter_scaling=None):
        '''
        Evaluate the laplace operator applied to the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons)
        '''
        self._model[0].activation = self.activation_xx

        if parameter_scaling is None:
            parameter_scaling = np.ones((self._model[0].weights.shape[0], ))
        parameter_scaling = parameter_scaling.reshape((-1, ))

        return self._model.transform(x) * (
            np.sum([parameter_scaling[k] * self._model[0].weights[k, :]**2 for k in range(parameter_scaling.shape[0])], axis=0, keepdims=False)
        )

    def evaluate_model_fourth_order_diff(self, x, parameter_scaling=None):
        '''
        Evaluate the fourth-order derivative operator applied to the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons)
        '''
        # Replace activation with the 4th derivative of the original activation
        self._model[0].activation = self.activation_xxxx  # Assumes this is defined

        if parameter_scaling is None:
            parameter_scaling = np.ones((self._model[0].weights.shape[0], ))
        parameter_scaling = parameter_scaling.reshape((-1, ))

        return self._model.transform(x) * (
            np.sum([
                parameter_scaling[k] * self._model[0].weights[k, :]**4
                for k in range(parameter_scaling.shape[0])
            ], axis=0, keepdims=False)
        )
    
    '''
    def evaluate_model_fourth_order_diff(self, x):
        self._model.steps[0][1].activation = self.activation_xxxx
        print(self._model.steps[0][1].transform(x))
        print(self._model.steps[0][1].weights)
        print(self._model.steps[1][1].weights)
        return (self._model.steps[0][1].transform(x) * np.linalg.norm(self._model.steps[0][1].weights, axis=0, keepdims=True)**4) @ self._model.steps[1][1].weights
    '''

    def fit_model(self, x, y):
        '''
        Fit the model to the data.

        Parameters:
        x: input values of shape (n_points, d)
        y: target values of shape (n_points,)
        '''
        self._model.steps[0][1].activation = self.activation
        self._model.fit(x, y)

        return self
    

    def fit_model_laplace(self, x, y):
        '''
        Fit the model with the laplace operator applied to it to the data.

        Parameters:
        x: input values of shape (n_points, d)
        y: target values of shape (n_points,)

        '''
        self._model.steps[0][1].activation = self.activation_xx
        self._model.fit(x, y)

        return self
    
    def fit_model_helmholtz(self, x, y):
        '''
        Fit the model with the laplace operator applied to it to the data.

        Parameters:
        x: input values of shape (n_points, d)
        y: target values of shape (n_points,)

        '''
        self._model.steps[0][1].activation = self.activation_xx
        self._model.fit(x, y)

        return self
