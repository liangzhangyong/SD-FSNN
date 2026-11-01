from abc import ABC, abstractmethod

class Ansatz(ABC):
    '''
    Base class for ansatz functions that can be used to construct PDE solutions by the solvers.

    Notes
    -----
    Not all solvers can be combined with all ansatz types as they may assume special properties.
    '''

    @abstractmethod
    def init_model(self, domain, boundary_condition = None, initial_condition = None):
        '''Assemble the model and initialize internal weights.'''
        pass

    @abstractmethod
    def evaluate_model(self, x):
        '''Evaluate the model.'''
        pass

    @abstractmethod
    def evaluate_model_gradient(self, x):
        '''Evaluate the gradient of the model.'''
        pass

    @abstractmethod
    def evaluate_model_laplace(self, x):
        '''Evaluate the laplace operator applied to the model.'''
        pass

    @abstractmethod
    def evaluate_model_fourth_order_diff(self, x):
        '''Evaluate the fourth order differential operator applied to the model.'''
        pass

    @abstractmethod
    def fit_model(self, x, y):
        '''Fit the model to the data.'''
        pass
    
    @abstractmethod
    def fit_model_laplace(self, x, y):
        '''Fit the model with the laplace operator applied to it to the data.'''
        pass
    
