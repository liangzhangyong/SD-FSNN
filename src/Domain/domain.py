from dataclasses import dataclass
import numpy as np

@dataclass
class Domain:
    '''
    Specifies a domain as a collection of points.

    Attributes:
    -----------
    interior_points: np.ndarray
        points in the interior of the domain, shape (n_interior_points, d)
    boundary_points: np.ndarray
        points on the boundary of the domain, shape (n_boundary_points, d)
    normal_vectors: np.ndarray
        normal vectors at the boundary points of the domain, shape (n_boundary_points, d)
    all_points: np.ndarray
        interior and boundary points combined (automatically deduced)
    n_dim: int
        dimension d of the domain (automatically deduced)
    '''

    interior_points: np.ndarray = None
    boundary_points: np.ndarray = None
    normal_vectors: np.ndarray = None
    sample_points: np.ndarray = None
    all_points = None
    n_dim: int = None
    _hi: float = None  # Upper bound for Frozen PINN SWIM sampling (e.g., L for [-L, L]^D)

    def __post_init__(self):
        self.n_dim = self.interior_points.shape[1]

        if self.interior_points is not None and self.boundary_points is not None:
            self.all_points = np.row_stack([self.boundary_points, self.interior_points])
    

    def set_all_points(self):
        if self.interior_points is not None and self.boundary_points is not None:
            self.all_points = np.row_stack([self.boundary_points, self.interior_points])
    