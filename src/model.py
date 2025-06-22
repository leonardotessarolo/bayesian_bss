import numpy as np
from abc import ABC, abstractmethod

class InstantaneousMixtureModel:
    
    def __init__(
        self,
        mixing_matrix:np.array
    ):
        # Mixing matrix
        self.mixing_matrix=mixing_matrix

        # Number of sources and mixtures
        self.nsources=np.array(mixing_matrix).shape[-1]
        self.nmixtures=np.array(mixing_matrix).shape[0]

        # Verify determinant for invertibility of mixing matrix
        if np.linalg.det(self.mixing_matrix)==0:
            raise ValueError('Mixing matrix given is non-invertible.')