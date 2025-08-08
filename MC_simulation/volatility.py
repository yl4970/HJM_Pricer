import numpy as np
from typing import List, Any
from functools import cached_property
from collections import namedtuple
from dataclasses import dataclass

@dataclass
class FitResult:
    params:List[Any]
    fittedVols:List[Any]

class Volatility:
    """
    Computes principal component volatilities and polynomial volatility surface fits
    from a forward rate covariance matrix.
    """
    def __init__(self, 
                 tenors:np.array=np.linspace(1,360,360), 
                 covariance_matrix:np.ndarray=None, 
                 n_factors:int=3):
        
        self.tenors = tenors
        self.covariance_matrix = covariance_matrix
        self.n_factors = n_factors
        if self.covariance_matrix.shape[0] != self.covariance_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")

        if self.n_factors > self.covariance_matrix.shape[0]:
            raise ValueError(f"n_factors ({self.n_factors}) exceeds number of dimensions in covariance matrix")

        self.princ_eigval, self.princ_eigvec = self._compute_principal_components(self.covariance_matrix, self.n_factors)

    def __repr__(self):
        return f"{self.__class__.__name__}(tenors={len(self.tenors)}, principal_eigenvalues={self.princ_eigval}, principal_eigenvectors={self.princ_eigvec}, pca_n_components={self.n_factors}.)" 

    @cached_property
    def pca(self) -> np.ndarray:
        """
        Compute volatility surfaces by scaling principal components with the square roots of their associated eigenvalues.
        """
        sqrt_eigval = np.sqrt(self.princ_eigval)
        princ_vols =  self.princ_eigvec * sqrt_eigval[:, np.newaxis]
        return princ_vols

    def polyfit(self, 
                degrees:list[int]=None
                ) -> FitResult:
        """
        Fit polynomial functions to each principal volatility component of the forward curves using the provided tenors.
        """
        if len(degrees) != self.n_factors:
            raise ValueError(f"Expected {self.n_factors} degrees, got {len(degrees)}")
        
        if not np.issubdtype(np.array(self.tenors).dtype, np.number):
            raise TypeError("Tenors must be numeric")
        
        self.polyfittedVols = []
        self.polyparams = []
        for i in range(self.n_factors):
            vol = self.pca[i]
            params = np.polyfit(self.tenors, vol, degrees[i])
            self.polyparams.append(params)
            fitted_vols = np.polyval(params, self.tenors)
            self.polyfittedVols.append(fitted_vols)

        return FitResult(params=self.polyparams, fittedVols=self.polyfittedVols)
    
    @staticmethod
    def _compute_principal_components(
        covariance_matrix:np.array=None,
        n_factors:int=3
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute and return top n principal components of the given covariance matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        index_eigvec = list((reversed(eigenvalues.argsort())))[:n_factors]
        princ_eigval = np.array([eigenvalues[i] for i in index_eigvec])
        princ_eigvec = np.hstack([[eigenvectors[:, i] for i in index_eigvec]])
        return (princ_eigval, princ_eigvec)