import numpy as np
from scipy.linalg import logm

def Purity(rho: np.ndarray) -> float:
    """
    Computes the purity of a quantum state, defined as the trace of the square of 
    the density matrix rho: Tr(rho^2).

    Parameters
    ----------
    rho : np.ndarray
        The density matrix of the quantum system. Must be a square matrix.

    Returns
    -------
    float
        The purity of the quantum state. Ranges from 1/d to 1, where d is the 
        dimension of the Hilbert space. Pure states have purity 1.
    
    Raises
    ------
    TypeError
        If rho is not a numpy ndarray.
    ValueError
        If rho is not a square matrix.
    """

    if not isinstance(rho, np.ndarray):
        raise TypeError(f'rho should be a numpy ndarray, not a {type(rho)}')
    
    # Ensure the density matrix is square
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrix must be square.")
    
    # Compute rho^2
    rho_squared = np.dot(rho, rho)
    
    # Compute the trace of rho^2
    trace_rho2 = np.trace(rho_squared)
    
    return trace_rho2

def VonNeumannEntropy(rho: np.ndarray) -> float:
    """
    Computes the von Neumann entropy of a quantum state:
    S(rho) = -Tr(rho * log2(rho)), where log2 is the base-2 logarithm.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix of the quantum system. Must be a square matrix.

    Returns
    -------
    float
        The von Neumann entropy of the quantum state in bits.
    
    Raises
    ------
    TypeError
        If rho is not a numpy ndarray.
    ValueError
        If rho is not a square matrix.
    """
    if not isinstance(rho, np.ndarray):
        raise TypeError(f'rho should be a numpy ndarray, not a {type(rho)}')
    
    # Ensure the density matrix is square
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrix must be square.")
    
    rho_log = logm(rho)
    
    # Convert the logarithm to base 2
    rho_log_base2 = rho_log / np.log(2)
    
    # Compute rho * log2(rho)
    rho_log2_rho = np.dot(rho, rho_log_base2)
    
    # Compute the trace of rho * log2(rho)
    trace_rho_log2_rho = -np.trace(rho_log2_rho)
    
    return trace_rho_log2_rho
