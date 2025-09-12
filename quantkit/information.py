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

def EntanglementEntropy(psi: np.ndarray, dA: int, dB: int, base: float = 2) -> float:
    """Compute von Neumann entanglement entropy of a pure state.
    
    Parameters
    ----------
    psi : np.ndarray
        State vector of shape (dA*dB,) or (dA,dB).
    dA : int
        Dimension of subsystem A.
    dB : int
        Dimension of subsystem B.
    base : float
        Logarithm base (2 for bits, np.e for nats).
    """
    psi = psi.flatten()
    assert dA*dB == psi.shape[0], "Incorrect bipartition: dA*dB!=psi.shape[0]"

    M = psi.reshape((dA, dB))       
    s= np.linalg.svd(M, compute_uv=False)
    lambdas = (s**2)                 # eigenvalues of rho_A
    # cut small lambdas to reduce numerical issues
    lambdas = lambdas[lambdas>1e-15]
    # computing the Von Neumann entropy
    S = -np.sum(lambdas * np.log(lambdas)) / np.log(base)

    return S

def LogschmidtEcho(psi: np.ndarray, psi_0: np.ndarray) -> float:
    """Compute the Loschmidt echo |<psi|psi0>|^2."""
    overlap = np.vdot(psi, psi_0)
    echo = np.abs(overlap)**2
    return float(echo.real) 