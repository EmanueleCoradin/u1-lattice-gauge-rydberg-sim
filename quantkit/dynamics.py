import numpy as np
from typing import List, Tuple
from .information import EntanglementEntropy, LogschmidtEcho

def psi_t(
    t: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    C: np.ndarray,
    normalize: bool = False ) -> np.ndarray:
    """
    Computes the time-evolved quantum state |psi(t)> given the spectral decomposition.

    The evolution is given by:
        |psi(t)> = sum_j C_j * exp(-i * E_j * t) * |v_j>
    where E_j are eigenvalues and |v_j> are eigenvectors.

    Parameters
    ----------
    t : float
        Time at which the state is evaluated.
    eigenvalues : np.ndarray
        1D array of Hamiltonian eigenvalues.
    eigenvectors : np.ndarray
        2D array of Hamiltonian eigenvectors (columns are eigenvectors).
    C : np.ndarray
        Expansion coefficients of the initial state in the eigenbasis.
    normalize : bool, optional
        Whether to normalize the resulting state. Default is False.

    Returns
    -------
    np.ndarray
        The time-evolved state |psi(t)> as a 1D complex array.
    """
    # Compute phase factors
    phase_factors = np.exp(-1j * eigenvalues * t)
    
    # Multiply coefficients by phase factors
    evolved_components = phase_factors * C
    
    # Reconstruct the state in the original basis
    psi = eigenvectors @ evolved_components
    
    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(psi)
        if norm != 0:
            psi /= norm
    
    return psi

def compute_expectations(
    psi_t_list: List[np.ndarray],
    n_ops: List[np.ndarray],
    E_ops: List[np.ndarray] ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the time-dependent expectation values of n_i and E_i operators.

    Parameters
    ----------
    psi_t_list : list of np.ndarray
        List of states |psi(t)> at different times.
    n_ops : list of (sparse) matrices
        Local occupation operators for each site.
    E_ops : list of (sparse) matrices
        Local electric field operators for each site/link.

    Returns
    -------
    n_expect : np.ndarray
        Array of shape (len(psi_t_list), len(n_ops)) with ⟨n_i⟩ values.
    E_expect : np.ndarray
        Array of shape (len(psi_t_list), len(E_ops)) with ⟨E_i⟩ values.
    """
    T = len(psi_t_list)
    L_n = len(n_ops)
    L_E = len(E_ops)

    n_expect = np.zeros((T, L_n))
    E_expect = np.zeros((T, L_E))

    for ti, psi in enumerate(psi_t_list):
        for i in range(L_n):
            n_expect[ti, i] = np.vdot(psi, n_ops[i] @ psi).real
        for j in range(L_E):
            E_expect[ti, j] = np.vdot(psi, E_ops[j] @ psi).real

    return n_expect, E_expect

def compute_entropy(psi_t_list: List[np.ndarray], projector: np.ndarray = None, dA: int=None, dB: int=None, base: float = 2) -> np.ndarray:
    """
    Compute entanglement entropy over a list of state vectors.
    
    Parameters
    ----------
    psi_t_list : List[np.ndarray]
        List of state vectors (each of size dA*dB).
    dA, dB : int, optional
        Dimensions of subsystem A and B. If None, defaults to an equal bipartition.
    base : float
        Logarithm base (2 for bits, np.e for nats).
    
    Returns
    -------
    entropy : np.ndarray
        Array of entropies for each state.
    """
    # Dimension of Hilbert space
    Tdim = psi_t_list[0].shape[0]
    
    # Infer bipartition if not provided
    if dA is None or dB is None:
        # try equal bipartition
        dA = int(np.sqrt(Tdim))
        dB = Tdim // dA
        assert dA * dB == Tdim, "Cannot infer a square bipartition for this state size."
    
    if projector is None:
        entropy = np.array([EntanglementEntropy(psi, dA, dB, base) for psi in psi_t_list])
    else:
        entropy = np.array([EntanglementEntropy(projector@psi, dA, dB, base) for psi in psi_t_list])
    return entropy

def compute_echo(psi_t_list: List[np.ndarray], psi_0: np.ndarray) -> np.ndarray:
    """
    Compute the Loschmidt echo for a list of time-evolved states.

    The Loschmidt echo is defined as
        L(t) = |<psi(t) | psi(0)>|^2

    Parameters
    ----------
    psi_t_list : List[np.ndarray]
        List of state vectors at different times.
    psi_0 : np.ndarray
        Initial state vector.

    Returns
    -------
    echo : np.ndarray
        Array of Loschmidt echo values for each state in `psi_t_list`.
    """
    echo = np.array([LogschmidtEcho(psi, psi_0) for psi in psi_t_list])
    return echo