import numpy as np
from typing import List, Tuple

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
