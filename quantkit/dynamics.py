import numpy as np
from typing import List, Tuple, Optional
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
    E_ops: List[np.ndarray],
    rho_ops: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute the time-dependent expectation values of n_i and E_i, and optionally rho_i.

    Parameters
    ----------
    psi_t_list : list of np.ndarray
        List of states |psi(t)> at different times.
    n_ops : list of (sparse) matrices
        Local occupation operators for each site.
    E_ops : list of (sparse) matrices
        Local electric field operators for each site/link.
    rho_ops: list of (sparse) matrices
        Local density of particles and antiparticles

    Returns
    -------
    n_expect   : np.ndarray
        Array of shape (len(psi_t_list), len(n_ops)) with ⟨n_i⟩ values.
    E_expect   : np.ndarray
        Array of shape (len(psi_t_list), len(E_ops)) with ⟨E_i⟩ values.
    rho_expect : np.ndarray
        Array of shape (len(psi_t_list), len(r_ops)) with (rho_i⟩ values.
    """
    T = len(psi_t_list)
    L_n = len(n_ops)
    L_E = len(E_ops)

    n_expect = np.zeros((T, L_n))
    E_expect = np.zeros((T, L_E))
    
    rho_expect = None
    if rho_ops is not None:
        L_rho = len(rho_ops)
        rho_expect = np.zeros((T, L_rho))

    for ti, psi in enumerate(psi_t_list):
        for i in range(L_n):
            n_expect[ti, i] = np.vdot(psi, n_ops[i] @ psi).real
        for j in range(L_E):
            E_expect[ti, j] = np.vdot(psi, E_ops[j] @ psi).real
        if rho_ops is not None:
            for r in range(L_n):
                E_left = E_expect[ti, r-1] if r > 0  else E_expect[ti, -1]
                E_right = E_expect[ti, r] if r < L_E else E_expect[ti, 0]
                rho_expect[ti, r] = np.abs(E_right - E_left)



    return n_expect, E_expect, rho_expect

def compute_model_expectations(
    models: List[str],
    psi_t_lists: List[List[np.ndarray]],
    n_ops_lists: List[List[np.ndarray]],
    E_ops_lists: List[List[np.ndarray]],
    rho_ops_lists: Optional[List[List[np.ndarray]]] = None,
) -> dict:
    """
    Compute expectation values for multiple models.

    Returns
    -------
    results : dict
        {
          "ModelName": {
             "n": np.ndarray (T × L_n),
             "E": np.ndarray (T × L_E),
             "rho": np.ndarray (T × L_rho or None)
          },
          ...
        }
    """
    results = {}
    for i, model in enumerate(models):
        rho_ops = rho_ops_lists[i] if rho_ops_lists is not None else None
        n_exp, E_exp, rho_exp = compute_expectations(
            psi_t_lists[i], n_ops_lists[i], E_ops_lists[i], rho_ops
        )
        results[model] = {"n": n_exp, "E": E_exp, "rho": rho_exp}
    return results


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

def compute_fft(
    times: np.ndarray,
    data: np.ndarray,
    site_index: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fourier transform of an observable (per-site or averaged).

    Parameters
    ----------
    times : np.ndarray
        Array of time points, shape (T,).
    data : np.ndarray
        Observable values, shape (T,) if averaged, or (T, L) if site-resolved.
    site_index : int or None
        If None: data is assumed already site-averaged (1D).
        If int: analyze that specific site from (T, L) data.

    Returns
    -------
    freqs : np.ndarray
        Array of positive frequency values.
    spectrum : np.ndarray
        Power spectrum |FFT|^2 corresponding to freqs.
    """
    dt = times[1] - times[0]   # assume uniform spacing
    T = len(times)

    # Select signal
    if data.ndim == 1:
        signal = data
    elif data.ndim == 2:
        if site_index is None:
            # average over sites
            signal = data.mean(axis=1)
        else:
            signal = data[:, site_index]
    else:
        raise ValueError("data must be 1D (averaged) or 2D (time × sites)")

    signal = signal - np.mean(signal)
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(T, d=dt)

    # Keep positive frequencies 
    # Observables are hermitian operators -> we expect FFT with even symmetry
    pos_mask = fft_freqs >= 0
    freqs = fft_freqs[pos_mask]
    spectrum = np.abs(fft_vals[pos_mask])**2 / T

    # One-sided normalization: double except DC and Nyquist (if present)
    if T % 2 == 0:
        spectrum[1:-1] *= 2
    else:
        spectrum[1:] *= 2

    return freqs, spectrum