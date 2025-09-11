import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import logm
from typing import Union, Sequence

# -------------------------------
# Basic State Utilities
# -------------------------------

def add_subsystem(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Computes the Kronecker product of two quantum states."""
    return np.kron(a, b)

# -------------------------------
# Random State Generation
# -------------------------------

def generate_random_state(dim: int) -> np.ndarray:
    """Generates a normalized random quantum state of a given dimension."""
    random_state = (np.random.rand(dim) * 2 - 1) + (np.random.rand(dim) * 2 - 1) * 1.j
    random_state = random_state / np.linalg.norm(random_state)
    return random_state

def generate_random_separable_state(num_subsystems: int, dims: Union[int, Sequence[int]]) -> np.ndarray:
    """
    Generates a random separable quantum state for a multi-partite system.

    Parameters
    ----------
    num_subsystems : int
        The number of subsystems in the system.
    dims : ndarray or int
        The dimensions of the subsystems (either an array of dimensions or a single dimension for all subsystems).

    Returns
    -------
    ndarray
        The random separable quantum state for the multi-partite system.
    """
    dims = np.asarray(dims)
    
    # If dims is a single integer, convert it into a list of the same dimension for all subsystems
    if dims.ndim == 0:
        dims = np.full(num_subsystems, dims)

    # Initialize state as the "vacuum" state
    multi_partite_state = np.array([1])

    # Generate random states for each subsystem and take the Kronecker product
    for dim in dims:
        multi_partite_state = add_subsystem(multi_partite_state, generate_random_state(dim))

    return multi_partite_state

def generate_random_non_separable_state(num_subsystems: int, dims: Union[int, Sequence[int]]) -> np.ndarray:
    """
    Generates a random entangled (non-separable) quantum state for a multi-partite system.

    Parameters
    ----------
    num_subsystems : int
        The number of subsystems in the system.
    dims : int or sequence of int
        The dimensions of the subsystems. Can be:
        - a single integer (all subsystems have the same dimension), or
        - a sequence of integers specifying the dimension of each subsystem.


    Returns
    -------
    ndarray
        The random entangled quantum state for the multi-partite system.
    """
    # Ensure dims is an array (for consistency)
    dims = np.asarray(dims)
    
    # If dims is a single integer, assume all subsystems have the same dimension
    if dims.ndim == 0:
        dims = np.full(num_subsystems, dims)
    
    # Compute the total dimension of the Hilbert space
    total_dim = np.prod(dims)

    # Start with a product state of |0> for each subsystem
    state = np.zeros(total_dim, dtype=complex)
    state[0] = 1.0  # Initial state is the computational basis |0...0>

    # Generate a random unitary matrix of dimension total_dim
    random_unitary = unitary_group.rvs(total_dim)

    # Apply the random unitary to the product state
    state = random_unitary @ state

    # Normalize the state (this ensures it's a pure state)
    state /= np.linalg.norm(state)

    return state

# -------------------------------
# Definite State Generation
# -------------------------------

def generate_initial_state(L: int, kind: str = 'vacuum') -> np.ndarray:
    """
    Generate initial states for a Schwinger / Rydberg lattice model.
    
    Args:
        L   : system size (number of sites)
        kind: one of ['vacuum', 'single-q', 'single-q-bar', 
                      'q-q-bar-pair', 'q-q-bar-scattering', 'q-q-scattering']
    """
    zero = np.array([1., 0.]) 
    one  = np.array([0., 1.])  
    next_state = (zero, one)

    # define middle position
    mid = (L // 4)*2  

    # background = staggered vacuum pattern
    def background(i):
        return next_state[i % 2]

    # choose excitation sites for each scenario
    excitation_sites = set()

    if kind == 'vacuum':
        pass  # no excitations, just vacuum staggering

    elif kind == 'single-q':
        excitation_sites.add(mid - 1)

    elif kind == 'single-q-bar':
        excitation_sites.add(mid)

    elif kind == 'q-q-bar-pair':
        excitation_sites.update([mid - 1, mid])   # pair next to each other

    elif kind == 'q-q-bar-scattering':
        # example: q at mid-2, qbar at mid+2
        excitation_sites.update([mid - 2, mid + 2])

    elif kind == 'q-q-scattering':
        # two same-charge excitations at symmetric positions
        excitation_sites.update([mid - 2, mid + 2])

    else:
        raise ValueError(f"Unknown initial state kind: {kind}")

    # build full tensor product state
    state = np.array([1.])  # start as scalar
    for i in range(L):
        if i in excitation_sites:
            site = one
        else:
            site = background(i)
        state = add_subsystem(state, site)

    return state

# -------------------------------
# Density Matrix Construction
# -------------------------------

def compute_reduced_density_matrix(
    state: np.ndarray,
    local_dims: list[int] | np.ndarray,
    num_sites: int,
    keep_indices: list[int],
    print_rho: bool = False
) -> np.ndarray:
    """
    Compute the reduced density matrix for a subsystem of a pure quantum state.

    This function reshapes the input state vector into a tensor with one leg per site,
    groups the subsystem and environment indices, and performs a partial trace over
    the environment degrees of freedom.

    Parameters
    ----------
    state : np.ndarray
        State vector of the entire system (1D array of complex amplitudes).
    local_dims : array_like of int
        Local dimensions of each site in the system (e.g., [2, 2, 2] for qubits).
    num_sites : int
        Total number of sites in the system.
    keep_indices : list of int
        Indices of the sites to keep in the reduced subsystem.
    print_rho : bool, optional
        If True, print the reduced density matrix (default is False).

    Returns
    -------
    np.ndarray
        The reduced density matrix of the specified subsystem.
    """
    if not isinstance(state, np.ndarray):
        raise TypeError(f'State should be a numpy ndarray, not a {type(state)}')

    if not isinstance(local_dims, (np.ndarray, list)):
        raise TypeError(f'local_dims should be a numpy array or list, not a {type(local_dims)}')

    if not isinstance(num_sites, int):
        raise TypeError(f'num_sites should be an integer, not a {type(num_sites)}')

    # Reshape state into a tensor with one leg per lattice site
    state_tensor = state.reshape(*[local_dims[site] for site in range(num_sites)])

    # Determine the environment and subsystem indices
    all_indices = list(range(num_sites))
    env_indices = [i for i in all_indices if i not in keep_indices]
    new_order = keep_indices + env_indices

    # Rearrange the tensor to group subsystem and environment indices
    state_tensor = np.transpose(state_tensor, axes=new_order)

    # Dimensions of subsystem and environment
    subsystem_dim = int(np.prod([local_dims[i] for i in keep_indices]))
    env_dim = int(np.prod([local_dims[i] for i in env_indices]))

    # Reshape the tensor to separate subsystem from environment
    partitioned_state = state_tensor.reshape((subsystem_dim, env_dim))

    # Compute reduced density matrix by tracing out the environment
    reduced_density_matrix = np.tensordot(partitioned_state, np.conjugate(partitioned_state), axes=([1], [1]))
    reduced_density_matrix = reduced_density_matrix.reshape((subsystem_dim, subsystem_dim))

    # Print the reduced density matrix if requested
    if print_rho:
        print(f"Reduced Density Matrix for subsystem {keep_indices}:")
        print(reduced_density_matrix)

    return reduced_density_matrix

def compute_mixed_density_matrix(
    states: list[np.ndarray],
    probabilities: list[float]
) -> np.ndarray:
    """
    Compute the density matrix for a probabilistic mixture of pure states.

    Given a list of pure states |ψᵢ⟩ and associated probabilities pᵢ,
    this function returns the mixed density matrix

        ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| .

    Parameters
    ----------
    states : list of np.ndarray
        List of state vectors (1D complex arrays) representing pure states.
    probabilities : list of float
        Probabilities associated with each state. Should sum to 1.

    Returns
    -------
    np.ndarray
        Density matrix of the mixed state (2D complex array of shape (d, d)),
        where d is the Hilbert space dimension.

    Raises
    ------
    ValueError
        If the lengths of `states` and `probabilities` do not match.
    """
    dim = states[0].ravel().shape[0]
    rho = np.zeros((dim, dim), dtype=complex)
    for s, p in zip(states, probabilities):
        rho += p * np.outer(s, s.conj())
    return rho

def compute_reduced_density_matrix_for_mixture(
    states: list[np.ndarray],
    probabilities: list[float],
    local_dims: list[int] | np.ndarray,
    num_sites: int,
    keep_indices: list[int],
    print_rho: bool = False
) -> np.ndarray:
    """
    Compute the reduced density matrix for a subsystem of a mixture of states.

    This function performs a partial trace over the environment for each pure state
    in the mixture, then combines the results weighted by their associated probabilities.

    Parameters
    ----------
    states : list of np.ndarray
        List of pure state vectors representing the states in the mixture.
    probabilities : list of float
        Probabilities associated with each state. Should sum to 1.
    local_dims : array_like of int
        Local dimensions of each site in the system.
    num_sites : int
        Total number of sites in the system.
    keep_indices : list of int
        Indices of the sites to keep in the reduced subsystem.
    print_rho : bool, optional
        If True, print the reduced density matrix (default is False).

    Returns
    -------
    np.ndarray
        The reduced density matrix of the specified subsystem for the mixture.
    """
    rho = probabilities[0] * compute_reduced_density_matrix(states[0], local_dims, num_sites, keep_indices)
    for s, p in zip(states[1:], probabilities[1:]):
        rho += p * compute_reduced_density_matrix(s, local_dims, num_sites, keep_indices)
    if print_rho:
        print(rho)
    return rho


    """Generate initial product state (string or custom pattern)."""
    state = np.array([1])
    basis = (np.array([1.,0.]), np.array([0.,1.]))
    for i in range(L):
        if not string and i == 4:
            state = add_subsystem(state, basis[1])
        else:
            state = add_subsystem(state, basis[i % 2])
    return state