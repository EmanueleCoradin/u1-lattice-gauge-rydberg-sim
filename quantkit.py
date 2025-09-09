import numpy as np
import pandas as pd
from itertools import product

from scipy.sparse import dok_matrix, csr_matrix, identity, kron, vstack
from scipy.sparse.linalg import eigsh, expm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

plt.style.use('tableau-colorblind10')
# Define the Pauli matrices as sparse matrices
sigma = [
    csr_matrix([[0, 1], [1, 0]]),  # sigma_x
    csr_matrix([[0, -1j], [1j, 0]]),  # sigma_y
    csr_matrix([[1, 0], [0, -1]])  # sigma_z
]

sigma_plus  = sigma[0] + 1j * sigma[1]
sigma_minus = sigma[0] - 1j * sigma[1]

ID_matrix = identity(2, format="csr")

n_projector = (sigma[2] + ID_matrix) * 0.5

def compute_reduced_density_matrix(state, local_dims, num_sites, keep_indices, print_rho=False):
    """
    Computes the reduced density matrix for a subsystem of a quantum system.

    Parameters
    ----------
    state : ndarray
        Quantum state of the entire system (vectorized form).
    local_dims : ndarray or list
        Local dimension of each site in the quantum system.
    num_sites : int
        Total number of sites in the system.
    keep_indices : list of ints
        Indices of the sites to keep in the subsystem.
    print_rho : bool, optional
        If True, prints the reduced density matrix, by default False.

    Returns
    -------
    ndarray
        Reduced density matrix of the subsystem.
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

def add_subsystem(state_a, state_b):
    """
    Computes the Kronecker product of two quantum states.

    Parameters
    ----------
    state_a : ndarray
        Quantum state of subsystem A.
    state_b : ndarray
        Quantum state of subsystem B.

    Returns
    -------
    ndarray
        The Kronecker product of state_a and state_b.
    """
    return np.kron(state_a, state_b)

def generate_random_state(dim):
    """
    Generates a random quantum state of a given dimension.

    Parameters
    ----------
    dim : int
        The dimension of the quantum state.

    Returns
    -------
    ndarray
        A normalized random quantum state of the given dimension.
    """
    random_state = (np.random.rand(dim) * 2 - 1) + (np.random.rand(dim) * 2 - 1) * 1.j
    random_state = random_state / np.linalg.norm(random_state)
    return random_state

def generate_random_separable_state(num_subsystems, dims):
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

def generate_random_non_separable_state(num_subsystems, dims):
    """
    Generates a random entangled (non-separable) quantum state for a multi-partite system.

    Parameters
    ----------
    num_subsystems : int
        The number of subsystems in the system.
    dims : tuple or ndarray
        The dimensions of the subsystems (either a tuple of dimensions or a numpy array).

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

def compute_mixed_density_matrix(states, probabilities):
    """
    Computes the density matrix for a mixture of quantum states with associated probabilities.

    Parameters
    ----------
    states : list of ndarray
        The quantum states in the mixture.
    probabilities : list of float
        The probabilities corresponding to each state.

    Returns
    -------
    ndarray
        The density matrix of the mixed state.
    """
    dim = states[0].ravel().shape[0]
    density_matrix = np.zeros((dim, dim), dtype=complex)
    
    for state, prob in zip(states, probabilities):
        density_matrix += prob * np.outer(state, state.conj())
    
    return density_matrix

def compute_reduced_density_matrix_for_mixture(states, probabilities, local_dims, num_sites, keep_indices, print_rho=False):
    """
    Computes the reduced density matrix for a mixture of quantum states.

    Parameters
    ----------
    states : list of ndarray
        The quantum states in the mixture.
    probabilities : list of float
        The probabilities associated with each state.
    local_dims : ndarray or list
        Local dimension of each site in the quantum system.
    num_sites : int
        Total number of sites in the quantum system.
    keep_indices : list of ints
        Indices of the sites to keep in the subsystem.
    print_rho : bool, optional
        If True, prints the reduced density matrix, by default False.

    Returns
    -------
    ndarray
        The reduced density matrix of the mixed subsystem.
    """
    reduced_density_matrix = probabilities[0] * compute_reduced_density_matrix(states[0], local_dims, num_sites, keep_indices, print_rho=False)
    
    for state, prob in zip(states[1:], probabilities[1:]):
        reduced_density_matrix += prob * compute_reduced_density_matrix(state, local_dims, num_sites, keep_indices, print_rho=False)
    
    return reduced_density_matrix

def Purity(rho):
    """
    Computes the trace of the square of a density matrix.

    Parameters
    ----------
    rho : ndarray
        The density matrix of the quantum system.

    Returns
    -------
    float
        The trace of rho^2.
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

def VonNeumannEntropy(rho):
    """
    Computes the trace of the product of a density matrix and its logarithm:
    Tr(rho * log(rho)).

    Parameters
    ----------
    rho : ndarray
        The density matrix of the quantum system.

    Returns
    -------
    float
        The trace of rho * log(rho).
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

def Plot_Matrix(density_matrix):
    # Plotting
    plt.figure(figsize=(7, 5))

    # Plot the real part of the matrix
    plt.imshow(np.real(density_matrix), cmap='coolwarm', interpolation='nearest', 
               vmin=np.min(np.real(density_matrix)), vmax=np.max(np.real(density_matrix)))

    # Add color bar to show the real part mapping
    cbar = plt.colorbar()
    cbar.set_label('Real Part of Matrix Elements')

    # Annotate each cell with the complex number (real + imaginary) up to 2 decimal places
    for i in range(density_matrix.shape[0]):
        for j in range(density_matrix.shape[1]):
            complex_value = density_matrix[i, j]
            real_part = np.real(complex_value)
            imag_part = np.imag(complex_value)
            # Display complex number with two decimal places for real and imaginary parts
            plt.text(j, i, f'{real_part:.2f} + {imag_part:.2f}j', ha='center', va='center', color='black', fontsize=9)

    # Title and labels
    plt.title("Mixed State density matrix")
    plt.xlabel('Index')
    plt.ylabel('Index')

    # Show the indices on the x and y axes
    plt.xticks(np.arange(density_matrix.shape[1]))  # Set x-axis ticks to be the column indices
    plt.yticks(np.arange(density_matrix.shape[0]))  # Set y-axis ticks to be the row indices

    # Tight layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def generate_blockade_basis_sparse(L, return_vectors=False, open_boundaries=False):
    """
    Generate blockade-constrained basis as indices (and optionally sparse vectors),
    supporting open or periodic boundary conditions.

    Parameters:
    -----------
    L : int
        Length of the chain (number of qubits).
    return_vectors : bool
        If True, return sparse vectors. Otherwise, return only indices.
    open_boundaries : bool
        If True, use open boundary conditions; else, periodic boundaries.

    Returns:
    --------
    valid_indices : list of int
        Indices of valid basis states in the full 2^L-dimensional space.
    sparse_vectors : list of csr_matrix (optional)
        Sparse vectors of valid basis states (if return_vectors is True).
    """

    def is_valid(bits):
        for i in range(L - 1):
            if bits[i] + bits[i + 1] == 0:
                return False
        if not open_boundaries and bits[0] + bits[-1] == 0:
            return False
        return True

    valid_indices = []
    sparse_vectors = []

    for bits in product([0, 1], repeat=L):
        if is_valid(bits):
            index = int("".join(map(str, bits)), 2)
            valid_indices.append(index)

            if return_vectors:
                vec = dok_matrix((1, 2**L), dtype=np.float64)  # Row vector
                vec[0, index] = 1.0
                sparse_vectors.append(vec.tocsr())

    if return_vectors:
        return valid_indices, sparse_vectors
    else:
        return valid_indices

def blockade_projection_matrix(L, open_boundaries=False):
    """
    Construct the projection matrix P from the full Hilbert space to
    the blockade-constrained subspace using sparse basis vectors.

    Parameters:
    -----------
    L : int
        Length of the chain.
    open_boundaries : bool
        Whether to use open or periodic boundary conditions.

    Returns:
    --------
    P : csr_matrix
        Sparse matrix (d × 2^L) where each row is a blockade-allowed basis vector.
    """
    _, basis_vectors = generate_blockade_basis_sparse(L, return_vectors=True, open_boundaries=open_boundaries)
    return vstack(basis_vectors, format='csr')

    
def Compute_H_FSS_sparse(Omega, delta, L):
    """
    Construct the full sparse Hamiltonian for a 1D Rydberg chain of length L
    with Rabi Frequency Omega and detuning delta.
    """
    # Precompute identity matrices
    identities = [identity(2**i, format="csr") for i in range(L)]

    # Initialize the Hamiltonian
    H = csr_matrix((2**L, 2**L), dtype=np.float32)

    for i in range(L):
        left = identities[i] if i > 0 else identity(1, format='csr')
        right = identities[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        H += Omega * kron(kron(left, sigma[0]), right)
        H += kron(kron(left, 2.0 * delta * n_projector), right)

    return H

def Compute_H_schwinger_sparse(w, m, J, alpha, L):
    assert J > 0, "Gauge coupling J must be positive."
    H = csr_matrix((2**L, 2**L), dtype=np.float64)
    identities = [identity(2**i, format="csr") for i in range(L)]

    # mass term
    for i in range(L):
        left  = identities[i] if i > 0 else identity(1, format='csr')
        right = identities[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        H += (m/2) * ((-1)**i) * kron(kron(left, sigma[2]), right)

    # kinetic (your convention uses -w in front of σ⁺σ⁻ + h.c.; keep it)
    for i in range(L-1):
        left  = identities[i] if i > 0 else identity(1, format='csr')
        right = identities[L - i - 2] if (L - i - 2) > 0 else identity(1, format='csr')
        H += -w * kron(kron(kron(left, sigma_plus),  sigma_minus), right)
        H += -w * kron(kron(kron(left, sigma_minus), sigma_plus),  right)

    # gauge energy from the corrected E_j
    E_ops = E_operators_by_site_schwinger(L, alpha)
    for E in E_ops:
        # E is diagonal -> E @ E is cheap, and this is exact
        H += J * (E @ E)

    return H
def n_operator(L):
    """
    Function to compute the n-operator based on tensor products of identity matrices and the number projector.
    
    Parameters:
        L (int): The number of qubits or degrees of freedom.
                
    Returns:
        n_operator (csr_matrix): The resulting sparse matrix for the n-operator.
    """
    # Precompute identity matrices for tensor product construction
    identities = [identity(2**i, format="csr") for i in range(L)]
    
    # Initialize the n_operator as a sparse zero matrix of appropriate size
    n_operator = csr_matrix((2**L, 2**L), dtype=np.float32)
    
    for i in range(L):
        # Construct the left and right parts of the tensor product
        left = identities[i] if i > 0 else identity(1, format='csr')
        right = identities[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        
        # Update the n_operator with the tensor product
        n_operator += kron(kron(left, n_projector), right)
    
    return n_operator

def n_operators_by_site(L):
    """
    Returns a list of sparse matrices [n_0, n_1, ..., n_{L-1}],
    where each n_i acts as the number operator on site i.
    """
    identities = [identity(2**i, format="csr") for i in range(L)]
    n_ops = []

    for i in range(L):
        left = identities[i] if i > 0 else identity(1, format='csr')
        right = identities[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        n_i = kron(kron(left, n_projector), right)
        n_ops.append(n_i)

    return n_ops

def E_operators_by_site(L):
    """
    Returns a list of sparse matrices [E_0, E_1, ..., E_{L-1}],
    where each E_i represents the electric field associated with site i.
    """
    identities = [identity(2**i, format="csr") for i in range(L)]
    E_ops = []

    for i in range(L):
        left = identities[i] if i > 0 else identity(1, format='csr')
        right = identities[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        E_i = kron(kron(left, (-1)**(i) * 0.5 * sigma[2]), right)
        E_ops.append(E_i)

    return E_ops

def E_operators_by_site_schwinger(L, alpha):
    E_ops = []
    I_full = identity(2**L, format='csr')
    for j in range(L - 1):
        E_j = alpha * I_full  # +α (not −α)
        for l in range(j + 1):
            left  = identity(2**l,          format='csr') if l > 0 else identity(1, format='csr')
            right = identity(2**(L - l - 1), format='csr') if (L - l - 1) > 0 else identity(1, format='csr')
            # 0.5 * σ^z_l
            E_j += 0.5 * kron(kron(left, sigma[2]), right)
            # 0.5 * (-1)^l * I
            E_j += 0.5 * ((-1)**l) * I_full
        E_ops.append(E_j)
    return E_ops


def Compute_First_k_levels(k, Omega, delta, L, return_eigenvectors=False):
    return eigsh(Compute_H_FSS_sparse(Omega, delta, L), k=k, which='SA', return_eigenvectors=return_eigenvectors, tol=1e-5)

def psi_t(t, eigenvalues, eigenvectors, C, normalize=False):
    phase_factors = np.exp(-1j * eigenvalues * t)
    evolved_components = phase_factors * C
    psi = eigenvectors @ evolved_components
    if normalize:
        norm = np.linalg.norm(psi)
        if norm != 0:
            psi /= norm
    return psi
