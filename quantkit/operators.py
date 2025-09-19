import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from typing import Optional, List

# -------------------------------
# Single Site Operators
# -------------------------------

sigma = [
    csr_matrix([[0, 1], [1, 0]]),  # sigma_x
    csr_matrix([[0, -1j], [1j, 0]]),  # sigma_y
    csr_matrix([[1, 0], [0, -1]])  # sigma_z
]

sigma_plus  = (sigma[0] + 1j * sigma[1])/2
sigma_minus = (sigma[0] - 1j * sigma[1])/2

I2 = identity(2, format="csr")

n_projector = (sigma[2] + I2) * 0.5
d_projector = -(sigma[2] - I2) * 0.5
# -------------------------------
# Templates
# -------------------------------

def one_site_operator(op, site, L):
    """Many-body operator with `op` at `site` (0-based), identity elsewhere.
       Left-to-right ordering: site 0 is leftmost factor."""
    res = None
    for i in range(L):
        local = op if i == site else I2
        res = local if res is None else kron(res, local, format='csr')
    return res

def two_site_operator(
    site_i: int,
    site_j: int,
    O_i: Optional[csr_matrix] = None,
    O_j: Optional[csr_matrix] = None,
    L: Optional[int] = None,
    precomputed_operators: Optional[List[csr_matrix]] = None ) -> csr_matrix:
    """
    Construct a two-site operator O_i O_j acting on a chain of length L.

    Either `precomputed_operators` is provided, or `O_i`, `O_j`, and `L` must be given.

    Parameters
    ----------
    site_i : int
        Index of the first site (must satisfy site_i < site_j).
    site_j : int
        Index of the second site.
    O_i : csr_matrix, optional
        Operator to act on site_i (required if precomputed_operators is None).
    O_j : csr_matrix, optional
        Operator to act on site_j (required if precomputed_operators is None).
    L : int, optional
        Total number of sites (required if precomputed_operators is None).
    precomputed_operators : list of csr_matrix, optional
        Precomputed single-site operators for all sites.

    Returns
    -------
    csr_matrix
        Sparse matrix representing the two-site operator O_i O_j on the full Hilbert space.

    Raises
    ------
    AssertionError
        If site_i >= site_j or if required arguments are missing.
    """
    assert site_i < site_j, "site_i must be less than site_j"

    if precomputed_operators is not None:
        return precomputed_operators[site_i] @ precomputed_operators[site_j]
    else:
        # On-the-fly construction requires O_i, O_j, and L
        if O_i is None or O_j is None or L is None:
            raise ValueError("O_i, O_j, and L must be provided if precomputed_operators is None.")
        left = one_site_operator(O_i, site_i, L)
        right = one_site_operator(O_j, site_j, L)
        return left @ right
# -------------------------------
# Composite Operators
# -------------------------------

def n_operator(L: int) -> csr_matrix:
    """
    Constructs the total number operator for L qubits/sites using tensor products.

    Parameters
    ----------
    L : int
        Number of qubits or sites.

    Returns
    -------
    csr_matrix
        Sparse matrix representing the total number operator.
    """
    dim = 2**L
    n_operator = csr_matrix((dim, dim), dtype=np.float32)
    
    for i in range(L):
        n_operator += one_site_operator(n_projector, i, L)
    
    return n_operator

def n_operators_by_site(L: int) -> list[csr_matrix]:
    """
    Constructs individual number operators n_i for each site i.

    Parameters
    ----------
    L : int
        Number of qubits or sites.

    Returns
    -------
    list of csr_matrix
        Sparse matrices [n_0, n_1, ..., n_{L-1}], each acting on a single site.
    """
    return [one_site_operator(n_projector, i, L) for i in range(L)]

def matter_density_by_site(L: int, model: str = 'QLM') -> list[csr_matrix]:
    """
    Constructs the site-resolved matter density operators:
    rho_j = -2 n_j + (1 + (-1)^j)/2
    
    Parameters
    ----------
    L : int
        Number of qubits or sites.

    Returns
    -------
    list of csr_matrix
        Sparse matrices [n_0, n_1, ..., n_{L-1}], each acting on a single site.
    """
    match model:
        case 'QLM':
            
            densities = [one_site_operator( 0.5 * ( (-(-1)**j) * sigma[2] + I2 ), j, L) 
                        for j in range(L)]
            return densities
        case 'schwinger':
            densities = [one_site_operator(-(-1)**i*sigma[2]/2., i, L) + identity(2**L, format="csr")/2. for i in range(L)]
            return densities
        case _:
            raise ValueError("Unrecognized model. Choose between QLM and schwinger.")
            
    return n_operator
def E_operators_by_site(L: int) -> list[csr_matrix]:
    """
    Constructs electric field operators E_i for each site i.

    Parameters
    ----------
    L : int
        Number of qubits or sites.

    Returns
    -------
    list of csr_matrix
        Sparse matrices [E_0, E_1, ..., E_{L-1}], each representing the local electric field.
    """
    return [one_site_operator((-1)**(i) * 0.5 * sigma[2], i, L) for i in range(L)]

def E_operators_by_site_schwinger(L: int, alpha: float) -> list[csr_matrix]:
    """
    Constructs Schwinger-model electric field operators with coupling alpha.

    Parameters
    ----------
    L : int
        Number of sites.
    alpha : float
        Coupling parameter for the electric field.

    Returns
    -------
    list of csr_matrix
        Sparse matrices representing Schwinger-model electric fields.
    """
    dim = 2**L
    E_ops = []
    I_full = identity(dim, format='csr')
    for j in range(L - 1):
        E_j = - alpha * I_full  
        for l in range(j + 1):
            E_j += 0.5 * one_site_operator(sigma[2] + (-1)**(l+1) * I2, l, L)  
        E_ops.append(E_j)
    return E_ops

# -------------------------------
#    Hamiltionian Operators
# -------------------------------

def H_E_lat(J, alpha, L):
    """
    Build the H^E_lat term (sparse csr) 
    Note: paper indices n,l,j run from 1..; in Python we map paper p -> python p-1.
    """
    dim = 2**L
    H = csr_matrix((dim, dim), dtype=np.float32)

    # Precompute one-site sigma^z operators 
    sz_sites = [one_site_operator(sigma[2], i, L) for i in range(L)]

    # To avoid confusion i keep the paper notation
    pref1 = J / 2.0
    for n in range(1, L-1):                         
        for l in range(n + 1, L):  
            weight = (L - l)            
            H += pref1 * weight * two_site_operator(n-1, l-1, precomputed_operators=sz_sites)

    pref2 = -(J / 4.0)
    for n in range(1, L):              
        parity = (-1) ** n
        for l in range(1, n + 1): 
            H += pref2 * (1-parity) * sz_sites[l-1]

    if alpha != 0:
        pref3 = -J * alpha
        for j in range(1, L):
            weight = (L - j)
            H += pref3 * weight * sz_sites[j-1]

    return H

def Compute_H_FSS_sparse(Omega: float, delta: float, L: int, Delta:float = 0) -> csr_matrix:
    """
    Constructs the full sparse Hamiltonian for a 1D Rydberg chain of length L.

    The Hamiltonian is given by:
        H = sum_i [ Omega * sigma_x^i + 2 * delta * n_i ]

    Parameters
    ----------
    Omega : float
        Rabi frequency.
    delta : float
        Detuning.
    L : int
        Number of qubits/sites.
    Delta: float
        Tunable Stark shift.
    Returns
    -------
    csr_matrix
        Sparse Hamiltonian matrix of size 2^L x 2^L.
    """
    # Precompute identity matrices
    dim = 2**L

    # Initialize the Hamiltonian
    H = csr_matrix((dim, dim), dtype=np.float32)

    for i in range(L):
        H += one_site_operator(Omega *sigma[0], i, L)
        H += one_site_operator(2.0 * delta * n_projector, i, L)
        H += one_site_operator(-Delta/2. * (-1)**i*sigma[2], i, L)
    return H

def Compute_H_schwinger_sparse(w: float, m: float, J: float, alpha: float, L: int) -> csr_matrix:
    """
    Constructs the full sparse Hamiltonian for the lattice Schwinger model.

    The Hamiltonian includes mass, kinetic, and gauge (electric field) terms.

    Parameters
    ----------
    w : float
        Hopping strength (kinetic term coefficient).
    m : float
        Mass of fermions.
    J : float
        Gauge field coupling.
    alpha : float
        Coupling parameter for the electric field operators.
    L : int
        Number of sites.

    Returns
    -------
    csr_matrix
        Sparse Hamiltonian matrix of size 2^L x 2^L.
    """
    dim = 2**L
    H = csr_matrix((dim, dim), dtype=np.float32)

    # mass term
    if m!= 0:
        for i in range(L):
            H += (m/2) * one_site_operator(((-1)**(i+1)) *sigma[2], i, L)  
                
    # kinetic (your convention uses -w in front of σ⁺σ⁻ + h.c.; keep it)
    for i in range(L-1):
        H += -w * two_site_operator(i, i+1, sigma_plus, sigma_minus, L)
        H += -w * two_site_operator(i, i+1, sigma_minus, sigma_plus, L)
    
    H += H_E_lat(J, alpha, L)
    return H
    