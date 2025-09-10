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

sigma_plus  = sigma[0] + 1j * sigma[1]
sigma_minus = sigma[0] - 1j * sigma[1]

I2 = identity(2, format="csr")

n_projector = (sigma[2] + I2) * 0.5

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
    precomputed_operators: Optional[List[csr_matrix]] = None
) -> csr_matrix:
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

    # Precompute identity matrices for tensor product construction
    I_full = [identity(2**i, format="csr") for i in range(L)]
    
    # Initialize the n_operator as a sparse zero matrix of appropriate size
    n_operator = csr_matrix((dim, dim), dtype=np.float32)
    
    for i in range(L):
        # Construct the left and right parts of the tensor product
        left = I_full[i] if i > 0 else identity(1, format='csr')
        right = I_full[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        
        # Update the n_operator with the tensor product
        n_operator += kron(kron(left, n_projector), right)
    
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
    I_full = [identity(2**i, format="csr") for i in range(L)]
    n_ops = []

    for i in range(L):
        left = I_full[i] if i > 0 else identity(1, format='csr')
        right = I_full[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        n_i = kron(kron(left, n_projector), right)
        n_ops.append(n_i)

    return n_ops

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
    I_full = [identity(2**i, format="csr") for i in range(L)]
    E_ops = []

    for i in range(L):
        left = I_full[i] if i > 0 else identity(1, format='csr')
        right = I_full[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        E_i = kron(kron(left, (-1)**(i) * 0.5 * sigma[2]), right)
        E_ops.append(E_i)

    return E_ops

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
    I_full = identity(dim, format='csr')
    H = csr_matrix((dim, dim), dtype=np.float64)

    # Precompute one-site sigma^z operators (python indices 0..L-1)
    sz_sites = [one_site_operator(sigma[2], site, L) for site in range(L)]

    # 1) (J/2) * sum_{n=1}^{L-2} sum_{l=n+1}^{L-1} (L - l) σ^z_n σ^z_l
    # paper indices n=1..L-2, l=n+1..L-1 -> python n_py=0..L-3, l_py = n_py+1..L-2
    pref1 = J / 2.0
    for n_paper in range(1, L-1):               # n_paper = 1..L-2
        n = n_paper - 1                         # python index
        for l_paper in range(n_paper + 1, L):  # l_paper = n+1 .. L-1
            l = l_paper - 1
            weight = (L - l_paper)             # (L - l) with paper l
            H += pref1 * weight * two_site_operator(n, l, precomputed_operators=sz_sites)

    # sigma piece: - (J/4) * sum_{n=1}^{L-1} [ - (-1)^n sum_{l=1}^n σ^z_l ]
    # compute as nested sum; map indices: l_paper=1..n -> python l=0..n-1
    pref_sigma_piece = -(J / 4.0)
    for n_paper in range(1, L):               # n_paper = 1..L-1
        parity = (-1) ** n_paper
        for l_paper in range(1, n_paper + 1): # l_paper = 1..n_paper
            l = l_paper - 1
            H += pref_sigma_piece * (1-parity) * sz_sites[l]

    # 3) - J * alpha * sum_{j=1}^{L-1} (L - j) σ^z_j
    # map j_paper -> python j = j_paper -1, j_paper runs 1..L-1
    if alpha != 0.0:
        pref3 = -J * alpha
        for j_paper in range(1, L):            # j_paper = 1..L-1
            j = j_paper - 1
            weight = (L - j_paper)
            H += pref3 * weight * sz_sites[j]

    return H


def Compute_H_FSS_sparse(Omega: float, delta: float, L: int) -> csr_matrix:
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

    Returns
    -------
    csr_matrix
        Sparse Hamiltonian matrix of size 2^L x 2^L.
    """
    # Precompute identity matrices
    dim = 2**L
    I_full = [identity(2**i, format="csr") for i in range(L)]

    # Initialize the Hamiltonian
    H = csr_matrix((dim, dim), dtype=np.float32)

    for i in range(L):
        left = I_full[i] if i > 0 else identity(1, format='csr')
        right = I_full[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        H += Omega * kron(kron(left, sigma[0]), right)
        H += kron(kron(left, 2.0 * delta * n_projector), right)

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
    H = csr_matrix((dim, dim), dtype=np.float64)
    I_full = [identity(2**i, format="csr") for i in range(L)]

    # mass term
    for i in range(L):
        left  = I_full[i] if i > 0 else identity(1, format='csr')
        right = I_full[L - i - 1] if (L - i - 1) > 0 else identity(1, format='csr')
        H += (m/2) * ((-1)**(i+1)) * kron(kron(left, sigma[2]), right)

    # kinetic (your convention uses -w in front of σ⁺σ⁻ + h.c.; keep it)
    for i in range(L-1):
        left  = I_full[i] if i > 0 else identity(1, format='csr')
        right = I_full[L - i - 2] if (L - i - 2) > 0 else identity(1, format='csr')
        H += -w * kron(kron(kron(left, sigma_plus),  sigma_minus), right)
        H += -w * kron(kron(kron(left, sigma_minus), sigma_plus),  right)

        H += H_E_lat(J, alpha, L)

    return H
    