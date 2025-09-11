import numpy as np
from itertools import product
from scipy.sparse import dok_matrix, csr_matrix, vstack
from typing import List, Tuple, Union

def generate_blockade_basis_sparse(
    L: int,
    return_vectors: bool = False,
    open_boundaries: bool = False ) -> Union[List[int], Tuple[List[int], List[csr_matrix]]]:
    """
    Generate blockade-constrained basis states for a 1D chain of qubits,
    supporting open or periodic boundary conditions.

    A state is allowed if no two neighboring sites are both in the ground state (0).

    Parameters
    ----------
    L : int
        Length of the chain (number of qubits/sites).
    return_vectors : bool, optional
        If True, return sparse vector representations of the basis states as well.
    open_boundaries : bool, optional
        If True, use open boundary conditions; otherwise, periodic boundaries are used.

    Returns
    -------
    valid_indices : list of int
        Indices of valid basis states in the full 2^L-dimensional Hilbert space.
    sparse_vectors : list of csr_matrix, optional
        Sparse row vectors of valid basis states (only returned if return_vectors=True).
    """
    def is_valid(bits: Tuple[int, ...]) -> bool:
        """Check if a given bit string satisfies the blockade constraint."""
        for i in range(L - 1):
            if bits[i] + bits[i + 1] == 0:
                return False
        if not open_boundaries and bits[0] + bits[-1] == 0:
            return False
        return True

    valid_indices: List[int] = []
    sparse_vectors: List[csr_matrix] = []

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

def blockade_projection_matrix(L: int, open_boundaries: bool = False) -> csr_matrix:
    """
    Construct the sparse projection matrix from the full Hilbert space to
    the blockade-constrained subspace.

    Each row of the projection matrix corresponds to a valid blockade state.

    Parameters
    ----------
    L : int
        Length of the chain (number of qubits/sites).
    open_boundaries : bool, optional
        Whether to use open boundary conditions (default False for periodic).

    Returns
    -------
    P : csr_matrix
        Sparse matrix of shape (d, 2^L), where d is the number of blockade-allowed states.
    """
    _, basis_vectors = generate_blockade_basis_sparse(L, return_vectors=True, open_boundaries=open_boundaries)
    P = vstack(basis_vectors, format='csr')
    return P
