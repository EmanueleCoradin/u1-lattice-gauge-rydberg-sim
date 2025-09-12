import numpy as np
import pickle
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Load previously–selected peak indices and Rydberg data
# ----------------------------------------------------------------------
selected_indices = np.load("selected_peak_indices.npy")
e_vals_r = np.load("rydberg_evals.npy")
e_vecs_r = np.load("rydberg_evecs.npy")
C_r      = np.load("rydberg_coeffs.npy")

with open("projector_P.pkl", "rb") as f:
    P = pickle.load(f)
dA = int(np.load("rydberg_dA.npy"))
dB = int(np.load("rydberg_dB.npy"))

# ----------------------------------------------------------------------
# Recompute masks to match the interactive script
# ----------------------------------------------------------------------
prob_r = np.abs(C_r)**2
mask = prob_r >= 1e-9
e_masked = e_vals_r[mask]
p_masked = prob_r[mask]
threshold = 2e-2
mask_sig  = p_masked > threshold

# Energies of the manually selected peaks (sorted)
selected_energies = np.sort(e_masked[selected_indices])

# ----------------------------------------------------------------------
# Plot 1: spectral decomposition with highlights
# ----------------------------------------------------------------------
plt.figure(figsize=(9,5))
plt.vlines(e_masked, ymin=2e-9, ymax=p_masked, color="tab:blue", alpha=0.6, lw=0.8)
plt.plot(e_masked, p_masked, "o", ms=4, color="tab:blue")
plt.plot(e_masked[mask_sig], p_masked[mask_sig], "o", ms=6, color="tab:red",
         label="Significant")
if len(selected_indices) > 0:
    plt.plot(selected_energies, p_masked[selected_indices], "o", ms=8,
             color="gold", mec="k", mew=0.8, label="Selected")
plt.xlabel(r"Eigenvalue $E_n$")
plt.ylabel(r"$|\langle \phi_n | \psi_0 \rangle|^2$")
plt.title("Rydberg Spectral Decomposition")
plt.yscale("log")
plt.grid(True, which="both", ls="--", lw=0.6, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Plot 2: entropy vs eigenvalue with the same highlights
# ----------------------------------------------------------------------
from numpy.linalg import svd

def compute_entropy(psi_list, projector, dA, dB):
    entropy = []
    for psi in psi_list:
        psi_proj = projector.T @ psi
        psi_proj /= np.linalg.norm(psi_proj)
        rhoA = psi_proj.reshape(dA, dB)
        s = svd(rhoA, compute_uv=False)
        p = s**2
        p = p[p > 1e-12]
        entropy.append(-np.sum(p * np.log2(p)))
    return np.array(entropy)

all_psi_list = [e_vecs_r[:, i] for i in range(e_vecs_r.shape[1])]
entropy_all  = compute_entropy(all_psi_list, projector=P, dA=dA, dB=dB)

entropy_masked   = entropy_all[mask]
entropy_sig      = entropy_masked[mask_sig]
entropy_selected = entropy_masked[selected_indices]

plt.figure(figsize=(9,5))
plt.plot(e_vals_r, entropy_all, "o", color="tab:blue", ms=4, alpha=0.6,
         label="All eigenstates")
plt.plot(e_masked[mask_sig], entropy_sig, "o", color="tab:red", ms=6,
         label="Significant")
if len(selected_indices) > 0:
    plt.plot(selected_energies, entropy_selected, "o",
             color="gold", ms=8, mec="k", mew=0.8, label="Selected")
plt.xlabel(r"Eigenvalue $E_n$")
plt.ylabel("Entanglement Entropy (bits)")
plt.title("Entanglement Entropy vs Eigenvalue")
plt.grid(True, ls="--", lw=0.6, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
