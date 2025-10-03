import numpy as np
import matplotlib.pyplot as plt
import pickle

# ----------------------------------------------------------------------
# Load data saved in main.py
# ----------------------------------------------------------------------
e_vals_r = np.load("rydberg_evals.npy")
e_vecs_r = np.load("rydberg_evecs.npy")
C_r      = np.load("rydberg_coeffs.npy")
dA       = int(np.load("rydberg_dA.npy"))
dB       = int(np.load("rydberg_dB.npy"))
with open("projector_P.pkl", "rb") as f:
    P = pickle.load(f)

# ----------------------------------------------------------------------
# Spectral weights
# ----------------------------------------------------------------------
prob_r = np.abs(C_r)**2
mask = prob_r >= 1e-9
e_masked = e_vals_r[mask]
p_masked = prob_r[mask]

threshold = 2e-2
mask_sig  = p_masked > threshold

# ----------------------------------------------------------------------
# Interactive selection of peaks
# ----------------------------------------------------------------------
selected_indices = []

def on_pick(event):
    if hasattr(event, "ind") and len(event.ind):
        idx = event.ind[0]
        if idx not in selected_indices:
            selected_indices.append(idx)
            print(f"Selected peak: E = {e_masked[idx]:.6f}, "
                  f"weight = {p_masked[idx]:.3e}")

fig, ax = plt.subplots(figsize=(9,5))
ax.vlines(e_masked, ymin=2e-9, ymax=p_masked, color="tab:blue", alpha=0.6, lw=0.8)
ax.plot(e_masked, p_masked, "o", ms=4, color="tab:blue", picker=5)
ax.plot(e_masked[mask_sig], p_masked[mask_sig], "o", ms=6, color="tab:red",
        label="Significant")
ax.set_xlabel(r"Eigenvalue $E_n$")
ax.set_ylabel(r"$|\langle \phi_n | \psi_0 \rangle|^2$")
ax.set_title("Rydberg Spectral Decomposition")
ax.set_yscale("log")
ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.7)
ax.legend()
fig.tight_layout()
fig.canvas.mpl_connect('pick_event', on_pick)

print("Click on peaks to select them. Close the window when done.")
plt.show()

# ----------------------------------------------------------------------
# Compute average ΔE between selected peaks
# ----------------------------------------------------------------------
selected_indices = np.array(selected_indices, dtype=int)
selected_energies = np.sort(e_masked[selected_indices])

if len(selected_energies) > 1:
    gaps = np.diff(selected_energies)
    avg_gap = gaps.mean()
    std_gap = gaps.std()
    print("\nSelected energies (sorted):")
    print(selected_energies)
    print("\nIndividual energy gaps ΔE:")
    for g in gaps:
        print(f"{g:.6f}")
    print(f"\nAverage ΔE between selected peaks: {avg_gap:.6f} \pm {std_gap:.6f}")
else:
    print("\nNot enough peaks selected to compute a spacing.")

# ----------------------------------------------------------------------
# Compute entanglement entropy of all eigenstates
# ----------------------------------------------------------------------
def compute_entropy(psi_list, projector, dA, dB):
    from numpy.linalg import svd
    entropy = []
    for psi in psi_list:
        psi_proj = projector.T @ psi
        psi_proj = psi_proj / np.linalg.norm(psi_proj)
        rhoA = psi_proj.reshape(dA, dB)
        s = svd(rhoA, compute_uv=False)
        p = s**2
        p = p[p > 1e-12]
        entropy.append(-np.sum(p * np.log2(p)))
    return np.array(entropy)

all_psi_list = [e_vecs_r[:, i] for i in range(e_vecs_r.shape[1])]
entropy_all = compute_entropy(all_psi_list, projector=P, dA=dA, dB=dB)

entropy_masked   = entropy_all[mask]
entropy_sig      = entropy_masked[mask_sig]
entropy_selected = entropy_masked[selected_indices]

# ----------------------------------------------------------------------
# Plot entropy vs eigenvalue
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9,5))
ax2.plot(e_vals_r, entropy_all, "o", color="tab:blue", ms=4, alpha=0.6,
         label="All eigenstates")
ax2.plot(e_masked[mask_sig], entropy_sig, "o", color="tab:red", ms=6,
         label="Significant (threshold)")
if len(selected_indices) > 0:
    ax2.plot(selected_energies, entropy_selected, "o",
             color="gold", ms=8, mec="k", mew=0.8,
             label="Manually selected")
ax2.set_xlabel(r"Eigenvalue $E_n$")
ax2.set_ylabel("Entanglement Entropy (bits)")
ax2.set_title("Entanglement Entropy vs Eigenvalue")
ax2.grid(True, ls="--", lw=0.6, alpha=0.7)
ax2.legend()
fig2.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Save your manual selection for reuse
# ----------------------------------------------------------------------
np.save("selected_peak_indices.npy", selected_indices)
print("Saved manual selection to selected_peak_indices.npy")
