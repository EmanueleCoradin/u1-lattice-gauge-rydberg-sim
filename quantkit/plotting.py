import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Sequence, Optional
from .dynamics import compute_expectations

def Plot_Matrix(density_matrix: np.ndarray) -> None:
    """
    Plot a density matrix with both its real and imaginary parts annotated.

    The plot shows the real part of the matrix as a heatmap, while each cell
    is annotated with the full complex number (real + imaginary).

    Parameters
    ----------
    density_matrix : np.ndarray
        2D square array representing the density matrix of a quantum state.

    Returns
    -------
    None
        Displays the plot; does not return a value.
    """
    # Check that the matrix is square
    if density_matrix.shape[0] != density_matrix.shape[1]:
        raise ValueError("Density matrix must be square.")

    plt.figure(figsize=(7, 5))

    # Plot the real part of the matrix as a heatmap
    plt.imshow(
        np.real(density_matrix),
        cmap='coolwarm',
        interpolation='nearest',
        vmin=np.min(np.real(density_matrix)),
        vmax=np.max(np.real(density_matrix))
    )

    # Add color bar to indicate the real part
    cbar = plt.colorbar()
    cbar.set_label('Real Part of Matrix Elements')

    # Annotate each cell with the complex number (real + imaginary) up to 2 decimal places
    for i in range(density_matrix.shape[0]):
        for j in range(density_matrix.shape[1]):
            val = density_matrix[i, j]
            plt.text(
                j, i,
                f'{np.real(val):.2f} + {np.imag(val):.2f}j',
                ha='center', va='center', color='black', fontsize=9
            )

    # Title and axis labels
    plt.title("Mixed State Density Matrix")
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    # Show indices on x and y axes
    plt.xticks(np.arange(density_matrix.shape[1]))
    plt.yticks(np.arange(density_matrix.shape[0]))

    # Use tight layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_expectations(
    n_expect: np.ndarray,
    E_expect: np.ndarray,
    t_vals: np.ndarray,
    title_n: str = "⟨n̂ᵢ⟩",
    title_E: str = "⟨Êᵢ⟩",
    vmin_n: float = 0.0,
    vmax_n: float = 1.0,
    vmin_E: float = -0.5,
    vmax_E: float = 0.5,
    figsize: Tuple[int,int]=(8,12)
) -> None:
    """
    Plot time-dependent expectation values for n_i and E_i operators.

    Parameters
    ----------
    n_expect : np.ndarray
        Expectation values of n_i over time (T x L_n).
    E_expect : np.ndarray
        Expectation values of E_i over time (T x L_E).
    t_vals : np.ndarray
        Array of time points corresponding to psi_t_list.
    title_n : str
        Title for n_i plot.
    title_E : str
        Title for E_i plot.
    vmin_n, vmax_n : float
        Color scale limits for n_i.
    vmin_E, vmax_E : float
        Color scale limits for E_i.
    figsize : tuple
        Figure size.
    """
    L_n = n_expect.shape[1]
    L_E = E_expect.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot n_i
    extent_n = [-0.5, L_n-0.5, t_vals[0], t_vals[-1]]
    im1 = axes[0].imshow(n_expect, aspect='auto', origin='lower',
                         extent=extent_n, cmap='inferno',
                         vmin=vmin_n, vmax=vmax_n)
    axes[0].set_ylabel("Time")
    axes[0].set_title(title_n)
    axes[0].grid(True)
    fig.colorbar(im1, ax=axes[0], label=title_n)

    # Plot E_i
    extent_E = [-0.5, L_E-0.5, t_vals[0], t_vals[-1]]
    im2 = axes[1].imshow(E_expect, aspect='auto', origin='lower',
                         extent=extent_E, cmap='bwr',
                         vmin=vmin_E, vmax=vmax_E)
    axes[1].set_xlabel("Site / Link index")
    axes[1].set_ylabel("Time")
    axes[1].set_title(title_E)
    axes[1].grid(True)
    fig.colorbar(im2, ax=axes[1], label=title_E)

    plt.tight_layout()
    plt.show()


def plot_models(
    models: List[str],
    psi_t_lists: List[List[np.ndarray]],
    n_ops_lists: List[List[np.ndarray]],
    E_ops_lists: List[List[np.ndarray]],
    t_vals: np.ndarray
) -> None:
    """
    Compare multiple models by plotting their ⟨n_i⟩ and ⟨E_i⟩ dynamics.

    Parameters
    ----------
    models : list of str
        Names of the models (e.g., "Rydberg", "QLM", "Schwinger").
    psi_t_lists : list of list of np.ndarray
        Each entry is a list of |psi(t)> states for that model.
    n_ops_lists : list of list of operators
        List of n_i operators for each model.
    E_ops_lists : list of list of operators
        List of E_i operators for each model.
    t_vals : np.ndarray
        Common time values.
    """
    for model, psi_t_list, n_ops, E_ops in zip(models, psi_t_lists, n_ops_lists, E_ops_lists):
        print(f"Plotting expectations for {model}")
        n_expect, E_expect = compute_expectations(psi_t_list, n_ops, E_ops)
        plot_expectations(n_expect, E_expect, t_vals,
                          title_n=f"{model}: ⟨n̂ᵢ⟩",
                          title_E=f"{model}: ⟨Êᵢ⟩")

def plot_model_expectations(
    results: dict,
    t_vals: np.ndarray,
    figsize: tuple[int, int] = (14, 12)
):
    """
    Plot expectation values ⟨n_i⟩, ⟨E_i⟩, and optionally ⟨ρ_i⟩
    for one or more models as space-time heatmaps.

    Parameters
    ----------
    results : dict
        Dictionary of model results, typically produced by
        `compute_model_expectations`. Expected format:
            {
                "ModelName": {
                    "n": np.ndarray (T × L_n),
                    "E": np.ndarray (T × L_E),
                    "rho": np.ndarray (T × L_rho) or None
                },
                ...
            }
        where T = number of time steps, L_* = number of sites/links.
    t_vals : np.ndarray
        Array of time values of length T, aligned with the evolution in `results`.
    figsize : tuple of int, optional
        Size of the matplotlib figure (default: (14, 12)).

    """
    n_models = len(results)
    has_rho = any(res["rho"] is not None for res in results.values())
    n_rows = 3 if has_rho else 2

    fig, axes = plt.subplots(n_rows, n_models, figsize=figsize, sharex=False)
    if n_models == 1:
        axes = np.array([[axes[i]] for i in range(n_rows)])

    for col, (model, res) in enumerate(results.items()):
        n_exp, E_exp, rho_exp = res["n"], res["E"], res["rho"]

        # n_i
        im_n = axes[0, col].imshow(
            n_exp, aspect="auto", origin="lower", cmap="inferno",
            extent=[-0.5, n_exp.shape[1]-0.5, t_vals[0], t_vals[-1]],
            vmin=0, vmax=1
        )
        axes[0, col].set_title(f"{model}: ⟨n̂ᵢ⟩")
        axes[0, col].set_ylabel("Time")
        fig.colorbar(im_n, ax=axes[0, col])

        # E_i
        im_E = axes[1, col].imshow(
            E_exp, aspect="auto", origin="lower", cmap="bwr",
            extent=[-0.5, E_exp.shape[1]-0.5, t_vals[0], t_vals[-1]],
            vmin=-0.5, vmax=0.5
        )
        axes[1, col].set_title(f"{model}: ⟨Êᵢ⟩")
        axes[1, col].set_ylabel("Time")
        fig.colorbar(im_E, ax=axes[1, col])

        # ρ_i
        if has_rho and rho_exp is not None:
            im_rho = axes[2, col].imshow(
                rho_exp, aspect="auto", origin="lower", cmap="viridis",
                extent=[-0.5, rho_exp.shape[1]-0.5, t_vals[0], t_vals[-1]]
            )
            axes[2, col].set_title(f"{model}: ⟨ρ̂ᵢ⟩")
            axes[2, col].set_ylabel("Time")
            fig.colorbar(im_rho, ax=axes[2, col])

    plt.tight_layout()
    plt.show()


def plot_echo_entropy(
    times: Sequence[float],
    echo: np.ndarray,
    entropy: np.ndarray,
    rho_expect: Optional[np.ndarray] = None
) -> None:
    """
    Plot Loschmidt echo, entanglement entropy, and optionally the
    mean matter density ⟨ρ̄(t)⟩ vs time.

    Parameters
    ----------
    times : Sequence[float]
        Time points corresponding to the states.
    echo : np.ndarray
        Array of Loschmidt echo values (|<psi(t)|psi(0)>|^2), in [0,1].
    entropy : np.ndarray
        Array of entanglement entropies.
    rho_expect : np.ndarray, optional
        Array of shape (T, L) with per-site ⟨ρ_j⟩ values.
        If provided, the mean density across sites is plotted (also ∈ [0,1]).
    """
    fig, ax1 = plt.subplots()

    # Loschmidt echo
    ax1.plot(times, echo, label="Loschmidt Echo", color="tab:blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Loschmidt Echo / Mean ⟨ρ⟩", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Mean density (if available)
    if rho_expect is not None:
        mean_rho = rho_expect.mean(axis=1)
        ax1.plot(times, mean_rho, label="Mean ⟨ρ⟩", color="tab:green", linestyle="--")

    # Entropy on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(times, entropy, label="Entanglement Entropy", color="tab:orange")
    ax2.set_ylabel("Entanglement Entropy", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Title
    plt.title("Loschmidt Echo, Entropy, and Mean ⟨ρ⟩ vs Time")

    # Combine legends
    lines, labels = [], []
    for ax in [ax1, ax2]:
        lns, lbs = ax.get_legend_handles_labels()
        lines.extend(lns)
        labels.extend(lbs)
    fig.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

def plot_electric_field(
    times: Sequence[float],
    E_expect: np.ndarray,
    site_indices: Sequence[int] = (0, 1)
) -> None:
    """
    Plot the time evolution of the electric field ⟨E_j⟩ for two selected sites.

    Parameters
    ----------
    times : Sequence[float]
        Array of time points.
    E_expect : np.ndarray
        Electric field expectation values, shape (timesteps, number_of_sites).
    site_indices : tuple of int, default=(0,1)
        Indices of the two sites to plot.
    """
    fig, ax = plt.subplots(figsize=(8,5))

    for j in site_indices:
        ax.plot(times, E_expect[:, j], label=f"Site {j}")

    ax.set_xlabel("Time")
    ax.set_ylabel("⟨E_j⟩")
    ax.set_title("Electric Field Evolution at Selected Sites")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_fft(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    site_index: int = None,
    observable_name: str = "Observable",
    plot_DC: bool = False
) -> None:
    """
    Plot the power spectrum of a given observable.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array from compute_fft.
    spectrum : np.ndarray
        Power spectrum |FFT|^2.
    site_index : int
        Site index (for labeling). 
    observable_name : str
        Name of the observable (for title/labels).
    plot_DC : bool
        if True the DC component of the power spectrum is also plotted
    """
    plt.figure(figsize=(8,5))
    plt.plot(freqs[1:], spectrum[1:], color="tab:blue")
    plt.xlabel("Frequency")
    plt.ylabel("Power Spectrum |FFT|²")
    if site_index is None:
        plt.title(f"{observable_name} Spectrum")
    else:
        plt.title(f"{observable_name} Spectrum at Site {site_index}")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
