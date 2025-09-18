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

def compare_models(
    models: List[str],
    psi_t_lists: List[List[np.ndarray]],
    n_ops_lists: List[List[np.ndarray]],
    E_ops_lists: List[List[np.ndarray]],
    rho_ops_lists: Optional[List[List[np.ndarray]]] = None,
    t_vals: np.ndarray = None,
    figsize: Tuple[int,int]=(14,12)
) -> None:
    """
    Compare multiple models with ⟨n_i⟩, ⟨E_i⟩, and optionally ⟨ρ_i⟩ dynamics 
    in a shared figure with subplots.

    Parameters
    ----------
    models : list of str
        Names of the models (e.g., ["Rydberg", "Schwinger"]).
    psi_t_lists : list of list of np.ndarray
        Each entry is a list of |psi(t)> states for that model.
    n_ops_lists : list of list of operators
        List of n_i operators for each model.
    E_ops_lists : list of list of operators
        List of E_i operators for each model.
    rho_ops_lists : list of list of operators
        List of rho_i operators for each model.
    t_vals : np.ndarray
        Common time values.
    figsize : tuple
        Size of the full comparison figure.
    
    """
    n_models = len(models)
    has_rho = rho_ops_lists is not None
    n_rows = 3 if has_rho else 2

    fig, axes = plt.subplots(n_rows, n_models, figsize=figsize, sharex=False)

    if n_models == 1:
        axes = np.array([[axes[i]] for i in range(n_rows)])

    for col, model in enumerate(models):
        psi_list = psi_t_lists[col]
        n_ops = n_ops_lists[col]
        E_ops = E_ops_lists[col]
        rho_ops = rho_ops_lists[col] if has_rho else None

        n_expect, E_expect, rho_expect = compute_expectations(psi_list, n_ops, E_ops, rho_ops)

        # n_i
        im_n = axes[0, col].imshow(n_expect, aspect='auto', origin='lower', cmap='inferno',
                                    extent=[-0.5, n_expect.shape[1]-0.5, t_vals[0], t_vals[-1]],
                                    vmin=0, vmax=1)
        axes[0, col].set_title(f"{model}: ⟨n̂ᵢ⟩")
        axes[0, col].set_ylabel("Time")
        fig.colorbar(im_n, ax=axes[0, col])

        # E_i
        im_E = axes[1, col].imshow(E_expect, aspect='auto', origin='lower', cmap='bwr',
                                   extent=[-0.5, E_expect.shape[1]-0.5, t_vals[0], t_vals[-1]],
                                   vmin=-0.5, vmax=0.5)
        axes[1, col].set_title(f"{model}: ⟨Êᵢ⟩")
        axes[1, col].set_ylabel("Time")
        fig.colorbar(im_E, ax=axes[1, col])

        # ρ_i
        if has_rho:
            im_rho = axes[2, col].imshow(rho_expect, aspect='auto', origin='lower', cmap='viridis',
                                         extent=[-0.5, rho_expect.shape[1]-0.5, t_vals[0], t_vals[-1]])
            axes[2, col].set_title(f"{model}: ⟨ρ̂ᵢ⟩")
            axes[2, col].set_ylabel("Time")
            fig.colorbar(im_rho, ax=axes[2, col])

    plt.tight_layout()
    plt.show()

def plot_echo_entropy(times: Sequence[float],
                      echo: np.ndarray,
                      entropy: np.ndarray) -> None:
    """
    Plot Loschmidt echo (or its logarithm) and entanglement entropy vs time.

    Parameters
    ----------
    times : Sequence[float]
        Time points corresponding to the states.
    echo : np.ndarray
        Array of Loschmidt echo values (|<psi(t)|psi(0)>|^2).
    entropy : np.ndarray
        Array of entanglement entropies.
    base : float, optional
        Logarithm base used when log_echo=True (default: natural log).
    """
    fig, ax1 = plt.subplots()

    y = echo
    ax1.set_ylabel("Loschmidt Echo L(t)")
    ax1.plot(times, y, label="Loschmidt Echo", color="tab:blue")
    ax1.set_xlabel("Time")

    # Plot entropy on the right axis
    ax2 = ax1.twinx()
    ax2.plot(times, entropy, label="Entanglement Entropy (bits)", color="tab:orange")
    ax2.set_ylabel("Entanglement Entropy")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Loschmidt Echo and Entanglement Entropy vs Time")
    plt.show()