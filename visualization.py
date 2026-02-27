# visualization.py
# =============================================================
# Visualization for Euler-Bernoulli Beam FEA Simulation Results
# Plots:
#   1. Acceleration time history
#   2. FFT (frequency content at each node)
#   3. FRF (Frequency Response Function)
#   4. Mode shapes (spatial deformation at resonant frequencies)
#
# SCALE CONVENTION:
#   X-axis (frequency) : log scale
#   Y-axis (magnitude) : linear scale
#   Y-axis (phase)     : linear scale
#   Frequency range    : full range (1 Hz to Nyquist)
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# =============================================================
# STEP 1 — LOAD CSV AND PARSE
# =============================================================

def load_simulation_csv(file_path, delimiter=';'):
    """
    Load one simulation CSV file and parse node acceleration data.

    Inputs:
        file_path : path to sim_XXXX.csv
        delimiter : delimiter used in serialized node columns

    Output:
        params      : dict of simulation parameters
        node_accels : numpy array shape (101, n_steps)
        time_vector : numpy array shape (n_steps,)
    """
    df = pd.read_csv(file_path)

    # --- Extract metadata ---
    params = {
        'sim_id'        : int(df['sim_id'].iloc[0]),
        'material'      : df['material'].iloc[0],
        'E_psi'         : float(df['E_psi'].iloc[0]),
        'rho_lbm_in3'   : float(df['rho_lbm_in3'].iloc[0]),
        'length_in'     : float(df['length_in'].iloc[0]),
        'width_in'      : float(df['width_in'].iloc[0]),
        'thickness_in'  : float(df['thickness_in'].iloc[0]),
        'impact_F0_lbf' : float(df['impact_F0_lbf'].iloc[0]),
        'impact_tau_s'  : float(df['impact_tau_s'].iloc[0]),
        'rayleigh_alpha': float(df['rayleigh_alpha'].iloc[0]),
        'rayleigh_beta' : float(df['rayleigh_beta'].iloc[0]),
        'dt_s'          : float(df['dt_s'].iloc[0]),
        'T_s'           : float(df['T_s'].iloc[0]),
        'n_steps'       : int(df['n_steps'].iloc[0]),
        'n_elements'    : int(df['n_elements'].iloc[0]),
        'n_nodes'       : int(df['n_nodes'].iloc[0]),
    }

    n_nodes = params['n_nodes']
    n_steps = params['n_steps']
    dt      = params['dt_s']

    # --- Parse node acceleration columns ---
    node_accels = np.zeros((n_nodes, n_steps))
    for i in range(n_nodes):
        col_name           = f"node_{i+1:03d}_accel"
        raw                = df[col_name].iloc[0]
        node_accels[i, :] = np.array(raw.split(delimiter), dtype=float)

    # --- Build time vector ---
    time_vector = np.linspace(0, dt * (n_steps - 1), n_steps)

    # --- Nyquist frequency ---
    f_nyquist = 1.0 / (2.0 * dt)

    print(f"Loaded  : {os.path.basename(file_path)}")
    print(f"  Material   : {params['material']}")
    print(f"  Length     : {params['length_in']} in")
    print(f"  Width      : {params['width_in']} in")
    print(f"  Force      : {params['impact_F0_lbf']} lbf")
    print(f"  n_nodes    : {n_nodes}")
    print(f"  n_steps    : {n_steps}")
    print(f"  dt         : {dt} s")
    print(f"  f_nyquist  : {round(f_nyquist, 1)} Hz")

    return params, node_accels, time_vector


# =============================================================
# STEP 2 — FFT
# =============================================================

def compute_fft(node_accels, dt):
    """
    Compute FFT for each node.

    Inputs:
        node_accels : (n_nodes, n_steps)
        dt          : timestep (s)

    Output:
        freqs   : one-sided frequency array (Hz)
        fft_mag : FFT magnitude (n_nodes, n_freq)
    """
    n_steps = node_accels.shape[1]
    freqs   = np.fft.rfftfreq(n_steps, d=dt)
    fft_raw = np.fft.rfft(node_accels, axis=1)
    fft_mag = (2.0 / n_steps) * np.abs(fft_raw)
    return freqs, fft_mag


def plot_fft(freqs, fft_mag, params,
             nodes_to_plot=[0, 24, 49, 74, 99],
             save_path=None):
    """
    Plot FFT magnitude spectrum.

    X-axis : log scale, full range (1 Hz to Nyquist)
    Y-axis : linear scale

    Inputs:
        freqs         : frequency array (Hz)
        fft_mag       : (n_nodes, n_freq) FFT magnitude
        params        : simulation parameter dict
        nodes_to_plot : list of node indices (0-based)
        save_path     : optional path to save figure
    """
    # Skip index 0 (DC / 0 Hz) to allow log scale
    f_plot  = freqs[1:]
    f_min   = f_plot[0]
    f_max   = f_plot[-1]

    colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(nodes_to_plot)))

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, node_idx in enumerate(nodes_to_plot):
        mag = fft_mag[node_idx, 1:]      # skip DC
        ax.plot(f_plot, mag,
                color=colors[idx],
                linewidth=1.2,
                label=f"Node {node_idx + 1}")

    ax.set_xscale('log')
    ax.set_xlim([f_min, f_max])
    ax.set_xlabel(
        f"Frequency (Hz) — log scale   "
        f"[{round(f_min, 2)} to {round(f_max, 1)} Hz]",
        fontsize=11)
    ax.set_ylabel("Acceleration Magnitude (in/s²)", fontsize=11)
    ax.set_title(
        f"FFT — Acceleration Frequency Spectrum\n"
        f"{params['material'].title()} | "
        f"L={params['length_in']} in | "
        f"b={params['width_in']} in | "
        f"F0={params['impact_F0_lbf']} lbf",
        fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"FFT plot saved: {save_path}")

    plt.show()
    return fig


# =============================================================
# STEP 3 — FRF
# =============================================================

def compute_frf(node_accels, force_time, dt):
    """
    Compute Frequency Response Function.

    FRF = FFT(output acceleration) / FFT(input force)

    Inputs:
        node_accels : (n_nodes, n_steps) acceleration (in/s²)
        force_time  : (n_steps,) force time history (lbf)
        dt          : timestep (s)

    Output:
        freqs     : frequency array (Hz)
        frf_mag   : FRF magnitude (n_nodes, n_freq) in/s²/lbf
        frf_phase : FRF phase (n_nodes, n_freq) degrees
    """
    n_steps    = node_accels.shape[1]
    freqs      = np.fft.rfftfreq(n_steps, d=dt)

    F_fft      = np.fft.rfft(force_time)
    A_fft      = np.fft.rfft(node_accels, axis=1)

    # Avoid division by zero
    F_fft_safe = F_fft.copy()
    F_fft_safe[np.abs(F_fft_safe) < 1e-10] = 1e-10

    frf        = A_fft / F_fft_safe[np.newaxis, :]
    frf_mag    = np.abs(frf)
    frf_phase  = np.angle(frf, deg=True)

    return freqs, frf_mag, frf_phase


def plot_frf(freqs, frf_mag, frf_phase, params,
             nodes_to_plot=[0, 24, 49, 74, 99],
             save_path=None):
    """
    Plot FRF magnitude and phase.

    Magnitude : X log scale, Y linear scale
    Phase     : X log scale, Y linear scale (-180 to 180 deg)
    Full frequency range (1 Hz to Nyquist)

    Inputs:
        freqs         : frequency array (Hz)
        frf_mag       : (n_nodes, n_freq) FRF magnitude
        frf_phase     : (n_nodes, n_freq) FRF phase (degrees)
        params        : simulation parameter dict
        nodes_to_plot : list of node indices (0-based)
        save_path     : optional path to save figure
    """
    # Skip DC bin for log scale
    f_plot  = freqs[1:]
    f_min   = f_plot[0]
    f_max   = f_plot[-1]

    colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(nodes_to_plot)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for idx, node_idx in enumerate(nodes_to_plot):
        mag   = frf_mag[node_idx, 1:]
        phase = frf_phase[node_idx, 1:]

        axes[0].plot(f_plot, mag,
                     color=colors[idx],
                     linewidth=1.2,
                     label=f"Node {node_idx + 1}")
        # Mask phase where magnitude is too small (noisy)
        mag_threshold = np.max(mag) * 0.001
        phase_masked  = np.where(mag > mag_threshold, phase, np.nan)
        axes[1].plot(f_plot, phase_masked,
                     color=colors[idx],
                     linewidth=1.0,
                     label=f"Node {node_idx + 1}")

    # --- Magnitude plot ---
    axes[0].set_xscale('log')
    axes[0].set_xlim([f_min, f_max])
    axes[0].set_yscale('log')
    axes[0].set_ylabel("FRF Magnitude (in/s²/lbf) — log scale", fontsize=11)
    axes[0].set_title(
        f"Frequency Response Function (FRF)\n"
        f"{params['material'].title()} | "
        f"L={params['length_in']} in | "
        f"b={params['width_in']} in | "
        f"F0={params['impact_F0_lbf']} lbf",
        fontsize=11)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, which='both', alpha=0.3)

    # --- Phase plot ---
    axes[1].set_xscale('log')
    axes[1].set_xlim([f_min, f_max])
    axes[1].set_ylim([-180, 180])
    axes[1].set_xlabel(
        f"Frequency (Hz) — log scale   "
        f"[{round(f_min, 2)} to {round(f_max, 1)} Hz]",
        fontsize=11)
    axes[1].set_ylabel("Phase (degrees)", fontsize=11)
    axes[1].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"FRF plot saved: {save_path}")

    plt.show()
    return fig

def compute_mode_shapes_from_matrices(params, n_modes=6):
    """
    Compute exact mode shapes using eigenvalue analysis
    on the assembled K and M matrices.

    This is the most reliable method — directly solves:
        K * phi = omega^2 * M * phi

    Inputs:
        params  : simulation parameter dict
        n_modes : number of modes to compute

    Output:
        nat_freqs   : natural frequencies in Hz (n_modes,)
        mode_shapes : (n_nodes, n_modes) array
                      each column is one normalized mode shape
                      Row 0 = node 0 (fixed end) = 0
                      Row 100 = node 100 (free end)
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from scipy.linalg import eigh
    from assembly import assemble_global_matrices, apply_boundary_conditions
    from materials import get_material
    from config import N_NODES

    # --- Rebuild matrices from params ---
    mat  = get_material(params['material'])
    E    = mat['E']
    rho  = mat['rho']
    b    = params['width_in']
    t    = params['thickness_in']
    L    = params['length_in']

    K, M                      = assemble_global_matrices(E, rho, b, t, L)
    K_free, M_free, free_dofs = apply_boundary_conditions(K, M)

    # --- Solve eigenvalue problem ---
    # K * phi = omega^2 * M * phi
    # eigsh returns smallest eigenvalues (lowest frequencies)
    K_dense = K_free.toarray()
    M_dense = M_free.toarray()

    # eigh solves full symmetric eigenvalue problem
    # returns ALL eigenvalues sorted ascending automatically
    eigenvalues, eigenvectors = eigh(K_dense, M_dense)

    # Take only first n_modes
    eigenvalues  = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]

    # Natural frequencies in Hz
    omega      = np.sqrt(np.abs(eigenvalues))
    nat_freqs  = omega / (2.0 * np.pi)

    # --- Map eigenvectors back to full 101 node space ---
    # free_dofs contains indices of free DOFs in global system
    # We only want transverse DOFs (even indices)
    n_nodes     = N_NODES
    mode_shapes = np.zeros((n_nodes, n_modes))

    for mode_idx in range(n_modes):
        evec = eigenvectors[:, mode_idx]

        for node_i in range(n_nodes):
            global_dof = 2 * node_i          # transverse DOF
            if global_dof in free_dofs:
                free_idx = np.where(
                    free_dofs == global_dof)[0][0]
                mode_shapes[node_i, mode_idx] = evec[free_idx]
            else:
                mode_shapes[node_i, mode_idx] = 0.0  # fixed end

        # Normalize so free end tip = positive and max abs = 1
        tip_val = mode_shapes[100, mode_idx]
        max_val = np.max(np.abs(mode_shapes[:, mode_idx]))
        if max_val > 0:
            mode_shapes[:, mode_idx] /= max_val
        # Flip so tip is positive
        if mode_shapes[100, mode_idx] < 0:
            mode_shapes[:, mode_idx] *= -1

    print(f"\nEigenvalue-based natural frequencies:")
    for i in range(n_modes):
        print(f"  Mode {i+1}: {round(nat_freqs[i], 2)} Hz")

    return nat_freqs, mode_shapes

# =============================================================
# STEP 4 — IDENTIFY RESONANT FREQUENCIES (PEAKS)
# =============================================================

def find_resonant_frequencies(freqs, frf_mag, tip_node_idx=100,
                               f_min=1.0, n_peaks=6):
    """
    Find the top N resonant frequencies from FRF at tip node.
    Searches full frequency range from f_min to Nyquist.

    Inputs:
        freqs        : frequency array (Hz)
        frf_mag      : (n_nodes, n_freq) FRF magnitude
        tip_node_idx : node to use for peak detection (tip = 100)
        f_min        : minimum frequency to search (Hz)
        n_peaks      : number of peaks to find

    Output:
        peak_freqs   : array of resonant frequencies (Hz)
        peak_indices : array of indices into freqs array
    """
    from scipy.signal import find_peaks

    freq_mask    = freqs >= f_min
    f_search     = freqs[freq_mask]
    mag_search   = frf_mag[tip_node_idx, freq_mask]

    prominence   = np.max(mag_search) * 0.01
    peaks, _     = find_peaks(mag_search,
                               prominence=prominence,
                               distance=5)

    peak_mags    = mag_search[peaks]
    sorted_peaks = peaks[np.argsort(peak_mags)[::-1]][:n_peaks]
    sorted_peaks = np.sort(sorted_peaks)

    peak_freqs   = f_search[sorted_peaks]
    peak_indices = np.where(np.isin(freqs, f_search[sorted_peaks]))[0]

    print(f"\nTop {len(peak_freqs)} resonant frequencies found:")
    for i, f in enumerate(peak_freqs):
        print(f"  Mode {i+1}: {round(f, 2)} Hz")

    return peak_freqs, peak_indices


# =============================================================
# STEP 5 — MODE SHAPES
# =============================================================

def extract_mode_shapes(frf_mag, frf_phase, freq_indices, n_nodes=101):
    """
    Extract normalized mode shape at each resonant frequency.

    Uses FRF magnitude only.
    Normalized so that the free end tip (node 100) = +1.
    Fixed end (node 0) forced to 0.

    Inputs:
        frf_mag      : (n_nodes, n_freq) FRF magnitude
        frf_phase    : (n_nodes, n_freq) FRF phase (degrees)
        freq_indices : indices into frequency array for each mode
        n_nodes      : total number of nodes (101)

    Output:
        mode_shapes : list of 1D arrays, each (n_nodes,)
    """
    mode_shapes = []

    for freq_idx in freq_indices:

        # Take raw FRF magnitude at this frequency for all nodes
        shape = frf_mag[:n_nodes, freq_idx].copy()

        # Force fixed end = 0 (cantilever BC)
        shape[0] = 0.0

        # Normalize so tip node (free end) = 1
        tip_val = shape[100]
        if np.abs(tip_val) > 0:
            shape = shape / tip_val
        else:
            # fallback: normalize by max
            max_val = np.max(np.abs(shape))
            if max_val > 0:
                shape = shape / max_val

        mode_shapes.append(shape)

    return mode_shapes


def plot_mode_shapes(mode_shapes, peak_freqs, params,
                     n_nodes=101, save_path=None):
    """
    Plot normalized mode shapes for all identified modes.

    Inputs:
        mode_shapes : list of 1D arrays (one per mode)
        peak_freqs  : list of resonant frequencies (Hz)
        params      : simulation parameter dict
        n_nodes     : number of nodes
        save_path   : optional path to save figure
    """
    n_modes = len(mode_shapes)
    L       = params['length_in']
    x_nodes = np.linspace(0, L, n_nodes)

    n_cols  = 3
    n_rows  = int(np.ceil(n_modes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(14, 4 * n_rows))
    axes = np.array(axes).flatten()

    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, n_modes))

    for i, (shape, freq) in enumerate(zip(mode_shapes, peak_freqs)):
        ax = axes[i]

        ax.fill_between(x_nodes, 0, shape,
                        alpha=0.25, color=colors[i])
        ax.plot(x_nodes, shape,
                color=colors[i], linewidth=2.0)
        ax.scatter(x_nodes[::10], shape[::10],
                   color=colors[i], s=30, zorder=5)

        ax.axvline(x=0, color='black', linewidth=3)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

        ax.set_title(f"Mode {i+1}  —  {round(freq, 2)} Hz",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Position along beam (in)", fontsize=9)
        ax.set_ylabel("Normalized amplitude", fontsize=9)
        ax.set_xlim([0, L])
        ax.set_ylim([-1.3, 1.3])
        ax.grid(True, alpha=0.25)

    for j in range(n_modes, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Mode Shapes — First {n_modes} Modes\n"
        f"{params['material'].title()} | "
        f"L={params['length_in']} in | "
        f"b={params['width_in']} in | "
        f"F0={params['impact_F0_lbf']} lbf",
        fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mode shape plot saved: {save_path}")

    plt.show()
    return fig


# =============================================================
# STEP 6 — FFT WATERFALL (all nodes)
# =============================================================

def plot_waterfall_fft(freqs, fft_mag, params, save_path=None):
    """
    2D heatmap of FFT magnitude across all nodes.

    X-axis : full frequency range, log scale
    Y-axis : beam position (in)
    Color  : acceleration magnitude

    Inputs:
        freqs     : frequency array (Hz)
        fft_mag   : (n_nodes, n_freq) FFT magnitude
        params    : simulation parameter dict
        save_path : optional path to save figure
    """
    # Skip DC bin for log scale
    f_plot   = freqs[1:]
    mag_plot = fft_mag[:, 1:]

    L        = params['length_in']
    n_nodes  = fft_mag.shape[0]
    x_nodes  = np.linspace(0, L, n_nodes)

    fig, ax  = plt.subplots(figsize=(13, 6))

    im = ax.pcolormesh(
        f_plot, x_nodes, mag_plot,
        cmap='inferno',
        shading='auto',
        norm=plt.Normalize(
            vmin=0,
            vmax=np.percentile(mag_plot, 98))
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Acceleration Magnitude (in/s²)", fontsize=10)

    ax.set_xscale('log')
    ax.set_xlim([f_plot[0], f_plot[-1]])
    ax.set_xlabel(
        f"Frequency (Hz) — log scale   "
        f"[{round(f_plot[0], 2)} to {round(f_plot[-1], 1)} Hz]",
        fontsize=11)
    ax.set_ylabel("Position along beam (in)", fontsize=11)
    ax.set_title(
        f"FFT Waterfall — All 101 Nodes\n"
        f"{params['material'].title()} | "
        f"L={params['length_in']} in | "
        f"b={params['width_in']} in | "
        f"F0={params['impact_F0_lbf']} lbf",
        fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Waterfall plot saved: {save_path}")

    plt.show()
    return fig


# =============================================================
# STEP 7 — TIME HISTORY
# =============================================================

def plot_time_history(node_accels, time_vector, params,
                      nodes_to_plot=[0, 24, 49, 74, 99],
                      save_path=None):
    """
    Plot acceleration time history for selected nodes.

    Inputs:
        node_accels   : (n_nodes, n_steps)
        time_vector   : (n_steps,) in seconds
        params        : simulation parameter dict
        nodes_to_plot : list of node indices (0-based)
        save_path     : optional path to save figure
    """
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(nodes_to_plot)))

    fig, axes = plt.subplots(len(nodes_to_plot), 1,
                              figsize=(12, 2.5 * len(nodes_to_plot)),
                              sharex=True)

    if len(nodes_to_plot) == 1:
        axes = [axes]

    for idx, node_idx in enumerate(nodes_to_plot):
        axes[idx].plot(time_vector,
                       node_accels[node_idx, :],
                       color=colors[idx],
                       linewidth=0.7)
        axes[idx].set_ylabel("Accel\n(in/s²)", fontsize=9)
        axes[idx].set_title(f"Node {node_idx + 1}", fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    plt.suptitle(
        f"Acceleration Time History\n"
        f"{params['material'].title()} | "
        f"L={params['length_in']} in | "
        f"b={params['width_in']} in | "
        f"F0={params['impact_F0_lbf']} lbf",
        fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Time history plot saved: {save_path}")

    plt.show()
    return fig


# =============================================================
# MAIN — RUN ALL PLOTS FOR ONE CSV FILE
# =============================================================

def run_visualization(csv_file_path,
                      output_dir=None,
                      n_modes=6,
                      nodes_to_plot=[0, 24, 49, 74, 99]):
    """
    Run complete visualization pipeline for one simulation CSV.

    Scale convention (applied everywhere):
        X-axis frequency : log scale, full range 1 Hz to Nyquist
        Y-axis magnitude : linear scale
        Y-axis phase     : linear scale

    Generates:
        1. Acceleration time history
        2. FFT spectrum
        3. FFT waterfall (all 101 nodes)
        4. FRF magnitude and phase
        5. Mode shapes

    Inputs:
        csv_file_path : path to sim_XXXX.csv
        output_dir    : folder to save plots (None = show only)
        n_modes       : number of modes to extract and plot
        nodes_to_plot : node indices for time/FFT/FRF plots
    """
    print("=" * 55)
    print(" BEAM VIBRATION VISUALIZATION")
    print("=" * 55)

    # --- Load data ---
    params, node_accels, time_vector = load_simulation_csv(csv_file_path)
    dt        = params['dt_s']
    f_nyquist = 1.0 / (2.0 * dt)

    print(f"\n  Frequency range : 1 Hz to {round(f_nyquist, 1)} Hz")
    print(f"  X-axis scale    : log")
    print(f"  Y-axis scale    : linear")

    # --- Save path helper ---
    def save_path(name):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            sim_id = params['sim_id']
            return os.path.join(output_dir,
                                f"sim_{sim_id:04d}_{name}.png")
        return None

    # --- 1. Time history ---
    print("\n[1/5] Plotting time history...")
    plot_time_history(
        node_accels, time_vector, params,
        nodes_to_plot=nodes_to_plot,
        save_path=save_path('time_history'))

    # --- 2. FFT ---
    print("\n[2/5] Computing and plotting FFT...")
    freqs, fft_mag = compute_fft(node_accels, dt)
    plot_fft(
        freqs, fft_mag, params,
        nodes_to_plot=nodes_to_plot,
        save_path=save_path('fft'))

    # --- 3. FFT Waterfall ---
    print("\n[3/5] Plotting FFT waterfall...")
    plot_waterfall_fft(
        freqs, fft_mag, params,
        save_path=save_path('fft_waterfall'))

    # --- 4. FRF ---
    print("\n[4/5] Computing and plotting FRF...")

    # Reconstruct force time history
    tau    = params['impact_tau_s']
    F_time = np.where(
        time_vector <= tau,
        params['impact_F0_lbf'] * np.sin(np.pi * time_vector / tau),
        0.0)

    freqs_frf, frf_mag, frf_phase = compute_frf(
        node_accels, F_time, dt)

    plot_frf(
        freqs_frf, frf_mag, frf_phase, params,
        nodes_to_plot=nodes_to_plot,
        save_path=save_path('frf'))

    # --- 5. Mode shapes from eigenvalue analysis ---
    print("\n[5/5] Computing exact mode shapes via eigenvalue analysis...")
    nat_freqs, mode_shapes_matrix = compute_mode_shapes_from_matrices(
        params, n_modes=n_modes)

    # Convert to list of arrays for plot_mode_shapes
    mode_shapes_list = [mode_shapes_matrix[:, i]
                        for i in range(n_modes)]

    plot_mode_shapes(
        mode_shapes_list, nat_freqs, params,
        n_nodes=params['n_nodes'],
        save_path=save_path('mode_shapes'))

    print("\n" + "=" * 55)
    print(" VISUALIZATION COMPLETE")
    print("=" * 55)
    if output_dir:
        print(f" Plots saved to: {os.path.abspath(output_dir)}")