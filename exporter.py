# exporter.py
# =============================================================
# CSV export for beam FEA simulation results
# One CSV file per simulation
# Each CSV: 1 row of metadata + 101 node columns
# =============================================================

import numpy as np
import pandas as pd
import os
from sensors import get_node_labels, serialize_node_accel
from config import N_NODES, N_STEPS, DT, T_TOTAL

def export_single_simulation(result, output_dir='simulation_results',
                              encoding='A', delimiter=';'):
    """
    Export one simulation result to its own CSV file.

    File naming: sim_0001.csv, sim_0002.csv, ...

    CSV layout:
        First columns  : all beam parameters
        Then           : node_001_accel ... node_101_accel
        Each node col  : 2000 acceleration values as
                         delimited string (Encoding A)

    Inputs:
        result     : single result dict from batch_runner
        output_dir : folder to save CSV files
        encoding   : 'A' serialized string (default)
                     'B' one column per timestep
        delimiter  : delimiter for Encoding A (default ';')

    Output:
        file_path  : path of saved CSV file
    """
    if result['status'] != 'success':
        return None

    params      = result['params']
    node_accels = result['node_accels']   # shape (101, 2000)
    sim_id      = params['sim_id']

    # --- Build metadata row ---
    row = {
        'sim_id'        : sim_id,
        'material'      : params['material'],
        'E_psi'         : params['E_psi'],
        'rho_lbm_in3'   : params['rho_lbm_in3'],
        'rho_consistent': params['rho_consistent'],
        'length_in'     : params['length_in'],
        'width_in'      : params['width_in'],
        'thickness_in'  : params['thickness_in'],
        'impact_F0_lbf' : params['impact_F0_lbf'],
        'impact_tau_s'  : params['impact_tau_s'],
        'rayleigh_alpha': params['rayleigh_alpha'],
        'rayleigh_beta' : params['rayleigh_beta'],
        'dt_s'          : params['dt_s'],
        'T_s'           : params['T_s'],
        'n_steps'       : params['n_steps'],
        'n_elements'    : params['n_elements'],
        'n_nodes'       : params['n_nodes'],
    }

    # --- Add node acceleration columns ---
    if encoding == 'A':
        # Encoding A: one serialized string per node
        labels = get_node_labels(N_NODES)
        for i, label in enumerate(labels):
            row[label] = serialize_node_accel(
                node_accels[i, :], delimiter)

    elif encoding == 'B':
        # Encoding B: one column per node per timestep
        for i in range(N_NODES):
            for t in range(N_STEPS):
                col_name = f"node_{i+1:03d}_t{t:04d}"
                row[col_name] = node_accels[i, t]

    else:
        raise ValueError(f"Unknown encoding '{encoding}'. Use 'A' or 'B'.")

    # --- Build single row DataFrame ---
    df = pd.DataFrame([row])

    # --- Save to individual CSV file ---
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"sim_{sim_id:04d}.csv"
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)

    return file_path


def export_all_simulations(results, output_dir='simulation_results',
                           encoding='A', delimiter=';'):
    """
    Export all simulation results to individual CSV files.

    Inputs:
        results    : list of result dicts from batch_runner
        output_dir : folder to save all CSV files
        encoding   : 'A' or 'B'
        delimiter  : delimiter for Encoding A

    Output:
        file_paths  : list of saved file paths
        n_exported  : number of files saved
        n_skipped   : number of failed sims skipped
    """
    print(f"Exporting simulations to folder: '{output_dir}'")
    print(f"Encoding : {encoding}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    file_paths = []
    n_exported = 0
    n_skipped  = 0

    for r in results:
        if r['status'] != 'success':
            n_skipped += 1
            continue

        file_path = export_single_simulation(
            r, output_dir, encoding, delimiter)

        if file_path:
            file_paths.append(file_path)
            n_exported += 1

    print(f"Files exported : {n_exported}")
    print(f"Files skipped  : {n_skipped} (failed simulations)")
    print(f"Output folder  : {os.path.abspath(output_dir)}")

    # Print size of first file as reference
    if file_paths:
        size_mb = os.path.getsize(file_paths[0]) / 1e6
        print(f"Size per file  : ~{round(size_mb, 2)} MB")
        print(f"Total est size : ~{round(size_mb * n_exported, 1)} MB")

    return file_paths, n_exported, n_skipped


def export_time_vector(output_dir='simulation_results',
                       output_path='time_vector.csv'):
    """
    Export shared time vector to a separate CSV file.
    Saved in the same output folder as simulation CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    time_vector = np.linspace(0, DT * (N_STEPS - 1), N_STEPS)
    df_time     = pd.DataFrame({'time_s': time_vector})

    full_path   = os.path.join(output_dir, output_path)
    df_time.to_csv(full_path, index=False)

    print(f"Time vector saved : {full_path}")
    print(f"  Steps   : {len(time_vector)}")
    print(f"  t_start : {time_vector[0]} s")
    print(f"  t_end   : {round(time_vector[-1], 4)} s")
    print(f"  dt      : {round(time_vector[1]-time_vector[0], 6)} s")

    return df_time