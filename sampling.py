# sampling.py
# =============================================================
# Parameter set generator for 500 beam simulations
# Produces randomized combinations of beam parameters
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
import random
from itertools import product
from materials import get_material
from config import (
    LENGTH_VALUES, WIDTH_VALUES, FORCE_VALUES,
    MATERIAL_NAMES, THICKNESS, PULSE_DURATION,
    RAYLEIGH_ALPHA, RAYLEIGH_BETA,
    T_TOTAL, N_STEPS, DT,
    N_ELEMENTS, N_NODES,
    RANDOM_SEED, N_SIMULATIONS
)

# -------------------------------------------------------------
# TWO SAMPLING MODES
# Mode 1 (discrete) : build all combinations from candidate
#                     lists, shuffle, take first N
# Mode 2 (continuous): uniform random draws from ranges
# Default            : Mode 1 if candidate lists provided
# -------------------------------------------------------------

def generate_parameter_sets(
        length_values  = LENGTH_VALUES,
        width_values   = WIDTH_VALUES,
        force_values   = FORCE_VALUES,
        material_names = MATERIAL_NAMES,
        n_simulations  = N_SIMULATIONS,
        seed           = RANDOM_SEED,
        mode           = 1):
    """
    Generate n_simulations parameter sets.

    Mode 1 — Discrete permutation (default):
        Builds all combinations of L x b x F0 x material,
        shuffles using seed, takes first n_simulations.

    Mode 2 — Continuous random:
        Draws uniformly from ranges for L, b, F0,
        randomly picks material.

    Inputs:
        length_values  : list of candidate lengths (in)
        width_values   : list of candidate widths (in)
        force_values   : list of candidate forces (lbf)
        material_names : list of material name strings
        n_simulations  : number of parameter sets to generate
        seed           : random seed for reproducibility
        mode           : 1 = discrete, 2 = continuous

    Output:
        param_sets : list of dicts, each containing
                     all parameters for one simulation
    """
    random.seed(seed)
    np.random.seed(seed)

    if mode == 1:
        param_sets = _discrete_mode(
            length_values, width_values,
            force_values, material_names,
            n_simulations, seed)
    else:
        param_sets = _continuous_mode(
            length_values, width_values,
            force_values, material_names,
            n_simulations, seed)

    return param_sets


def _discrete_mode(length_values, width_values, force_values,
                   material_names, n_simulations, seed):
    """
    Build all combinations, shuffle, take first n_simulations.
    """
    # All possible combinations
    all_combos = list(product(
        length_values,
        width_values,
        force_values,
        material_names
    ))

    total_combos = len(all_combos)
    print(f"Total possible combinations : {total_combos}")
    print(f"Simulations requested       : {n_simulations}")

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(all_combos)

    # If not enough unique combos, allow repeats with warning
    if total_combos < n_simulations:
        print(f"⚠  Only {total_combos} unique combos available.")
        print(f"   Repeating combinations to reach {n_simulations}.")
        while len(all_combos) < n_simulations:
            all_combos.extend(all_combos)

    # Take first n_simulations
    selected = all_combos[:n_simulations]

    # Build parameter dicts
    param_sets = []
    for sim_id, (L, b, F0, mat_name) in enumerate(selected):
        mat   = get_material(mat_name)
        entry = _build_param_dict(sim_id, L, b, F0, mat_name, mat)
        param_sets.append(entry)

    return param_sets


def _continuous_mode(length_values, width_values, force_values,
                     material_names, n_simulations, seed):
    """
    Draw uniformly from ranges for L, b, F0, random material.
    """
    np.random.seed(seed)

    L_min,  L_max  = min(length_values),  max(length_values)
    b_min,  b_max  = min(width_values),   max(width_values)
    F0_min, F0_max = min(force_values),   max(force_values)

    param_sets = []
    for sim_id in range(n_simulations):
        L        = np.random.uniform(L_min, L_max)
        b        = np.random.uniform(b_min, b_max)
        F0       = np.random.uniform(F0_min, F0_max)
        mat_name = np.random.choice(material_names)
        mat      = get_material(mat_name)
        entry    = _build_param_dict(sim_id, L, b, F0, mat_name, mat)
        param_sets.append(entry)

    return param_sets


def _build_param_dict(sim_id, L, b, F0, mat_name, mat):
    """
    Build a single simulation parameter dictionary.
    Contains all parameters needed for one simulation run.
    """
    return {
        # Simulation ID
        'sim_id'        : sim_id,

        # Material
        'material'      : mat_name,
        'E_psi'         : mat['E'],
        'rho_lbm_in3'   : mat['rho_lbm'],
        'rho_consistent': mat['rho'],

        # Geometry
        'length_in'     : L,
        'width_in'      : b,
        'thickness_in'  : THICKNESS,

        # Impact
        'impact_F0_lbf' : F0,
        'impact_tau_s'  : PULSE_DURATION,

        # Damping
        'rayleigh_alpha': RAYLEIGH_ALPHA,
        'rayleigh_beta' : RAYLEIGH_BETA,

        # Time
        'dt_s'          : DT,
        'T_s'           : T_TOTAL,
        'n_steps'       : N_STEPS,

        # Mesh
        'n_elements'    : N_ELEMENTS,
        'n_nodes'       : N_NODES,
    }