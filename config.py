# config.py
# =============================================================
# Central configuration file for Euler-Bernoulli Beam FEA
# All units in IPS system: inches, lbf, seconds
# =============================================================

# -------------------------------------------------------------
# UNIT SYSTEM NOTE
# Length   : inches (in)
# Force    : pound-force (lbf)
# Time     : seconds (s)
# Pressure : psi (lbf/in²)
# Mass     : lbf·s²/in  (consistent mass unit)
# -------------------------------------------------------------

# Gravitational constant (for density unit conversion)
G_C = 386.088  # in/s²

# -------------------------------------------------------------
# GEOMETRY
# -------------------------------------------------------------
THICKNESS = 0.5  # t, constant beam thickness (in)

LENGTH_VALUES = [20, 40, 60, 80, 100, 120, 140, 160]  # L (in)
WIDTH_VALUES  = [0.1, 0.3, 0.5, 0.7, 0.9]             # b (in)

# -------------------------------------------------------------
# IMPACT FORCE
# -------------------------------------------------------------
FORCE_VALUES   = [5, 20, 45, 60, 75, 90, 150, 300, 500]  # F0 (lbf)
PULSE_DURATION = 0.001  # tau, impact duration (s) — 1ms pulse

# -------------------------------------------------------------
# TIME SETTINGS
# Matched to Photron FASTCAM SA5 at 2000 fps
# dt = 0.0005 s, f_max captured = 1000 Hz
# Resolution at 2000 fps: 1024 x 500 pixels
# -------------------------------------------------------------
T_TOTAL = 1.0   # total simulation time (s)
N_STEPS = 2000  # number of timesteps (matches SA5 at 2000 fps)
DT      = T_TOTAL / (N_STEPS - 1)  # 0.0005005 s per step

# -------------------------------------------------------------
# MESH
# -------------------------------------------------------------
N_ELEMENTS = 100  # number of beam elements
N_NODES    = 101  # number of nodes (N_ELEMENTS + 1)

# -------------------------------------------------------------
# RAYLEIGH DAMPING
# -------------------------------------------------------------
RAYLEIGH_ALPHA = 0.01     # mass proportional damping coefficient
RAYLEIGH_BETA  = 0.00001  # stiffness proportional damping coefficient

# -------------------------------------------------------------
# NEWMARK-BETA PARAMETERS
# -------------------------------------------------------------
NEWMARK_BETA  = 0.25  # average acceleration (unconditionally stable)
NEWMARK_GAMMA = 0.50

# -------------------------------------------------------------
# BOUNDARY CONDITION
# -------------------------------------------------------------
# Supported types: 'cantilever', 'simply_supported', 'fixed_fixed'
# Only cantilever is implemented now. Others can be added later.
BOUNDARY_CONDITION = 'cantilever'

# -------------------------------------------------------------
# SIMULATION SETTINGS
# -------------------------------------------------------------
N_SIMULATIONS = 500
RANDOM_SEED   = 42

# -------------------------------------------------------------
# MATERIALS AVAILABLE
# -------------------------------------------------------------
MATERIAL_NAMES = ['steel', 'aluminum']