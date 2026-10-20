import pybamm

# ==============================================================================
# 1. GEOMETRY & ENVIRONMENTAL CONSTANTS
# ==============================================================================
W = 0.047           # Electrode width [m]
H = 0.051           # Electrode height [m]
TAB_W = 0.01        # Tab width [m]
TEMP_AMBIENT = 298.15 # Default ambient temperature [K]

# ==============================================================================
# 2. TAB COORDINATES OF THE REFERENCE CELL (baseline validation, Appendix B)
# ==============================================================================
TOPOLOGIES = {
    "NT": { # Normal-Tab: both tabs on the top edge
        "Negative tab centre y-coordinate [m]": W * 0.2, "Negative tab centre z-coordinate [m]": H,
        "Positive tab centre y-coordinate [m]": W * 0.8, "Positive tab centre z-coordinate [m]": H,
    },
}

# ==============================================================================
# 3. MESH OF THE 1D VALIDATION MODEL
# ==============================================================================
MESH_1D = {'x_n': 20, 'x_s': 20, 'x_p': 20, 'r_n': 10, 'r_p': 10}

# ==============================================================================
# 4. SHARED PARAMETER BUILDER
# ==============================================================================
def get_base_parameters(L_n=None, L_p=None):
    """
    Loads the baseline Ai2020 parameters and injects our geometric
    constants and custom electrode thicknesses.
    """
    param = pybamm.ParameterValues("Ai2020")

    # Inject our SSOT geometries
    defaults = {
        "Ambient temperature [K]": TEMP_AMBIENT,
        "Electrode width [m]": W,
        "Electrode height [m]": H,
        "Negative tab width [m]": TAB_W,
        "Positive tab width [m]": TAB_W,
    }

    # Inject optimized thicknesses if provided
    if L_n is not None:
        defaults["Negative electrode thickness [m]"] = L_n
    if L_p is not None:
        defaults["Positive electrode thickness [m]"] = L_p

    param.update(defaults, check_already_exists=False)
    return param
