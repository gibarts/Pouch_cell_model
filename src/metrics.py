"""Cell mass used for the specific-energy objective (OF1)."""

# Constants for mass calculation (Typical values if not in param)
RHO_PCC = 2700.0  # Aluminum [kg/m^3]
RHO_NCC = 8960.0  # Copper [kg/m^3]
RHO_L   = 1200.0  # Electrolyte [kg/m^3]
RHO_S   = 940.0   # Separator (Polyethylene) [kg/m^3]

def calculate_cell_mass(param):
    """
    Calculates the total mass of the cell in kg based on geometric
    and material properties from the PyBaMM parameter set.
    """
    # Geometric and material properties
    A = float(param["Electrode width [m]"]) * float(param["Electrode height [m]"])
    n = float(param.get("Number of electrodes connected in parallel to make a cell", 1.0))

    # Thicknesses
    L_p = float(param["Positive electrode thickness [m]"])
    L_n = float(param["Negative electrode thickness [m]"])
    L_s = float(param["Separator thickness [m]"])
    L_pcc = float(param["Positive current collector thickness [m]"])
    L_ncc = float(param["Negative current collector thickness [m]"])

    # Porosities (Liquid fractions)
    eps_pl = float(param["Positive electrode porosity"])
    eps_nl = float(param["Negative electrode porosity"])
    eps_sl = float(param["Separator porosity"])

    # Active material fractions (Solid fractions)
    eps_ps = 1.0 - eps_pl
    eps_ns = 1.0 - eps_nl

    # Component masses
    mass_pcc = L_pcc * RHO_PCC
    mass_ncc = L_ncc * RHO_NCC
    mass_p_solid = L_p * eps_ps * float(param["Positive electrode density [kg.m-3]"])
    mass_p_liquid = L_p * eps_pl * RHO_L
    mass_n_solid = L_n * eps_ns * float(param["Negative electrode density [kg.m-3]"])
    mass_n_liquid = L_n * eps_nl * RHO_L
    mass_s_liquid = L_s * eps_sl * RHO_L
    mass_s_solid = L_s * RHO_S

    # Total mass equation
    total_areal_mass = (mass_pcc + mass_ncc + mass_p_solid + mass_p_liquid +
                        mass_n_solid + mass_n_liquid + mass_s_liquid + mass_s_solid)

    mass_kg = total_areal_mass * A * n
    return mass_kg
