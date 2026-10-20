"""1D reference-cell model used for the baseline validation (Appendix B)."""
import pybamm
from src.config import get_base_parameters, TOPOLOGIES, MESH_1D

def run_dfn_swelling_simulation(C_rate, temp_ambient=298.15, L_n=None, L_p=None):
    """Runs a 1D DFN model with mechanics enabled."""
    options = {
        "particle": "Fickian diffusion",
        "cell geometry": "pouch",
        "thermal": "x-lumped",
        "current collector": "potential pair",
        "dimensionality": 1,
        "surface form": "algebraic",
        "particle mechanics": "swelling only",
    }
    model = pybamm.lithium_ion.DFN(options=options)

    param = get_base_parameters(L_n=L_n, L_p=L_p)
    param.update({"Ambient temperature [K]": temp_ambient}, check_already_exists=False)

    # Patch missing thermal parameters
    param.update({
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 1.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 1.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0.0,
        # "potential pair" current collector still needs tab z-coordinates
        **TOPOLOGIES["NT"],
    }, check_already_exists=False)

    n_stacks = param["Number of electrodes connected in parallel to make a cell"]
    effective_C_rate = C_rate / n_stacks

    experiment = pybamm.Experiment(
        [(f"Discharge at {effective_C_rate}C until 3.0V")],
        period="10 seconds"
    )

    # 1D "potential pair" current collector needs its own mesh point count
    var_pts_1d = dict(MESH_1D, z=10)

    solver = pybamm.CasadiSolver()
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment, var_pts=var_pts_1d, solver=solver)
    sol = sim.solve()

    return sol, param
