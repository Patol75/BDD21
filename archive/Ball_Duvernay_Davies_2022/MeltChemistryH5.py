#!/usr/bin/env python3
from pathlib import Path

import h5py
import numpy as np
from constants import (
    T_mantle,
    adiab_grad,
    attrib,
    cs_0,
    depth_lab,
    domain_dim,
    rho_mantle,
)
from MeltChemistryFunctions import (
    calc_X_spl_in_gnt_out,
    non_zero_initial_melt,
    run_integrator,
)
from numba import float64
from numba.typed import Dict
from numba.types import unicode_type
from scipy.constants import g
from tqdm.contrib.concurrent import process_map


# Extract particle attributes relative to the melting path and calculate
# corresponding mineralogy and chemistry
def main(part_id, part_proc_id):
    # Initialise lists to hold relevant particle attributes
    depth, melt_fraction, melt_rate, pressure, temperature = [], [], [], [], []
    # Determine timesteps over which calculation is to be performed
    timesteps = [0, len(h5f.keys())] if tstep_range is None else tstep_range
    # Extract and store particle attributes iteratively
    for timestep in range(*timesteps):
        # Identify particle
        index = np.asarray(
            (h5f[f"Step#{timestep}"]["id"][()] == part_id)
            & (h5f[f"Step#{timestep}"]["proc_id"][()] == part_proc_id)
        ).nonzero()[0][0]
        # Extract values
        depth.append(domain_dim[1] - h5f[f"Step#{timestep}"]["y"][()][index])
        melt_fraction.append(h5f[f"Step#{timestep}"][f"{attrib}1"][()][index])
        melt_rate.append(h5f[f"Step#{timestep}"][f"{attrib}3"][()][index])
        pressure.append(h5f[f"Step#{timestep}"][f"{attrib}4"][()][index])
        temperature.append(h5f[f"Step#{timestep}"][f"{attrib}5"][()][index])

    # If melting did not occur, return
    if max(melt_rate) == 0:
        return

    # Identify entries where melting occurs
    mask = [x > 0 for x in melt_rate]
    # Include timestep before melting started
    if melt_fraction[0] == 0:
        mask[mask.index(True) - 1] = True
    elif DISCARD_NON_ZERO_INITIAL_MELT:
        return
    elif melt_rate[0] == 0:
        mask[0] = True

    # Define the dictionary of arrays that describe the melting path
    part_arr = Dict.empty(key_type=unicode_type, value_type=float64[:])
    for attr_arr in ["melt_fraction", "pressure", "temperature"]:
        part_arr[attr_arr] = np.array(locals()[attr_arr])[mask]
    # If non-zero initial melts are allowed, update the dictionary of arrays
    if part_arr["melt_fraction"][0] > 0:
        part_arr = non_zero_initial_melt(
            part_arr,
            temperature[-1] - adiab_grad * depth[-1],
            T_mantle,
            rho_mantle * g * depth_lab,
            adiab_grad / rho_mantle / g * 1e9,
        )

    # Determine where spinel crystallises and garnet exhausts along the path
    depth = np.array(depth)[mask]
    X_spl_in, X_gnt_out = calc_X_spl_in_gnt_out(
        1,
        1,
        depth[0],
        depth[-1],
        part_arr,
        adiab_grad / rho_mantle / g * 1e9,
        rho_mantle,
    )

    # Initialise variables required during integration
    ode_dX = 1e-6

    # Calculate mineralogy and chemistry (pick one of the two below strategies)

    # Single integration over the whole melting path, yielding final values
    ode_dX, Fn, cs, cl = run_integrator(part_arr, ode_dX, X_spl_in, X_gnt_out, cs_0)

    # # Step-by-step integration with possibility to save values along the path
    # cs = cs_0
    # for i in range(sum(mask) - 1):
    #     # Restrict dictionary of arrays to relevant values for the step
    #     for attr_arr in ["melt_fraction", "pressure", "temperature"]:
    #         part_arr[attr_arr] = np.array(locals()[attr_arr])[mask][i:i+2]
    #     # Calculate mineralogy and chemistry
    #     ode_dX, Fn, cs, cl = run_integrator(
    #         part_arr, ode_dX, X_spl_in, X_gnt_out, cs)
    #     # Interrupt integration loop if all elements are exhausted
    #     if np.count_nonzero(cs) == 0:
    #         break

    return part_id, part_proc_id, [timesteps, mask, cs, cl]


# Range of timesteps to process; either a 2-tuple (upper-bound not included) or
# None (all steps processed)
tstep_range = None
# Set to True to discard particles whose melting rate is zero at the final
# time-step considered; useful for steady-state simulations
ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END = True
# Set to True to discard particles initialised above the solidus; useful for
# systems where the melting region depth extent does not change
DISCARD_NON_ZERO_INITIAL_MELT = True

model_path = Path("/path/to/model")

compositions = {}
h5part = "particles.h5part"
with h5py.File(model_path / h5part, "r") as h5f:
    last_step = tstep_range[1] - 1 if tstep_range else len(h5f.keys()) - 1
    if ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END:
        mask = h5f[f"Step#{last_step}"][f"{attrib}3"][()] > 0
    else:
        mask = h5f[f"Step#{last_step}"][f"{attrib}2"][()] > 0
    h5_ids = h5f[f"Step#{last_step}"]["id"][()][mask]
    h5_proc_ids = h5f[f"Step#{last_step}"]["proc_id"][()][mask]
    if __name__ == "__main__":
        for output in process_map(
            main, h5_ids, h5_proc_ids, max_workers=16, chunksize=1
        ):
            if output is not None:
                compositions[f"{output[1]}_{output[0]}"] = output[2]
np.savez(model_path / "RESULTS", compositions=compositions)
