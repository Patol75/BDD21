#!/usr/bin/env python3
import h5py
from numba import float64
from numba.types import unicode_type
from numba.typed import Dict
import numpy as np
from pathlib import Path
from scipy.constants import g
from tqdm.contrib.concurrent import process_map

from MeltChemistryFunctions import (calc_X_spl_in_gnt_out,
                                    non_zero_initial_melt, run_integrator)
from constants import (adiab_grad, attrib, cs_0, domain_dim,
                       rho_mantle, T_mantle)


# Extract particle attributes relative to the melting path and initiate
# concentration calculations
def main(part_id, part_proc_id):
    depth, melt_fraction, melt_rate, pressure, temperature = [], [], [], [], []
    timesteps = [0, len(h5f.keys())] if tstep_range is None else tstep_range
    for timestep in range(*timesteps):  # Extract and store particle attributes
        index = np.asarray((h5f[f'Step#{timestep}']['id'][()] == part_id)
                           & (h5f[f'Step#{timestep}']['proc_id'][()]
                              == part_proc_id)).nonzero()[0][0]
        depth.append(domain_dim[1] - h5f[f'Step#{timestep}']['y'][()][index])
        melt_fraction.append(h5f[f'Step#{timestep}'][f'{attrib}1'][()][index])
        melt_rate.append(h5f[f'Step#{timestep}'][f'{attrib}3'][()][index])
        pressure.append(h5f[f'Step#{timestep}'][f'{attrib}4'][()][index])
        temperature.append(h5f[f'Step#{timestep}'][f'{attrib}5'][()][index])

    if max(melt_rate) == 0:  # Never melted
        return
    mask = [x > 0 for x in melt_rate]  # Timesteps when melting happened
    if melt_fraction[0] == 0:  # Include timestep before melting started
        mask[mask.index(True) - 1] = True
    elif DISCARD_NON_ZERO_INITIAL_MELT:
        return
    elif melt_rate[0] == 0:
        mask[0] = True

    part_arr = Dict.empty(key_type=unicode_type, value_type=float64[:])
    for attr_arr in ["melt_fraction", "pressure", "temperature"]:
        part_arr[attr_arr] = np.array(locals()[attr_arr])[mask]
    if part_arr["melt_fraction"][0] > 0:
        part_arr = non_zero_initial_melt(part_arr, T_mantle)
    depth = np.array(depth)[mask]

    X_spl_in, X_gnt_out = calc_X_spl_in_gnt_out(
        1, 1, np.inf, depth[-1], part_arr, adiab_grad / rho_mantle / g * 1e9)

    # mask_ode, ode_dX, X_0, P_0, Fn, cs, cl = run_integrator(
    #     part_arr, np.ones(cs_0.size, dtype=bool), 1e-6,
    #     0., part_arr["pressure"][0], cs_0, X_spl_in, X_gnt_out)

    mask_ode = np.ones(cs_0.size, dtype='bool')
    ode_dX, X_0, P_0, cs = 1e-6, 0., part_arr["pressure"][0], cs_0
    for i in range(sum(mask) - 1):
        for attr_arr in ["melt_fraction", "pressure", "temperature"]:
            part_arr[attr_arr] = np.array(locals()[attr_arr])[mask][i:i+2]
        mask_ode, ode_dX, X_0, P_0, Fn, cs, cl = run_integrator(
            part_arr, mask_ode, ode_dX, X_0, P_0, cs, X_spl_in, X_gnt_out)
        print(cs[14], cl[14])

    return part_id, part_proc_id, [timesteps, mask, cs, cl]


# Range of timesteps to process; either a 2-tuple (upper-bound not included) or
# None (all steps processed)
tstep_range = None
# Set to True to discard particles whose melting rate is zero at the final
# time-step considered; useful for ridge simulations
ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END = True
# Set to True to discard particles initialised above the solidus; useful for
# systems where the melting region depth extent does not change
DISCARD_NON_ZERO_INITIAL_MELT = True

model_path = Path("/home/thomas/Documents/PaddyCode/ChemistryOnParticles")

compositions = {}
h5part = "standard_case.particles.melt_fraction_and_chemistry.h5part"
with h5py.File(model_path / h5part, "r") as h5f:
    last_step = tstep_range[1] - 1 if tstep_range else len(h5f.keys()) - 1
    if ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END:
        mask = h5f[f"Step#{last_step}"][f'{attrib}3'][()] > 0
    else:
        mask = h5f[f"Step#{last_step}"][f'{attrib}2'][()] > 0
    h5_ids = h5f[f"Step#{last_step}"]["id"][()][mask]
    h5_proc_ids = h5f[f"Step#{last_step}"]["proc_id"][()][mask]
    index = np.nonzero((h5_ids == 798322) & (h5_proc_ids == 16))[0][0]
    main(h5_ids[index], h5_proc_ids[index])
    print('\n')
    index = np.nonzero((h5_ids == 35380) & (h5_proc_ids == 16))[0][0]
    main(h5_ids[index], h5_proc_ids[index])
    assert False
    if __name__ == "__main__":
        for output in process_map(main, h5_ids, h5_proc_ids,
                                  max_workers=16, chunksize=1):
            if output is not None:
                compositions[f"{output[1]}_{output[0]}"] = output[2]
np.savez(model_path / "test", compositions=compositions)

# 1548 K        |  3.27 | 3.62 |  9.32
# xf =  55.0 km |  2.97 | 3.55 | 10.03
# xf =  61.5 km |  3.05 | 3.58 |  9.90
# ------------------------------------
# 1598 K        |  6.75 | 2.66 |  8.70
# xf =  57.0 km |  5.75 | 2.50 |  9.09
# xf =  77.5 km |  6.23 | 2.60 |  9.03
# ------------------------------------
# 1598 K PM     |  6.05 | 2.04 |  7.67
# xf =  57.0 km |  5.15 | 1.97 |  8.36
# xf =  78.0 km |  5.58 | 2.02 |  8.08
# ------------------------------------
# 1598 K 59-60  |  6.05 | 2.55 |  9.04
# xf =  57.0 km |  5.15 | 2.41 |  9.28
# xf =  78.0 km |  5.58 | 2.51 |  9.26
# ------------------------------------
# 1598 K 79-80  |  6.05 | 2.56 |  8.30
# xf =  57.0 km |  5.15 | 2.41 |  8.97
# xf =  78.0 km |  5.58 | 2.51 |  8.81
# ------------------------------------
# 1648 K        |  9.15 | 2.15 |  8.78
# xf =  61.0 km |  7.29 | 1.95 |  8.79
# xf =  93.0 km |  8.29 | 2.08 |  8.86
# ------------------------------------
# 1698 K        | 13.51 | 1.72 |  8.74
# xf =  58.0 km | 10.01 | 1.49 |  8.49
# xf = 106.0 km | 12.22 | 1.65 |  8.66
# ------------------------------------
# 0.5 cm/yr     |  3.46 | 3.59 |  9.00
# xf =  36.0 km |  3.22 | 3.66 |  9.33
# xf =  61.0 km |  3.46 | 3.60 |  8.99
# ------------------------------------
# 5 cm/yr       |  6.49 | 2.46 |  8.61
# xf =  89.0 km |  5.44 | 2.28 |  8.93
# xf =  70.0 km |  5.05 | 2.19 |  8.97
# ------------------------------------
# 10 cm/yr      |  6.92 | 2.35 |  8.59
# xf = 123.0 km |  5.76 | 2.14 |  8.86
# xf =  55.0 km |  4.27 | 1.87 |  8.91
