import matplotlib.pyplot as plt
import numpy as np

from scripts.ridge_data import (
    elements_ext_ree,
    grid_chem_integ_xy,
    grid_rate_integ_xy,
    n_morb_comp,
    prim_mantle_comp,
)
from src.Duvernay_et_al_2023.ChemistryData import DM_WH_2005

fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 5.5))

ax.grid(which="both")

ax.set_xticks(range(len(elements_ext_ree)), labels=elements_ext_ree)

ax.set_ylim(0.1, 30)
ax.set_yscale("log")
ax.set_ylabel("Normalized to Primitive Mantle")

n_morb = np.asarray([n_morb_comp["gale_2013"][ele] for ele in elements_ext_ree])
ms_1995 = np.asarray(
    [prim_mantle_comp["mcdonough_1995"][ele] for ele in elements_ext_ree]
)
wh_2005 = np.asarray([DM_WH_2005[ele] for ele in elements_ext_ree])
parental_magma = np.asarray(
    [
        grid_chem_integ_xy["d9_wh"][ele] / grid_rate_integ_xy["d9_wh"]
        for ele in elements_ext_ree
    ]
)

ax.plot(elements_ext_ree, n_morb / ms_1995, label="N-MORB (Gale et al., 2013)")
ax.plot(elements_ext_ree, wh_2005 / ms_1995, label="DMM (Workman and Hart, 2005)")
ax.plot(
    elements_ext_ree,
    (0.9 * wh_2005 + 0.1 * ms_1995) / ms_1995,
    label="90% DMM (Workman and Hart, 2005)\n10% PM (McDonough and Sun, 1995)",
)
ax.plot(
    elements_ext_ree,
    parental_magma / ms_1995,
    label="Parental magma (Duvernay et al., 2024)",
)

ax.legend()

plt.show()
