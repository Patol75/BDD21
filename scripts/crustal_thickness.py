from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from sys import modules

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
from scipy.integrate import simpson
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree

# model_path = Path("/media/thomas/LaCie/Paddy/CoP_1598_NEW")
# model_path = Path("/mnt/cosgrove/home/thomas/BDD21_Models")
# source_path = Path("/mnt/gadi/scratch/xd2/td5646/Simulations/bdd_update")
source_path = Path("/mnt/gadi/scratch/xd2/td5646/Simulations/ridge_morb_chem_paper")
# source_path = Path("/home/thomas/Videos")
model_path = source_path / "ridge_t1623_v6_w100_d10_wh_restart_2"

spec = spec_from_file_location("constants", model_path / "constants.py")
constants = module_from_spec(spec)
modules["constants"] = constants
spec.loader.exec_module(constants)
attrib, spreading_rate = constants.attrib, constants.vel_x * 2
# attrib = "katz_mckenzie_bdd21_"
# spreading_rate = 2.1 / 100 / 365.25 / 8.64e4 * 2

h5file = "ridge_restart_2.particles.melt_fraction_and_chemistry.h5part"
# h5file = "Ridge_1598.h5part"
with File(model_path / h5file, "r") as h5f:
    step = 20
    h5x, h5y = h5f[f"Step#{step}"]["x"][()], h5f[f"Step#{step}"]["y"][()]
    h5rate = h5f[f"Step#{step}"][f"{attrib}3"][()]
h5coords = np.column_stack((h5x, h5y))

# gridY, gridX = np.mgrid[5.6e5:6.6e5:201j, 6.9e5:1.29e6:601j]
# gridY, gridX = np.mgrid[5.6e5:6.6e5:201j, 2.9e5:1.69e6:1401j]
gridY, gridX = np.mgrid[5.6e5:6.6e5:201j, 9.015e5:1.0785e6:178j]
# gridY, gridX = np.mgrid[1e5:2e5:201j, 5e3:5.95e5:591j]
# gridY, gridX = np.mgrid[1e5:2e5:201j, 2.325e5:3.675e5:136j]
gridCoords = np.column_stack((gridX.flatten(), gridY.flatten()))
iRate = RBFInterpolator(h5coords, h5rate, neighbors=32, kernel="thin_plate_spline")
gridRate = iRate(gridCoords).reshape(gridX.shape)
print(gridRate.max())
h5tree = KDTree(h5coords)
nbNeighbours = h5tree.query_ball_point(gridCoords, 1e3, return_length=True).reshape(
    gridY.shape
)
gridRate[nbNeighbours == 0] = 0

fig, (ax, bx) = plt.subplots(nrows=2, constrained_layout=True, sharex=True, sharey=True)
ctrf = ax.contourf(
    gridX / 1e3,
    gridY / 1e3,
    gridRate,
    extend="both",
    levels=np.logspace(-5, -1, 21),
    norm=LogNorm(),
)
for iso in ctrf.collections:
    iso.set_edgecolor("face")
ax.set_aspect("equal")

tri = Triangulation(h5x / 1e3, h5y / 1e3)
tctrf = bx.tricontourf(
    tri,
    np.clip(h5rate, 1e-9, None),
    extend="both",
    levels=np.logspace(-5, -1, 21),
    norm=LogNorm(),
)
for iso in tctrf.collections:
    iso.set_edgecolor("face")
bx.set_aspect("equal")
plt.show()

iMeltIntY = simpson(gridRate, x=gridX[0] / 1e3)
iMeltIntXY = simpson(iMeltIntY, x=gridY[:, 0] / 1e3)
print(3300 / 2900 * iMeltIntXY / spreading_rate * 1e3 / 8.64e4 / 365.25 / 1e6)
