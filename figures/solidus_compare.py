import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)

ax.set_xlabel(r"Temperature ($^\circ$C)", fontsize=16, fontweight="semibold")
ax.set_ylabel("Pressure (GPa)", fontsize=16, fontweight="semibold")
ax.tick_params(labelsize=16)
ax.grid()

pressure = np.linspace(0, 6, 301)

ax.plot(
    1107.1 + 135.7 * pressure + -5.3 * pressure**2,
    pressure,
    linewidth=3,
    label="Hirschmann (2000), Post-1988",
)
ax.plot(
    1120.7 + 132.9 * pressure + -5.1 * pressure**2,
    pressure,
    linewidth=3,
    label="Hirschmann (2000), Recommended",
)
ax.plot(
    1085.7 + 132.9 * pressure + -5.1 * pressure**2,
    pressure,
    linewidth=3,
    label="Katz et al. (2003)",
)
ax.plot(
    1101.3 + 141.4 * pressure + -6.8 * pressure**2,
    pressure,
    linewidth=3,
    label="Updated BDD21",
)

ax.legend(fontsize=16, shadow=True, fancybox=True)

fig.savefig("solidus_compare.pdf", dpi=300, bbox_inches="tight")
