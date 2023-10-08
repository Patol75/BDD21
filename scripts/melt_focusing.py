# import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def quadra_fit(x, a, b, c):
    return a * x**2 + b * x + c


def arc_tan_fit(x, a, b, c):
    return a - np.arctan(b * x + c)


# Only Bulk below is up to date.

# Bulk
Rs = np.array([[0, 1], [1, 0.9074], [3, 0.8730], [5, 0.8639], [7, 0.8590]])
Tp = np.array([[1300, 0.9124], [1350, 0.8730], [1400, 0.7942]])
MORB = np.array([[15, 0.8075], [25, 0.8730], [35, 0.9110]])

# MORB
# Rs = np.array([[0, 1], [1, 0.9596], [3, 0.9110], [5, 0.8997], [7, 0.8932]])
# Tp = np.array([[1300, 0.9927], [1350, 0.9111], [1400, 0.8125]])
# MORB = np.array([[15, 0.8781], [25, 0.9108], [35, 0.9411]])

# popt, pcov = curve_fit(quadra_fit, MORB[:, 0], MORB[:, 1])
# plt.plot(MORB[:, 0], MORB[:, 1], linestyle="none", marker="o")
# plt.plot(np.linspace(15, 35, 101), quadra_fit(np.linspace(15, 35, 101), *popt))
# plt.show()

# print(quadra_fit(19, *popt) / quadra_fit(25, *popt))
# 0.9528 -> -0.0472 | BDD21
# 0.9588 -> -0.0412 | Bulk
# 0.9792 -> -0.0208 | MORB

popt, pcov = curve_fit(
    arc_tan_fit,
    Rs[:, 0],
    # Rs[:, 1],
    # Rs[:, 1] * (1 - 0.0472 + 0.0163),
    Rs[:, 1] * (1 - 0.0412),  # + 0.0282),
    # Rs[:, 1] * (1 - 0.0208 + 0.0498),
)
print(arc_tan_fit(6, *popt))

# plt.plot(
#     Rs[:, 0],
#     # Rs[:, 1],
#     # Rs[:, 1] * (1 - 0.0472 + 0.0163),
#     Rs[:, 1] * (1 - 0.0412 + 0.0282),
#     # Rs[:, 1] * (1 - 0.0208 + 0.0498),
#     linestyle="none",
#     marker="o",
# )
# plt.plot(np.linspace(0, 10, 101), arc_tan_fit(np.linspace(0, 10, 101), *popt))
# plt.show()

# print(arc_tan_fit(2.1, *popt) / arc_tan_fit(3, *popt))
# 1.0067 -> +0.0067 | BDD21
# 1.0104 -> +0.0104 | Bulk
# 1.0156 -> +0.0156 | MORB

popt, pcov = curve_fit(
    quadra_fit,
    Tp[:, 0],
    # Tp[:, 1],
    # Tp[:, 1] * (1 - 0.0472 + 0.0067),
    Tp[:, 1] * (1 - 0.0412 + 0.0104),
    # Tp[:, 1] * (1 - 0.0208 + 0.0156),
)
print(quadra_fit(1350, *popt))

# plt.plot(
#     Tp[:, 0],
#     # Tp[:, 1],
#     # Tp[:, 1] * (1 - 0.0472 + 0.0067),
#     Tp[:, 1] * (1 - 0.0412 + 0.0104),
#     # Tp[:, 1] * (1 - 0.0208 + 0.0156),
#     linestyle="none",
#     marker="o",
# )
# plt.plot(
#     np.linspace(1275, 1425, 101),
#     quadra_fit(np.linspace(1275, 1425, 101), *popt),
# )
# plt.show()

# print(quadra_fit(1325, *popt) / quadra_fit(1350, *popt))
# 1.0163 -> +0.0163 | BDD21
# 1.0282 -> +0.0282 | Bulk
# 1.0498 -> +0.0498 | MORB

# # # # # # # # # # # # # #
# Anhydrous solidus depth #
# # # # # # # # # # # # # #

# from numpy import sqrt
# from scipy.constants import g

# from constants import T_mantle, rho_mantle, adiab_grad
# from Melt import Katz
# from Melt_NEW import Katz

# katz = Katz()
# A1, A2, A3 = katz.A1, katz.A2, katz.A3
# A1, A2, A3 = Katz.A1, Katz.A2, Katz.A3
# a, b, c = A3, A2 - adiab_grad / rho_mantle / g * 1e9, A1 - T_mantle
# print((-b + sqrt(b ** 2 - 4 * a * c)) / 2 / a * 1e9 / rho_mantle / g / 1e3)

# 1548 K ->  51.97 km
# 1598 K ->  67.47 km | 59.56 km
# 1648 K ->  83.96 km
# 1698 K -> 101.63 km
# 1748 K -> 120.75 km

# 0.5 cm/yr -> u0 / 3 = 5.281e-11
# 2.1 cm/yr -> u0 / 3 = 2.218e-10
# 5.0 cm/yr -> u0 / 3 = 5.281e-10
# 10. cm/yr -> u0 / 3 = 1.056e-9
