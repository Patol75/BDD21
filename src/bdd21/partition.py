import numpy as np
from numba import njit, optional
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import DictType, ListType, Tuple, boolean, float64, int64, unicode_type
from scipy.constants import Avogadro, R, pi


@njit
def solid_composition(
    P: float,
    F: float,
    mnrl_mode_coeff: dict[str, np.ndarray],
    src_depletion: float = 0.0,
) -> np.ndarray:
    """Determines modal mineral abundances in the residual solid.

    The modal abundances are calculated based on pressure (P) and melt fraction (F)
    using polynomial coefficients for each mineral derived from a curve-fitting
    procedure of experimental data. A correction is also applied to account for
    systematic differences between primitive and depleted modal abundances, as described
    by Kimura and Kawabata - Geochemistry, Geophysics, Geosystems (2014).

    Args:
        P (float): Pressure in GPa
        F (float): Melt fraction
        mnrl_mode_coeff (dict[str, np.ndarray]): Dictionary of mineral mode coefficients
        src_depletion (float, optional): Source depletion factor

    Returns:
        np.ndarray: Array of mineral modal abundances
    """
    poly_var = np.kron(np.array([F**2, F, 1.0]), np.array([P**2, P, 1.0]))
    mnrl_map = {"ol": 0, "cpx": 2, "pl": 3, "spl": 4, "grt": 5}

    # Modal abundances of parameterised minerals
    mod_abund = np.zeros(6)
    for mnrl, index in mnrl_map.items():
        mod_abund[index] = mnrl_mode_coeff[mnrl] @ poly_var
    mod_abund.clip(0, 1, mod_abund)

    # Deduce orthopyroxene mode and ensure modal abundances sum to 1
    mod_abund_sum = mod_abund.sum()
    if mod_abund_sum <= 1.0:  # Remaining proportion -> orthopyroxene
        mod_abund[1] = 1.0 - mod_abund_sum
    else:  # Enforce proportions sum to 1
        mod_abund /= mod_abund_sum

    # Account for systematic differences between primitive and depleted modal abundances
    # using the correction (Figure 3) of
    # Kimura and Kawabata - Geochemistry, Geophysics, Geosystems (2014)
    correction = 0.04 * src_depletion
    mod_abund[0] += correction
    mod_abund[1] -= max(correction - mod_abund[2], 0)
    mod_abund[2] -= min(correction, mod_abund[2])

    return mod_abund


spec = [
    ("const_ele", optional(ListType(unicode_type))),
    ("elements", ListType(unicode_type)),
    ("ind_frte", ListType(Tuple((int64, unicode_type)))),
    ("frte_spl_grt", DictType(unicode_type, float64[:, :])),
    ("mask_single", DictType(unicode_type, boolean[:])),
    ("part_coeff", DictType(unicode_type, float64[:])),
    ("radii", DictType(unicode_type, float64[:])),
]


@jitclass(spec)
class PartitionCoefficients:
    D: float64[:, :]
    ri: float64[:, :]
    P: float64
    T: float64
    mod_abund: float64[:]
    kron_FT: float64[:]
    mask_monovalent: boolean[:]
    # mask_bivalent: boolean[:]
    mask_ree_y: boolean[:]
    mask_hfse: boolean[:]

    def __init__(self, elements, part_coeff, radii, const_ele=None) -> None:
        # https://numba.readthedocs.io/en/stable/user/troubleshoot.html
        const_ele = const_ele or List(["" for _ in range(0)])

        self.D = np.empty((len(elements), 6))
        self.ri = np.empty((len(elements), 2))
        for i, ele in enumerate(elements):
            self.D[i] = part_coeff[ele]
            self.ri[i] = radii[ele]

        self.set_extra_data()
        self.set_element_masks(elements, const_ele)

    def set_extra_data(self) -> None:
        """Defines additional data used in parameterisations."""
        # Le Roux et al. - American Mineralogist (2015)
        self.frte_spl_grt = Dict.empty(key_type=unicode_type, value_type=float64[:, :])
        self.frte_spl_grt["Cu"] = np.array([[0.13, 0.12, 0.09], [0.13, 0.12, 0.09]])
        self.frte_spl_grt["Ga"] = np.array([[0.08, 0.23, 0.28], [0.026, 0.38, 0.37]])
        self.frte_spl_grt["Ge"] = np.array([[0.67, 1.04, 1.12], [0.43, 0.87, 0.87]])
        self.frte_spl_grt["Ti"] = np.array([[0.01, 0.24, 0.34], [8e-3, 0.0656, 0.124]])
        self.frte_spl_grt["Sc"] = np.array([[0.20, 0.35, 1.51], [0.15, 0.495, 0.84]])
        self.frte_spl_grt["V"] = np.array([[0.10, 0.30, 0.8], [0.14, 1.06, 1.48]])
        self.frte_spl_grt["Cr"] = np.array([[0.8, 2.5, 8.0], [0.79, 8.8, 7.5]])
        self.frte_spl_grt["Mn"] = np.array([[0.77, 0.75, 1.11], [0.781, 0.640, 0.768]])
        self.frte_spl_grt["Zn"] = np.array([[0.99, 0.68, 0.48], [0.96, 0.451, 0.333]])
        self.frte_spl_grt["Fe"] = np.array([[1.06, 0.69, 0.71], [1.034, 0.55, 0.49]])
        self.frte_spl_grt["Co"] = np.array([[2.1, 1.04, 1.06], [2.37, 1.29, 0.86]])
        self.frte_spl_grt["Ni"] = np.array([[6.2, 3.7, 3.2], [6.2, 3.7, 22.0]])

    def set_element_masks(self, elements, const_ele) -> None:
        """Defines masks for elements based on their valence and other properties."""
        monovalent = List(["Na", "K", "Rb", "Cs"])
        # bivalent = List(["Ba", "Pb", "Sr"])
        ree_y = List(["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd"])
        ree_y.extend(["Tb", "Dy", "Y", "Ho", "Er", "Tm", "Yb", "Lu"])
        hfse = List(["Ti", "Zr", "Hf"])
        singles = List(["Na", "Ti", "Sr", "Zr", "Nb", "Sm"])
        singles.extend(["Hf", "Ta", "Pb", "Th", "U"])

        self.mask_single = Dict.empty(key_type=unicode_type, value_type=boolean[:])
        for s_ele in singles:
            self.mask_single[s_ele] = np.asarray(
                [ele == s_ele and s_ele not in const_ele for ele in elements]
            )

        self.mask_monovalent = np.asarray(
            [ele in monovalent and ele not in const_ele for ele in elements]
        )
        # self.mask_bivalent = np.asarray(
        #     [ele in bivalent and ele not in const_ele for ele in elements]
        # )
        self.mask_ree_y = np.asarray(
            [ele in ree_y and ele not in const_ele for ele in elements]
        )
        self.mask_hfse = np.asarray(
            [ele in hfse and ele not in const_ele for ele in elements]
        )

        self.ind_frte = List(
            [
                (i, ele)
                for i, ele in enumerate(elements)
                if ele in self.frte_spl_grt and ele not in const_ele
            ]
        )

    def al_phase_select(
        self, spl: float | np.ndarray, grt: float | np.ndarray
    ) -> float | np.ndarray:
        """Selects values based on the stability of the aluminium-rich phase.

        Linear interpolation between values is performed when garnet and spinel are
        present; spinel values are returned when the aluminium-rich phase is exhausted.
        """
        if self.mod_abund[4] + self.mod_abund[5] > 0:
            return (spl * self.mod_abund[4] + grt * self.mod_abund[5]) / (
                self.mod_abund[4] + self.mod_abund[5]
            )
        else:
            return spl

    def part_coeff_ideal_radius(
        self, D0: float, E: float, r0: float, ri: np.ndarray | float
    ):
        """Calculates the partition coefficient based on an ideal cation.

        Partition coefficient between a cation and a mineral at a crystal site according
        to the lattice strain model (more information can be found at
        https://doi.org/10.1007/978-3-319-39312-4_347).

        The partition coefficient (D0) is dimensionless, Young's modulus (E) is in GPa,
        and ionic radii (r0 and ri) are in Å.

        Equations 11 and 12 of Brice - Journal of Crystal Growth (1975)
        """
        radius_term = r0 / 2 * (ri - r0) ** 2 + (ri - r0) ** 3 / 3

        return D0 * np.exp(-4 * pi * E * Avogadro / R / self.T * radius_term * 1e-21)

    def part_coeff_same_charge(
        self, Da: float, E: float, r0: float, ri: np.ndarray | float, ra: float
    ):
        """Calculates the partition coefficient based on another cation.

        Partition coefficient between a cation and a mineral at a crystal site from the
        value of a cation of similar charge but different radius.

        The partition coefficient (D0) is dimensionless, Young's modulus (E) is in GPa,
        and ionic radii (r0 and ri) are in Å.

        Equation 3 of Blundy and Wood - Nature (1994)
        """
        radius_term = r0 / 2 * (ra**2 - ri**2) + (ri**3 - ra**3) / 3

        return Da * np.exp(-4 * pi * E * Avogadro / R / self.T * radius_term * 1e-21)

    def ol_opx_cpx_le_roux_2015(self) -> None:
        """Table 3 from Le Roux et al. - American Mineralogist (2015)"""

        for i, ele in self.ind_frte:
            self.D[i, :3] = self.al_phase_select(
                self.frte_spl_grt[ele][0], self.frte_spl_grt[ele][1]
            )

    def ol_ree_sun_liang_2014(self) -> None:
        """Equations A1a-c of Sun and Liang - Chemical Geology (2014)"""
        # Al content in olivine (mole per four-oxygen)
        X_Al = self.al_phase_select(
            np.array([-1.166e-4, 1.628e-1, 6.632e-5, -1.001e-1]) @ self.kron_FT,
            np.array([4.717e-6, -7.203e-3, -2.738e-6, 1.069e-2]) @ self.kron_FT,
        )
        # Forsterite content in olivine (Magnesium number)
        Mg_num = self.al_phase_select(
            np.array([4.746e-5, -2.941e-2, 4.057e-6, 8.990e-1]) @ self.kron_FT,
            np.array([-6.102e-7, 6.992e-2, 1.893e-5, 8.604e-1]) @ self.kron_FT,
        )

        D0 = np.exp(-0.44 - 0.18 * self.P + 123.75 * X_Al - 1.49 * Mg_num)
        r0, E = 0.809, 298

        self.D[self.mask_ree_y, 0] = self.part_coeff_ideal_radius(
            D0, E, r0, self.ri[self.mask_ree_y, 1]
        )

    def opx_bivalent_wood_blundy_2014(self) -> None:
        """Section 3.11.10.6.3 of Wood and Blundy - Treatise on Geochemistry (2014)"""
        # Cation content of Al per six-oxygen
        X_Al = self.al_phase_select(
            np.array([-9.015e-4, 7.717e-1, 5.431e-4, -5.914e-1]) @ self.kron_FT,
            np.array([1.832e-3, -3.710, -9.000e-4, 1.953]) @ self.kron_FT,
        )
        # Cation content of Ca per six-oxygen
        X_Ca = self.al_phase_select(
            np.array([-1.097e-3, 1.747, 2.037e-4, -2.385e-1]) @ self.kron_FT,
            np.array([-1.862e-4, 2.338e-1, 1.426e-4, -1.559e-1]) @ self.kron_FT,
        )

        D_Mg, r_Mg = 1, 0.72
        r0 = 0.753 + 0.118 * X_Al + 0.114 * X_Ca + 0.08
        E = 240

        self.D[self.mask_bivalent, 1] = self.part_coeff_same_charge(
            D_Mg, E, r0, self.ri[self.mask_bivalent, 0], r_Mg
        )

    def opx_ree_sun_liang_2013(self) -> None:
        """Equations 4-6 of Sun and Liang - Geochimica et Cosmochimica Acta (2013)"""
        # Cation content of Al on the tetrahedral site per six-oxygen
        X_Al_T = self.al_phase_select(
            np.array([-8.137e-4, +1.060, +1.786e-4, -1.333e-1]) @ self.kron_FT,
            np.array([1.070e-3, -2.114, -5.671e-4, +1.196]) @ self.kron_FT,
        )
        # Cation content of Ca on the M2 site per six-oxygen
        X_Ca_M2 = self.al_phase_select(
            np.array([-1.097e-3, +1.747, +2.037e-4, -2.385e-1]) @ self.kron_FT,
            np.array([-1.862e-4, +2.338e-1, +1.426e-4, -1.559e-1]) @ self.kron_FT,
        )
        # Cation content of Mg on the M2 site per six-oxygen
        X_Mg_M2 = self.al_phase_select(
            np.array([1.274e-3, -1.969, -2.631e-4, +1.242]) @ self.kron_FT,
            np.array([1.386e-4, -6.299e-2, -1.497e-4, +1.063]) @ self.kron_FT,
        )
        # Cation fraction of Ti in the melt per six-oxygen
        X_Ti_melt = self.al_phase_select(
            np.array([2.087e-4, -4.405e-1, +1.498e-5, +1.156e-2]) @ self.kron_FT,
            np.array([1.225e-4, -3.037e-1, -2.206e-5, +8.261e-2]) @ self.kron_FT,
        )

        D0 = np.exp(
            -5.37
            + 3.87e4 / R / self.T
            + 3.54 * X_Al_T
            + 3.56 * X_Ca_M2
            - 0.84 * X_Ti_melt
        )
        r0 = 0.693 + 0.432 * X_Ca_M2 + 0.228 * X_Mg_M2
        E = (1.85 * r0 - 1.37 - 0.53 * X_Ca_M2) * 1e3

        self.D[self.mask_ree_y, 1] = self.part_coeff_ideal_radius(
            D0, E, r0, self.ri[self.mask_ree_y, 1]
        )

    def opx_hfse_sun_liang_2013(self) -> None:
        """Equations 9-11 of Sun and Liang - Geochimica et Cosmochimica Acta (2013)"""
        # Cation content of Al on the tetrahedral site per six-oxygen
        X_Al_T = self.al_phase_select(
            np.array([-8.137e-4, 1.060, 1.786e-4, -1.333e-1]) @ self.kron_FT,
            np.array([1.070e-3, -2.114, -5.671e-4, 1.196]) @ self.kron_FT,
        )
        # Cation content of Ca on the M2 site per six-oxygen
        X_Ca_M2 = self.al_phase_select(
            np.array([-1.097e-3, 1.747, 2.037e-4, -2.385e-1]) @ self.kron_FT,
            np.array([-1.862e-4, 2.338e-1, 1.426e-4, -1.559e-1]) @ self.kron_FT,
        )
        # Cation content of Fe on the M1 site per six-oxygen
        X_Fe_M1 = self.al_phase_select(
            np.array([-1.392e-4, 2.231e-1, -2.464e-5, 1.158e-1]) @ self.kron_FT,
            np.array([-1.032e-5, -1.845e-2, -3.754e-6, 8.845e-2]) @ self.kron_FT,
        )
        # Cation content of Mg on the M1 site per six-oxygen
        X_Mg_M1 = self.al_phase_select(
            np.array([-6.692e-5, 6.084e-1, -4.662e-4, 1.507]) @ self.kron_FT,
            np.array([-6.528e-4, 1.442, 3.373e-4, 1.330e-1]) @ self.kron_FT,
        )
        # Cation content of Mg on the M2 site per six-oxygen
        X_Mg_M2 = self.al_phase_select(
            np.array([1.274e-3, -1.969, -2.631e-4, 1.242]) @ self.kron_FT,
            np.array([1.386e-4, -6.299e-2, -1.497e-4, 1.063]) @ self.kron_FT,
        )

        D0 = np.exp(
            -4.825
            + 3.178e4 / R / self.T
            + 4.172 * X_Al_T
            + 8.551 * X_Ca_M2 * X_Mg_M2
            - 2.616 * X_Fe_M1
        )
        r0 = 0.618 + 0.032 * X_Ca_M2 + 0.03 * X_Mg_M1
        E = 2203

        self.D[self.mask_hfse, 1] = self.part_coeff_ideal_radius(
            D0, E, r0, self.ri[self.mask_hfse, 0]
        )

    def cpx_monovalent_wood_blundy_2014(self) -> None:
        """Section 3.11.10.1.3 of Wood and Blundy - Treatise on Geochemistry (2014)"""
        # Cation content of Al on the M1 site per six-oxygen
        X_Al_M1 = self.al_phase_select(
            np.array([-3.546e-3, 5.281, 5.916e-4, -7.892e-1]) @ self.kron_FT,
            np.array([5.612e-5, -4.683e-2, -3.754e-4, 8.188e-1]) @ self.kron_FT,
        )
        # Cation content of Ca on the M2 site per six-oxygen
        X_Ca_M2 = self.al_phase_select(
            np.array([-2.459e-3, 4.017, -1.587e-3, 3.098]) @ self.kron_FT,
            np.array([-1.699e-4, 1.979e-1, -2.198e-4, 7.507e-1]) @ self.kron_FT,
        )

        D_Na = np.exp(
            (10_367 + 2100 * self.P - 165 * self.P**2) / self.T
            - 10.27
            + 0.358 * self.P
            - 0.0184 * self.P**2
        )
        # Shannon - Acta Crystallographica (1976)
        r_Na = 1.18
        # r0 revised from Figure 8 of Mollo et al. - Earth-Science Reviews (2020)
        r0 = 0.974 + 0.067 * X_Ca_M2 - 0.051 * X_Al_M1 + 0.23
        E = (318.6 + 6.9 * self.P - 0.036 * self.T) / 3

        self.D[self.mask_monovalent, 2] = self.part_coeff_same_charge(
            D_Na, E, r0, self.ri[self.mask_monovalent, 1], r_Na
        )

    def cpx_ree_schoneveld_2018(self) -> None:
        """Equation 84 from Schoneveld - ANU Thesis (2018)"""
        # Cation content of Mg on the M2 site per six-oxygen
        X_Mg_M2 = self.al_phase_select(
            np.array([3.000e-3, -4.675, 1.270e-3, -1.688]) @ self.kron_FT,
            np.array([-5.157e-6, 2.422e-1, 2.201e-4, 1.312e-1]) @ self.kron_FT,
        )
        # Molar fraction of Al2O3 in the melt
        X_Al2O3_melt = self.al_phase_select(
            np.array([3.055e-4, -6.387e-1, -1.313e-4, 3.259e-1]) @ self.kron_FT,
            np.array([1.264e-4, -2.287e-1, -1.855e-4, 4.052e-1]) @ self.kron_FT,
        )
        # Molar fraction of SiO2 in the melt
        X_SiO2_melt = self.al_phase_select(
            np.array([5.050e-4, -8.057e-1, -3.469e-4, 1.065]) @ self.kron_FT,
            np.array([8.899e-8, 2.692e-2, -1.265e-4, 6.800e-1]) @ self.kron_FT,
        )

        D0 = np.exp(-2.07 + 0.52e4 / self.T)
        r0 = 1.036 - 0.08 * X_Mg_M2
        E = 238

        γ_REE = 2.54 * np.log(X_SiO2_melt)
        # Partition coefficient for Ca - Run 1948
        # Adam and Green - Contributions to Mineralogy and Petrology (2006)
        D_Ca = 1.99

        extra_terms = np.exp(γ_REE) * 2 * X_Al2O3_melt / X_SiO2_melt * D_Ca

        self.D[self.mask_ree_y, 2] = (
            self.part_coeff_ideal_radius(D0, E, r0, self.ri[self.mask_ree_y, 1])
            * extra_terms
        )

    def grt_ree_sun_liang_2014(self) -> None:
        """Equations A4a-c of Sun and Liang - Chemical Geology (2014)"""
        # Cation content of Ca in garnet per 12-oxygen
        X_Ca = np.array([4.516e-5, -2.685e-1, -2.325e-4, 7.989e-1]) @ self.kron_FT

        D0 = np.exp(
            -2.01
            + (9.03e4 - 93.02 * self.P * (37.78 - self.P)) / R / self.T
            - 1.04 * X_Ca
        )
        r0 = 0.785 + 0.153 * X_Ca
        E = (-1.67 + 2.35 * r0) * 1e3

        self.D[self.mask_ree_y, 5] = self.part_coeff_ideal_radius(
            D0, E, r0, self.ri[self.mask_ree_y, 1]
        )

    def opx_bedard_2007(self) -> None:
        """Equations 56a and 56b from Bedard - Chemical Geology (2007)"""
        if self.P < 3.14:
            self.D[self.mask_single["Na"], 1] = np.exp(-4.291548 + 0.719040 * self.P)
        else:
            self.D[self.mask_single["Na"], 1] = np.exp(-3.041418 + 0.320851 * self.P)

    def cpx_bedard_2014(self) -> None:
        """Equations from Bedard - Geochemistry, Geophysics, Geosystems (2014)

        Several expressions rely on the partition coefficient of Sm. Ensure the
        appropriate method (currently `cpx_ree_schoneveld_2018`) runs first to update
        the distribution value of Sm with clinopyroxene. If Sm is not present, then
        related partition coefficients are not updated.
        """
        # Cation content of Al on the tetrahedral site per six-oxygen
        X_Al_T = self.al_phase_select(
            np.array([-1.493e-3, 2.142, 2.126e-4, -1.765e-1]) @ self.kron_FT,
            np.array([-1.008e-4, 3.649e-1, -4.349e-4, 8.978e-1]) @ self.kron_FT,
        )
        # Weight per cent of MgO in the melt
        X_MgO_melt = self.al_phase_select(
            np.array([1.039e-3, 1.433e1, 2.330e-2, -2.742e1]) @ self.kron_FT,
            np.array([-3.266e-3, 1.423e1, 2.762e-2, -3.341e1]) @ self.kron_FT,
        )
        # Molar ratio Mg / (Mg + Fe) in clinopyroxene
        Mg_num = self.al_phase_select(
            np.array([7.865e-4, -1.185, -1.340e-4, 1.112]) @ self.kron_FT,
            np.array([-1.785e-4, 4.067e-1, 7.077e-5, 7.590e-1]) @ self.kron_FT,
        )

        if "Sm" in self.mask_single:
            # Equation S388c
            D_Sm = self.D[self.mask_single["Sm"], 2]
            D_Ti = np.exp((np.log(D_Sm) + 0.162190) / 0.999163)
            self.D[self.mask_single["Ti"], 2] = D_Ti
            # Equation S133
            D_Hf = np.exp(-0.436056 + 1.027121 * np.log(D_Ti))
            self.D[self.mask_single["Hf"], 2] = D_Hf
            # Equation S119
            self.D[self.mask_single["Zr"], 2] = np.exp(
                -0.48986 + 1.071278 * np.log(D_Hf)
            )
            # Equation S214
            D_Th = np.exp(-2.78091 + 1.76865 * np.log(D_Ti))
            self.D[self.mask_single["Th"], 2] = D_Th
            # Equation S224
            self.D[self.mask_single["U"], 2] = np.exp(
                -0.78777 + 0.885892 * np.log(D_Th)
            )

        # Equation S92
        D_Ta = np.exp(-4.92448 + 8.40847 * X_Al_T)
        self.D[self.mask_single["Ta"], 2] = D_Ta
        # Equation S90
        self.D[self.mask_single["Nb"], 2] = np.exp(-1.31570 + 0.88397 * np.log(D_Ta))
        # Equation S19
        self.D[self.mask_single["Sr"], 2] = np.exp(
            -1.87504 - 0.23387 * np.log(X_MgO_melt)
        )
        # Equation S240
        self.D[self.mask_single["Pb"], 2] = np.exp(-0.20031 - 4.51647 * Mg_num)

    def cpx_wood_blundy_2014(self) -> None:
        """Equations from Wood and Blundy - Treatise on Geochemistry (2014)"""
        # Cation content of Al on the tetrahedral site per six-oxygen
        X_Al_T = self.al_phase_select(
            np.array([-1.493e-3, 2.142, 2.126e-4, -1.765e-1]) @ self.kron_FT,
            np.array([-1.008e-4, 3.649e-1, -4.349e-4, 8.978e-1]) @ self.kron_FT,
        )

        # Equation 50
        D_Ta = 10 ** (-2.127 + 3.769 * X_Al_T)
        self.D[self.mask_single["Ta"], 2] = D_Ta
        # Equation 51
        self.D[self.mask_single["Nb"], 2] = 0.003 + 0.292 * D_Ta

    def update_part_coeff(
        self, P: float, T: float, F: float, mod_abund: np.ndarray
    ) -> None:
        """Updates partition coefficient based on published parameterisations."""
        self.P = P
        self.T = T
        self.mod_abund = mod_abund

        self.kron_FT = np.kron(np.array([F, 1.0]), np.array([T, 1.0]))

        self.ol_opx_cpx_le_roux_2015()

        self.ol_ree_sun_liang_2014()
        # self.opx_bivalent_wood_blundy_2014()
        self.opx_ree_sun_liang_2013()
        self.opx_hfse_sun_liang_2013()
        self.cpx_monovalent_wood_blundy_2014()
        self.cpx_ree_schoneveld_2018()
        self.grt_ree_sun_liang_2014()

        self.opx_bedard_2007()
        self.cpx_bedard_2014()
        # self.cpx_wood_blundy_2014()
