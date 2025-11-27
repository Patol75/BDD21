import numpy as np

from .melting import PeridotiteMelting


def non_zero_initial_melt(
    PTF: np.ndarray,
    T_pot_part: float,
    T_pot: float,
    P_LAB: float,
    adiab_grad: float,
    melt_inputs: dict[str, float],
) -> np.ndarray:
    """Prepends a fictitious melting path based on LAB pressure.

    Args:
        PTF (np.ndarray): Pressure, temperature, and melt fraction data.
        T_pot_part (float): Potential temperature in the mantle.
        T_pot (float): Potential temperature in the mantle.
        P_LAB (float): Pressure at the lithosphere-asthenosphere boundary.
        adiab_grad (float): Adiabatic gradient.
        melt_inputs (dict[str, float]): Inputs for melting calculations.

    Returns:
        np.ndarray: Updated pressure, temperature, and melt fraction data.
    """
    P_fict, P_step = 6.0, 0.02
    P_step_count = int((P_fict - PTF[0, 0]) / P_step) + 1
    P = np.linspace(P_fict, PTF[0, 0], P_step_count)

    if PTF[0, 0] > P_LAB - P_step:
        T = T_pot * np.exp(adiab_grad / T_pot * P)
    else:
        T = T_pot_part * np.exp(adiab_grad / T_pot * P)
        distrib_temp_over = int((P_LAB - PTF[0, 0]) / P_step) + 1
        T += np.pad(
            np.linspace(T_pot - T_pot_part, 0, distrib_temp_over),
            (P.size - distrib_temp_over, 0),
            constant_values=T_pot - T_pot_part,
        )
    T[-1] = PTF[1, 0]

    F = np.empty_like(P)
    melting_path = PeridotiteMelting(**melt_inputs)
    for i in range(P.size):  # Fictitious, prior melting path
        F[i] = melting_path.equilibrium_batch_melting(P[i], T[i])
    assert F[0] == 0 and F[-1] == PTF[2, 0]

    mask = np.zeros(F.size, dtype=bool)
    max_F = 0.0
    for i in range(mask.size):
        if F[i] > max_F:
            max_F = F[i]
            mask[i] = True
    mask[mask.nonzero()[0][0] - 1] = True  # Include entry before solidus
    PTF[0], PTF[1] = P[mask], T[mask]
    PTF[2] = F[mask]
    assert all(np.diff(PTF[2]) > 0)

    return PTF
