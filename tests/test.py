from time import perf_counter_ns

from numba.typed import List

from bdd21.chemistry_data import mnrl_mode_coeff, part_coeff, radii
from bdd21.partition import PartitionCoefficients, solid_composition

elements = ["Cs", "Rb", "Ba", "Th", "U", "Nb", "Ta", "La", "Ce", "Pb", "Pr", "Sr", "Nd"]
elements.extend(["Zr", "Hf", "Sm", "Eu", "Gd", "Tb", "Dy", "Y", "Ho", "Er", "Yb", "Lu"])
elements.extend(["Na", "K", "Ti", "P", "Li", "Sc", "V", "Cr", "Co", "Ni", "Cu", "Zn"])

test = PartitionCoefficients(part_coeff, radii, const_ele=List(["Cs", "Rb", "Li"]))
test.set_elements(elements)
test.set_PTF(2.96, 1660.7160372, 0.0)
test.set_mod_abund(solid_composition(test.P, test.F, mnrl_mode_coeff))
test.update_part_coeff()
print(test.D)
begin = perf_counter_ns()
for _ in range(100_000):
    test.update_part_coeff()
print(f"Time taken: {(perf_counter_ns() - begin) / 1e6} ms")
# print(test.D)
