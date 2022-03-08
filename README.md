# BDD21
Readily estimate incompatible-element concentrations within both solid and liquid phases along a melting path, with seamless integration into your geodynamic workflow.

BDD21 is a collection of Python scripts (found in the sub-directory of your choice in the `src` directory):
- `AdiabaticDecompression.py`, which generates the mineralogy and chemistry along a prescribed 1-D adiabatic melting path;
- `ChemistryData.py`, which contains necessary parameters that characterise probed chemical elements, such as their valency, effective ionic radii ([Shannon, 1976](https://scripts.iucr.org/cgi-bin/paper?a12967)) and reference partition coefficients ([McKenzie and O'Nions, 1995](https://academic.oup.com/petrology/article-abstract/36/1/133/1433534)). It also provides estimated compositions of the Depleted MORB Mantle (DMM; [Salters and Stracke, 2004](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003GC000597)) and Primitive Mantle (PM; [McDonough and Sun, 1995](https://www.sciencedirect.com/science/article/pii/0009254194001404); [McKenzie and O'Nions, 1995](https://academic.oup.com/petrology/article-abstract/36/1/133/1433534));
- `Melt.py`, which is an implementation of the hydrous, mantle-melting parameterisation of [Katz, Spiegelman and Langmuir (2003)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2002GC000433) coupled to the thermodynamic framework of [McKenzie (1984)](https://academic.oup.com/petrology/article-abstract/25/3/713/1394279);
- `MeltChemistryFunctions.py`, which contains the main integrator function that returns element concentrations within both melt and residue along a melting path by solving the governing equations ![Equations](https://latex.codecogs.com/svg.latex?%5Csmall%20%5Ccolor%7BGolden%7D%20%5Cfrac%7Bdc_%7Bs%7D%7D%7BdX%7D%20=%20%5Cfrac%7Bc_%7Bs%7D%20-%20c_%7Bl%7D%7D%7B1%20-%20X%7D%20%5Cqquad%20c_%7Bl%7D%20=%20c_%7Bs%7D%5Cfrac%7B1%20-%20X%27%7D%7B%5Cbar%7BD%7D%20-%20%5Cbar%7BP%7DX%27%7D%20%5Cqquad%20X%27%20=%20%5Cfrac%7BX%20-%20X_%7B0%7D%7D%7B1%20-%20X_%7B0%7D%7D) ([White, McKenzie and O'Nions, 1992](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/92JB01749));
- `MeltChemistryH5.py`, which demonstrates how to use the framework in a post-processing manner using particle attributes stored in HDF5 format;
- `MeltChemistryNumba.py`, which contains the core functions of the BDD21 framework that calculate the mineralogy and partition coefficients at given melt fraction, pressure and temperature.

Example model outputs generated through [Fluidity](https://github.com/FluidityProject/fluidity) can be found in the `examples` directory. FLML files contain Python functions that demonstrate the use of BDD21 while the geodynamic simulation runs.

Contributions to the framework are highly encouraged; the current code architecture is sufficiently modular to accommodate both targeted and global improvements.

**Note**: If you make use of the dense output capability of LSODA through SciPy, be aware that the interpolation scheme is incorrect and can lead to inaccuracies. It is recommended you apply the proposed changes included in this [SciPy PR](https://github.com/scipy/scipy/pull/14552).
