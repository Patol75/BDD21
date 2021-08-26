#!/usr/bin/env python3

def func(X):
    from numpy import clip, exp, sqrt
    from scipy.optimize import root
    from scipy.special import erf
    from Constants import (conDepth, d, domainDim, k, kappa, mantTemp, oceAge,
                           rhoH0, stepLoc, stepWidth, surfTemp)

    def solvStep(var, dist2shall, Z, width, surfTemp, mantTemp, kappa, oceAge,
                 k1, k2, rhoH0, d):
        T, zCON, zOCE = var
        return ((erf(dist2shall / width * 4 - 2) + 1) / 2
                * (zCON - zOCE) + zOCE - Z,
                T - surfTemp - (mantTemp - surfTemp)
                * erf(zOCE / 2 / sqrt(kappa * oceAge)),
                T - (k1 * zCON + k2 - rhoH0 * d ** 2 * exp(-zCON / d)) / 3)

    depth = clip(domainDim[1] - X[1], 0, domainDim[1])
    k2 = k * surfTemp + rhoH0 * d ** 2
    k1 = (k * mantTemp + rhoH0 * d ** 2 * exp(-conDepth / d) - k2) / conDepth
    conTemp = (k1 * depth + k2 - rhoH0 * d ** 2 * exp(-depth / d)) / k
    oceTemp = surfTemp + (mantTemp - surfTemp) * erf(depth / 2
                                                     / sqrt(kappa * oceAge))
    # Depth
    if depth > conDepth:
        return mantTemp
    # Ocean
    elif (X[0] <= stepLoc[0] - stepWidth / 2
          or X[0] >= stepLoc[1] + stepWidth / 2):
        return oceTemp
    # Continent
    elif (X[0] >= stepLoc[0] + stepWidth / 2
          and X[0] <= stepLoc[1] - stepWidth / 2):
        return conTemp
    # Steps
    elif X[0] < stepLoc[0] + stepWidth / 2:
        res = root(solvStep, x0=(800, 1e5, 5e4), method='lm',
                   args=(X[0] - stepLoc[0] + stepWidth / 2, depth, stepWidth,
                         surfTemp, mantTemp, kappa, oceAge, k1, k2, rhoH0, d))
        return res.x[0]
    elif X[0] > stepLoc[1] - stepWidth / 2:
        res = root(solvStep, x0=(800, 1e5, 5e4), method='lm',
                   args=(stepLoc[1] + stepWidth / 2 - X[0], depth, stepWidth,
                         surfTemp, mantTemp, kappa, oceAge, k1, k2, rhoH0, d))
        return res.x[0]
