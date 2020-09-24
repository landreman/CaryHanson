#!/usr/bin/env python3

"""
This module provides the routine for computing the island width
"""

import numpy as np

class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """
    pass

def island_width(pfl, tm):
    """
    pfl should be an output of circumference().
    tm: an output of tangent_map().
    """

    # We can only proceed for O points, not X points.
    if tm.residue <= 0 or tm.residue >= 1:
        return None, None
    
    periods = pfl.periods

    omega = tm.iota_per_period * pfl.nfp / periods
    q0 = int(np.round(-0.5 * periods + 0.25 * pfl.nfp / omega))
    print('q0:', q0)
    if q0 < 0:
        raise RuntimeError('q0 is negative.')

    # Form Sigma
    skq = tm.eperps[0]
    vects = [skq]
    for j in range(q0):
        print('Applying tangent map ', np.mod(j, periods))
        skq = np.dot(tm.single_period_tangent_maps[np.mod(j, periods)], skq)
        vects.append(skq)

    Sigma = 0
    print('Starting to accumulate Sigma')
    for j in range(periods):
        index = q0 + j
        index_mod = np.mod(index, periods)
        indexp1_mod = np.mod(index + 1, periods)
        print('Applying tangent map ', index_mod)
        skq = np.dot(tm.single_period_tangent_maps[index_mod], skq)
        delta = np.dot(tm.epars[indexp1_mod], skq)
        print('Contribution to Sigma: ', delta)
        Sigma += delta
        vects.append(skq)

    M = periods
    width = 2 * periods * pfl.circumference / (M * np.pi * Sigma)
    return width, vects
