#!/usr/bin/env python3

"""
This module contains the parent class for all B field varieties
"""

import numpy as np

class Field:
    """
    This class is the parent class for all types of B fields.
    """

    def d_RZ_d_phi(self, phi, RZ):
        """
        This is the function used for integrating the field line ODE.
        """
        R = RZ[0]
        Z = RZ[1]
        BR, Bphi, BZ = self.BR_Bphi_BZ(R, phi, Z)
        return [R * BR / Bphi, R * BZ / Bphi]
