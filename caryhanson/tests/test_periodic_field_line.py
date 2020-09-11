#!/usr/bin/env python3

import unittest
import numpy as np
from scipy.interpolate import interp1d
from caryhanson.helicalcoil import HelicalCoil
from caryhanson.periodic_field_line import periodic_field_line

class PeriodicFieldLineTests(unittest.TestCase):

    def test_default_axis(self):
        """
        Verify that we find the magnetic axis for the default coil shape
        """
        # Reference values were computed for n = 49:
        R_hires = np.array([0.98343287, 0.98343264, 0.98343194, 0.98343081, 0.98342928,
                       0.9834274 , 0.98342525, 0.98342287, 0.98342036, 0.98341778,
                       0.9834152 , 0.98341269, 0.98341031, 0.9834081 , 0.98340609,
                       0.98340432, 0.98340278, 0.98340148, 0.98340041, 0.98339956,
                       0.9833989 , 0.9833984 , 0.98339805, 0.98339783, 0.98339773,
                       0.98339773, 0.98339783, 0.98339805, 0.9833984 , 0.9833989 ,
                       0.98339956, 0.98340041, 0.98340148, 0.98340278, 0.98340432,
                       0.98340609, 0.9834081 , 0.98341031, 0.98341269, 0.9834152 ,
                       0.98341778, 0.98342036, 0.98342287, 0.98342525, 0.9834274 ,
                       0.98342928, 0.98343081, 0.98343194, 0.98343264])
        Z_hires = np.array([-6.90415050e-16,  1.54957848e-06,  3.10795179e-06,  4.68157071e-06,
                       6.27237439e-06,  7.87596478e-06,  9.48026686e-06,  1.10647878e-05,
                       1.26005494e-05,  1.40507254e-05,  1.53719656e-05,  1.65163459e-05,
                       1.74338356e-05,  1.80751373e-05,  1.83947243e-05,  1.83538816e-05,
                       1.79235496e-05,  1.70867767e-05,  1.58406072e-05,  1.41972616e-05,
                       1.21845118e-05,  9.84519941e-06,  7.23590636e-06,  4.42483520e-06,
                       1.48901443e-06, -1.48901443e-06, -4.42483520e-06, -7.23590637e-06,
                       -9.84519941e-06, -1.21845118e-05, -1.41972616e-05, -1.58406072e-05,
                       -1.70867767e-05, -1.79235496e-05, -1.83538816e-05, -1.83947243e-05,
                       -1.80751373e-05, -1.74338356e-05, -1.65163459e-05, -1.53719656e-05,
                       -1.40507254e-05, -1.26005494e-05, -1.10647878e-05, -9.48026686e-06,
                       -7.87596479e-06, -6.27237439e-06, -4.68157071e-06, -3.10795179e-06,
                       -1.54957848e-06])
        n_hires = len(R_hires)
        phi_hires = np.linspace(0, 2 * np.pi / 5, n_hires, endpoint=False)
        # Create a default helical coil
        hc = HelicalCoil()
        for nphi in [9, 18, 23]:
            R, phi, Z = periodic_field_line(hc, nphi)
            # Interpolate the reference result to the lower-resolution phi grid:
            R_ref = interp1d(phi_hires, R_hires, kind='cubic')(phi)
            Z_ref = interp1d(phi_hires, Z_hires, kind='cubic')(phi)
            print('For n={}, errors in R,Z are {}, {}'.format( \
                nphi, np.max(np.abs(R - R_ref)), np.max(np.abs(Z - Z_ref))))
            np.testing.assert_allclose(R, R_ref, atol=1.0e-8)
            np.testing.assert_allclose(Z, Z_ref, atol=1.0e-8)
        
if __name__ == "__main__":
    unittest.main()
