#!/usr/bin/env python3

import unittest
from caryhanson.helicalcoil import HelicalCoil

class HelicalCoilTests(unittest.TestCase):

    def test_default(self):
        """
        Test B for the default coil shape
        """
        for nphi in [400, 800, 1600]:
            if nphi==400:
                # Reference values are computed for nphi=400, so we
                # should get really good agreement in this case.
                places=13
            else:
                # For other values of nphi, the computed B should be
                # similar but it will not be the same to all digits.
                places=6
            hc = HelicalCoil(nphi=nphi)
            BR, Bphi, BZ = hc.BR_Bphi_BZ(1.1, 0.7, 0.2)
            self.assertAlmostEqual(BR, 0.4305941097518284, places=places)
            self.assertAlmostEqual(Bphi, 0.7557506105818288, places=places)
            self.assertAlmostEqual(BZ, 0.17403729222415867, places=places)


    def test_phi_derivative(self):
        """
        Verify that the d/dphi derivatives are reasonably close to finite-difference derivatives.
        """
        
if __name__ == "__main__":
    unittest.main()
