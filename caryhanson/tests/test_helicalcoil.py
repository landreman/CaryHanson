#!/usr/bin/env python3

import unittest
from caryhanson.helicalcoil import HelicalCoil

class HelicalCoilTests(unittest.TestCase):

    def test_default(self):
        """
        Test B for the default coil shape
        """
        for nphi in [400, 401, 503, 800, 1600]:
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
            BR_true = 0.4305941097518284
            Bphi_true = 0.7557506105818288
            BZ_true = 0.17403729222415867
            print("For nphi={}, diff in BR={}, Bphi={}, BZ={}".format(
                nphi, BR - BR_true, Bphi - Bphi_true, BZ - BZ_true))
            self.assertAlmostEqual(BR, BR_true, places=places)
            self.assertAlmostEqual(Bphi, Bphi_true, places=places)
            self.assertAlmostEqual(BZ, BZ_true, places=places)


    def test_optimized(self):
        """
        Test B for the optimized coil shape
        """
        for nphi in [400, 401, 503, 800, 1600]:
            if nphi==400:
                # Reference values are computed for nphi=400, so we
                # should get really good agreement in this case.
                places=13
            else:
                # For other values of nphi, the computed B should be
                # similar but it will not be the same to all digits.
                places=6
            hc = HelicalCoil.optimized(nphi=nphi)
            BR, Bphi, BZ = hc.BR_Bphi_BZ(1.1, 0.7, 0.2)
            BR_true = 0.2740135193723583
            Bphi_true = 0.8537024493578993
            BZ_true = 0.2459219974428098
            print("For nphi={}, diff in BR={}, Bphi={}, BZ={}".format(
                nphi, BR - BR_true, Bphi - Bphi_true, BZ - BZ_true))
            self.assertAlmostEqual(BR, BR_true, places=places)
            self.assertAlmostEqual(Bphi, Bphi_true, places=places)
            self.assertAlmostEqual(BZ, BZ_true, places=places)


    def test_phi_derivative(self):
        """
        Verify that the d/dphi derivatives are reasonably close to finite-difference derivatives.
        """
        
        
if __name__ == "__main__":
    unittest.main()
