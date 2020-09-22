#!/usr/bin/env python3

import unittest
import numpy as np
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


    def test_phi_derivative_of_position(self):
        """
        Verify that the d/dphi derivatives are reasonably close to finite-difference derivatives.
        """
        hc = HelicalCoil(nphi=20000)
        d_X_d_phi = (np.roll(hc.X_coil, -1, axis=0) - np.roll(hc.X_coil, 1, axis=0)) / (2 * hc.dphi)
        d_Y_d_phi = (np.roll(hc.Y_coil, -1, axis=0) - np.roll(hc.Y_coil, 1, axis=0)) / (2 * hc.dphi)
        d_Z_d_phi = (np.roll(hc.Z_coil, -1, axis=0) - np.roll(hc.Z_coil, 1, axis=0)) / (2 * hc.dphi)
        print('Differences in d_X_d_phi, d_Y_d_phi, d_Z_d_phi: {}, {}, {}'.format(
            np.max(np.abs(d_X_d_phi - hc.d_X_d_phi_coil)),
            np.max(np.abs(d_Y_d_phi - hc.d_Y_d_phi_coil)),
            np.max(np.abs(d_Z_d_phi - hc.d_Z_d_phi_coil))))

        rtol = 1e-6
        atol = 1e-6
        np.testing.assert_allclose(d_X_d_phi, hc.d_X_d_phi_coil, rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_Y_d_phi, hc.d_Y_d_phi_coil, rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_Z_d_phi, hc.d_Z_d_phi_coil, rtol=rtol, atol=atol)

    def test_derivatives_of_B(self):
        """
        Test the derivatives of B.
        """
        #hc = HelicalCoil()
        for j in range(10):
            B0 = np.random.rand() + 0.5
            R0 = np.random.rand() + 0.5
            I = (np.random.rand() - 0.5) * 0.1 * np.array([1, -1])
            
            hc = HelicalCoil(B0=B0, R0=R0, I=I)
            # Pick a random location:
            R = np.random.rand() + 0.5
            phi = np.random.rand() * 10 - 5
            Z = np.random.rand() * 4 - 2
            #R = 1.1
            #phi = 0
            #Z = 0.0

            # Evaluate finite difference derivatives
            delta = 1e-6
            delta2 = delta * 2
            
            BRp, Bphip, BZp = hc.BR_Bphi_BZ(R + delta, phi, Z)
            BRm, Bphim, BZm = hc.BR_Bphi_BZ(R - delta, phi, Z)
            
            d_BR_d_R_fd = (BRp - BRm) / delta2
            d_Bphi_d_R_fd = (Bphip - Bphim) / delta2
            d_BZ_d_R_fd = (BZp - BZm) / delta2
            
            BRp, Bphip, BZp = hc.BR_Bphi_BZ(R, phi + delta, Z)
            BRm, Bphim, BZm = hc.BR_Bphi_BZ(R, phi - delta, Z)

            d_BR_d_phi_fd = (BRp - BRm) / delta2
            d_Bphi_d_phi_fd = (Bphip - Bphim) / delta2
            d_BZ_d_phi_fd = (BZp - BZm) / delta2

            BRp, Bphip, BZp = hc.BR_Bphi_BZ(R, phi, Z + delta)
            BRm, Bphim, BZm = hc.BR_Bphi_BZ(R, phi, Z - delta)

            d_BR_d_Z_fd = (BRp - BRm) / delta2
            d_Bphi_d_Z_fd = (Bphip - Bphim) / delta2
            d_BZ_d_Z_fd = (BZp - BZm) / delta2

            print('Finite difference derivatives:')
            grad_B_fd = np.array([[d_BR_d_R_fd, d_BR_d_phi_fd, d_BR_d_Z_fd],
                         [d_Bphi_d_R_fd, d_Bphi_d_phi_fd, d_Bphi_d_Z_fd],
                         [d_BZ_d_R_fd, d_BZ_d_phi_fd, d_BZ_d_Z_fd]])
            print(grad_B_fd)
            # Evaluate the analytic derivatives:
            grad_B = hc.grad_B(R, phi, Z)
            print('Analytic derivatives:')
            print(grad_B)

            print('Differences:')
            print(grad_B - grad_B_fd)

            places = 6
            self.assertAlmostEqual(d_BR_d_R_fd, grad_B[0,0], places=places)
            self.assertAlmostEqual(d_BR_d_phi_fd, grad_B[0,1], places=places)
            self.assertAlmostEqual(d_BR_d_Z_fd, grad_B[0,2], places=places)
            
            self.assertAlmostEqual(d_Bphi_d_R_fd, grad_B[1,0], places=places)
            self.assertAlmostEqual(d_Bphi_d_phi_fd, grad_B[1,1], places=places)
            self.assertAlmostEqual(d_Bphi_d_Z_fd, grad_B[1,2], places=places)
            
            self.assertAlmostEqual(d_BZ_d_R_fd, grad_B[2,0], places=places)
            self.assertAlmostEqual(d_BZ_d_phi_fd, grad_B[2,1], places=places)
            self.assertAlmostEqual(d_BZ_d_Z_fd, grad_B[2,2], places=places)

        
if __name__ == "__main__":
    unittest.main()
