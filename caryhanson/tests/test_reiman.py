#!/usr/bin/env python3

import unittest
import numpy as np
from caryhanson.reiman import ReimanField

class ReimanFieldTests(unittest.TestCase):

    def test_default(self):
        """
        Test B for the default parameters
        """
        rf = ReimanField()
        BR, Bphi, BZ = rf.BR_Bphi_BZ(1.1, 0.7, 0.2)
        #BR_true = 0.03074843023437838
        BR_true = 0.03069701258456715
        Bphi_true = -1
        BZ_true = -0.015359884586783487
        #BZ_true = -0.015385593411689104
        places = 13
        print("Reiman field: diff in BR={}, Bphi={}, BZ={}".format(
            BR - BR_true, Bphi - Bphi_true, BZ - BZ_true))
        self.assertAlmostEqual(BR, BR_true, places=places)
        self.assertAlmostEqual(Bphi, Bphi_true, places=places)
        self.assertAlmostEqual(BZ, BZ_true, places=places)

    def test_derivatives(self):
        """
        Test the derivatives of B.
        """
        #rf = ReimanField(iotaj=[0])
        for j in range(30):
            niota = np.random.randint(1,4) # 1, 2, or 3
            iotaj = np.random.rand(niota) * 2 - 1
            B0 = np.random.rand() + 0.5
            R0 = np.random.rand() + 0.5
            nm = np.random.randint(1,4) # 1, 2, or 3
            ms = np.random.randint(1, 10, size=(nm,))
            eps = (np.random.rand(nm) - 0.5) * 0.1
            print('iotaj: ', iotaj)
            print('ms: ', ms)
            print('eps: ', eps)
            
            rf = ReimanField(iotaj=iotaj, B0=B0, R0=R0, ms=ms, eps=eps)
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
            
            BRp, Bphip, BZp = rf.BR_Bphi_BZ(R + delta, phi, Z)
            BRm, Bphim, BZm = rf.BR_Bphi_BZ(R - delta, phi, Z)
            Q0p, Q1p = rf.Q0_Q1(R + delta, phi, Z)
            Q0m, Q1m = rf.Q0_Q1(R - delta, phi, Z)
            
            d_BR_d_R_fd = (BRp - BRm) / delta2
            d_Bphi_d_R_fd = (Bphip - Bphim) / delta2
            d_BZ_d_R_fd = (BZp - BZm) / delta2
            d_Q0_d_R = (Q0p - Q0m) / delta2
            d_Q1_d_R = (Q1p - Q1m) / delta2
            print('d_Q0_d_R FD:', d_Q0_d_R)
            print('d_Q1_d_R FD:', d_Q1_d_R)
            
            BRp, Bphip, BZp = rf.BR_Bphi_BZ(R, phi + delta, Z)
            BRm, Bphim, BZm = rf.BR_Bphi_BZ(R, phi - delta, Z)
            Q0p, Q1p = rf.Q0_Q1(R, phi + delta, Z)
            Q0m, Q1m = rf.Q0_Q1(R, phi - delta, Z)

            d_BR_d_phi_fd = (BRp - BRm) / delta2
            d_Bphi_d_phi_fd = (Bphip - Bphim) / delta2
            d_BZ_d_phi_fd = (BZp - BZm) / delta2
            d_Q0_d_phi = (Q0p - Q0m) / delta2
            d_Q1_d_phi = (Q1p - Q1m) / delta2

            BRp, Bphip, BZp = rf.BR_Bphi_BZ(R, phi, Z + delta)
            BRm, Bphim, BZm = rf.BR_Bphi_BZ(R, phi, Z - delta)
            Q0p, Q1p = rf.Q0_Q1(R, phi, Z + delta)
            Q0m, Q1m = rf.Q0_Q1(R, phi, Z - delta)

            d_BR_d_Z_fd = (BRp - BRm) / delta2
            d_Bphi_d_Z_fd = (Bphip - Bphim) / delta2
            d_BZ_d_Z_fd = (BZp - BZm) / delta2
            d_Q0_d_Z = (Q0p - Q0m) / delta2
            d_Q1_d_Z = (Q1p - Q1m) / delta2
            print('d_Q0_d_Z FD:', d_Q0_d_Z)
            print('d_Q1_d_Z FD:', d_Q1_d_Z)

            print('Finite difference derivatives:')
            grad_B_fd = np.array([[d_BR_d_R_fd, d_BR_d_phi_fd, d_BR_d_Z_fd],
                         [d_Bphi_d_R_fd, d_Bphi_d_phi_fd, d_Bphi_d_Z_fd],
                         [d_BZ_d_R_fd, d_BZ_d_phi_fd, d_BZ_d_Z_fd]])
            print(grad_B_fd)
            # Evaluate the analytic derivatives:
            grad_B = rf.grad_B(R, phi, Z)
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
