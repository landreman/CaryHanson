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
        BR_true = 0.03074843023437838
        Bphi_true = -1
        BZ_true = -0.015385593411689104
        places = 13
        print("Reiman field: diff in BR={}, Bphi={}, BZ={}".format(
            BR - BR_true, Bphi - Bphi_true, BZ - BZ_true))
        self.assertAlmostEqual(BR, BR_true, places=places)
        self.assertAlmostEqual(Bphi, Bphi_true, places=places)
        self.assertAlmostEqual(BZ, BZ_true, places=places)


        
if __name__ == "__main__":
    unittest.main()
