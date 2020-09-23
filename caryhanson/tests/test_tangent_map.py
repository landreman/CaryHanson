#!/usr/bin/env python3

import unittest
import numpy as np
from caryhanson.helicalcoil import HelicalCoil
from caryhanson.reiman import ReimanField
from caryhanson.periodic_field_line import periodic_field_line, func, jacobian
from caryhanson.tangent_map import tangent_map

class TangentMapTests(unittest.TestCase):

    def test_helical_coil_axis(self):
        """
        Tests properties of the tangent map for the magnetic axis of the
        helical coil with straight winding law.
        """

        field = HelicalCoil()
        tm = tangent_map(field, R0=0.98343287162797)
        places = 10
        self.assertAlmostEqual(tm.iota_per_period, 0.11287146743203362, places=places)
        self.assertAlmostEqual(tm.iota, 0.5643573371601681, places=places)
        M = np.array([[ 0.75888781375709 ,  1.37679244594353 ],
                      [-0.308026489831821,  0.758887765057904]])
        np.testing.assert_allclose(tm.mat, M, atol=1.0e-6)
        self.assertAlmostEqual(tm.residue, 0.12055610529625149, places=places)
        
    def test_reiman_axis(self):
        """
        Tests properties of the tangent map for the magnetic axis of the
        Reiman model field.
        """

        field = ReimanField()
        tm = tangent_map(field, R0=1.0, rtol=1e-12, atol=1e-12)
        places = 10
        self.assertAlmostEqual(tm.iota_per_period, 0.15, places=places)
        self.assertAlmostEqual(tm.iota, 0.15, places=places)
        M = np.array([[ 0.587785252292183, -0.809016994374573],
                      [ 0.809016994374573, 0.587785252292183]])
        np.testing.assert_allclose(tm.mat, M, atol=1.0e-6)
        self.assertAlmostEqual(tm.residue, 0.20610737385390826, places=places)
        
    def test_reiman_Opoint(self):
        """
        Tests properties of the tangent map for the O-point of the
        Reiman model field.
        """

        field = ReimanField()
        tm = tangent_map(field, R0=1.210161051261806, periods=6, rtol=1e-12, atol=1e-12)
        places = 10
        self.assertAlmostEqual(tm.iota_per_period, 0.02892783769228813, places=places)
        self.assertAlmostEqual(tm.iota, 0.02892783769228813, places=places)
        M = np.array([[ 0.983527264227946, -0.026329866716251],
                      [ 1.240952748303061, 0.983527264237678]])
        np.testing.assert_allclose(tm.mat, M, atol=1.0e-6)
        self.assertAlmostEqual(tm.residue, 0.008236367883594053, places=places)
        
if __name__ == "__main__":
    unittest.main()
