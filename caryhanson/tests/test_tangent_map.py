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
        pfl = periodic_field_line(field, 31, periods=1, R0=0.983)
        #tm = tangent_map(field, R0=0.98343287162797)
        tm = tangent_map(field, pfl)
        places = 8
        self.assertAlmostEqual(tm.iota_per_period, 0.11287146743203362, places=places)
        self.assertAlmostEqual(tm.iota, 0.5643573371601681, places=places)
        M = np.array([[ 0.75888781375709 ,  1.37679244594353 ],
                      [-0.308026489831821,  0.758887765057904]])
        np.testing.assert_allclose(tm.full_orbit_tangent_maps[0], M, atol=1.0e-6)
        np.testing.assert_allclose(tm.single_period_tangent_maps[0], M, atol=1.0e-6)
        self.assertAlmostEqual(tm.residue, 0.12055610529625149, places=places)
        
    def test_reiman_axis(self):
        """
        Tests properties of the tangent map for the magnetic axis of the
        Reiman model field.
        """

        field = ReimanField()
        pfl = periodic_field_line(field, 11, periods=1, R0=1.0)
        #tm = tangent_map(field, R0=1.0, rtol=1e-12, atol=1e-12)
        tm = tangent_map(field, pfl, rtol=1e-12, atol=1e-12)
        places = 10
        self.assertAlmostEqual(tm.iota_per_period, 0.15, places=places)
        self.assertAlmostEqual(tm.iota, 0.15, places=places)
        M = np.array([[ 0.587785252292183, -0.809016994374573],
                      [ 0.809016994374573, 0.587785252292183]])
        np.testing.assert_allclose(tm.full_orbit_tangent_maps[0], M, atol=1.0e-6)
        np.testing.assert_allclose(tm.single_period_tangent_maps[0], M, atol=1.0e-6)
        self.assertAlmostEqual(tm.residue, 0.20610737385390826, places=places)
        
    def test_reiman_Opoint(self):
        """
        Tests properties of the tangent map for the O-point of the
        Reiman model field.
        """

        field = ReimanField()
        pfl = periodic_field_line(field, 31, periods=6, R0=1.2)
        #tm = tangent_map(field, R0=1.210161051261806, periods=6, rtol=1e-12, atol=1e-12)
        tm = tangent_map(field, pfl, rtol=1e-12, atol=1e-12)
        places = 10
        self.assertAlmostEqual(tm.iota_per_period, 0.02892783769228813, places=places)
        self.assertAlmostEqual(tm.iota, 0.02892783769228813, places=places)
        M = np.array([[ 0.983527264227946, -0.026329866716251],
                      [ 1.240952748303061, 0.983527264237678]])
        np.testing.assert_allclose(tm.full_orbit_tangent_maps[0], M, atol=1.0e-6)
        self.assertAlmostEqual(tm.residue, 0.008236367883594053, places=places)
        
    def test_CaryHanson1986_page2469(self):
        """
        Compare to some numbers quoted on page 2469 of
        Cary & Hanson (1986), for a helical coil with straight
        winding law.
        """

        field = HelicalCoil(I=[0.0307,-0.0307])

        # First consider the magnetic axis.
        # Cary & Hanson quote the axis position as 0.9822, but I get a bit higher, 0.9833
        pfl = periodic_field_line(field, 31, R0=1.0)
        tm = tangent_map(field, pfl)
        #tm = tangent_map(field, R0=0.983393787292938)
        places = 8
        # Cary & Hanson quote iota_per_period = 0.324. I get a bit lower.
        self.assertAlmostEqual(tm.iota_per_period, 0.3102585112979056, places=places)
        
if __name__ == "__main__":
    unittest.main()
