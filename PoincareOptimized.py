#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('/Users/mattland/CaryHanson')
from caryhanson import HelicalCoil, poincare

hc = HelicalCoil.optimized()
r0 = np.linspace(0.955, 1.2, 16)
npoints=200
poincare(hc, r0, r0*0, npoints=npoints, rmin=0.72, zmin=-0.23, zmax=0.23,
         pdf=True,
         extra_str='npoints{}'.format(npoints))
