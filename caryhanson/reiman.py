#!/usr/bin/env python3

"""
This module contains a class for the Reiman-Greensides model magnetic field.
For derivation of expressions here, see
20200917-05 Reiman field.docx
"""

# These next 2 lines tell jax to use double precision:
#from jax.config import config
#config.update("jax_enable_x64", True)

#import jax.numpy as jnp
import numpy as jnp
import numpy as np
#from jax import grad, jit
from .field import Field

#@jit
def B_func(R_phi_Z, iotaj, eps, B0, R0, ms):
    """
    This function computes B in a form that is convenient for automatic differentiation.
    The arguments iotaj and eps should be numpy arrays.
    """
    R = R_phi_Z[0]
    phi = R_phi_Z[1]
    Z = R_phi_Z[2]
    theta = jnp.arctan2(Z, R - R0)

    iotaja = np.array(iotaj)
    epsa = np.array(eps)
    msa = np.array(ms)

    r = jnp.sqrt(Z * Z + (R - R0) * (R - R0))
    psi = 0.5 * B0 * r * r
    jarr = jnp.arange(len(iotaj)) + 1

    temp = ms * epsa * (r ** (msa - 2))
    Q0 = -jnp.sum(temp * jnp.cos(msa * theta - phi)) + B0 * jnp.sum(iotaja * jarr * (psi ** (jarr - 1)))
    Q1 =  jnp.sum(temp * jnp.sin(msa * theta - phi))

    BR = (Z * Q0 + (R - R0) * Q1) / R
    Bphi = -B0
    BZ = (Z * Q1 - (R - R0) * Q0) / R
    return np.array([BR, Bphi, BZ])

#mygrad = grad(B_func)

class ReimanField(Field):
    """
    This class represents a magnetic field generated by one or more
    helical coils.
    """
    def __init__(self, R0=1, B0=1, iotaj=[0.15, 0.38], eps=[0.01], ms=[6]):
        assert len(eps) == len(ms)
        self.R0 = R0
        self.B0 = B0
        self.iotaj = np.array(iotaj)
        self.eps = np.array(eps)
        self.ms = np.array(ms)
        self.nfp = 1        

    def BR_Bphi_BZ(self, R, phi, Z):
        """
        Return the cylindrical components of the magnetic field at a specified point.
        This subroutine only works for single points as arguments.
        """

        return B_func([R, phi, Z], self.iotaj, self.eps, self.B0, self.R0, self.ms)

    def grad_B(self, R, phi, Z):
        """
        Return the derivatives with respect to (R, phi, Z) of the cylindrical components of B.
        """
        R0 = self.R0
        B0 = self.B0
        theta = np.arctan2(Z, R - R0)

        r = np.sqrt(Z * Z + (R - R0) * (R - R0))
        psi = 0.5 * B0 * r * r
        jarr = np.arange(len(self.iotaj)) + 1

        temp = self.ms * self.eps * (r ** (self.ms - 2))
        sines = np.sin(self.ms * theta - phi)
        coses = np.cos(self.ms * theta - phi)
        cos_terms = np.sum(temp * coses)
        Q0 = -cos_terms + B0 * np.sum(self.iotaj * jarr * (psi ** (jarr - 1)))
        Q1 = np.sum(temp * sines)

        d_Q0_d_phi = -Q1
        d_Q1_d_phi = -cos_terms
        
        temp = self.ms * self.eps * (r ** (self.ms - 4))
        m2_cos_terms = np.sum(temp * (self.ms - 2) * coses)
        m_cos_terms  = np.sum(temp * (self.ms    ) * coses)
        m2_sin_terms = np.sum(temp * (self.ms - 2) * sines)
        m_sin_terms  = np.sum(temp * (self.ms    ) * sines)

        iota_terms = self.iotaj * jarr * (jarr - 1) * (psi ** (jarr - 2))
        if len(self.iotaj) > 0:
            iota_terms[0] = 0
        #if len(self.iotaj) > 1:
        #    iota_terms[1] = 0
        iota_terms_sum = np.sum(iota_terms)
        #print('iota_terms_sum: ', iota_terms_sum)
        d_Q1_d_R = (R - R0) * m2_sin_terms - Z * m_cos_terms
        d_Q1_d_Z = Z * m2_sin_terms + (R - R0) * m_cos_terms
        d_Q0_d_R = -((R - R0) * m2_cos_terms + Z * m_sin_terms) + B0 * B0 * (R - R0) * iota_terms_sum
        d_Q0_d_Z = -(Z * m2_cos_terms - (R - R0) * m_sin_terms) + B0 * B0 * Z * iota_terms_sum
        
        #print('d_Q0_d_R an:', d_Q0_d_R)
        #print('d_Q1_d_R an:', d_Q1_d_R)
        #print('d_Q0_d_Z an:', d_Q0_d_Z)
        #print('d_Q1_d_Z an:', d_Q1_d_Z)
        #BR = (Z * Q0 + (R - R0) * Q1) / R
        #Bphi = -B0
        #BZ = (Z * Q1 - (R - R0) * Q0) / R

        d_BR_d_R = -Z * Q0 / (R * R) + (Z / R) * d_Q0_d_R + R0 * Q1 / (R * R) + (R - R0) * d_Q1_d_R / R
        d_BR_d_phi = (Z * d_Q0_d_phi + (R - R0) * d_Q1_d_phi) / R
        d_BR_d_Z = (Q0 + Z * d_Q0_d_Z + (R - R0) * d_Q1_d_Z) / R
        
        d_Bphi_d_R = 0
        d_Bphi_d_phi = 0
        d_Bphi_d_Z = 0

        d_BZ_d_R = -R0 * Q0 / (R * R) - (R - R0) * d_Q0_d_R / R - Z * Q1 / (R * R) + Z * d_Q1_d_R / R
        d_BZ_d_phi = (-(R - R0) * d_Q0_d_phi + Z * d_Q1_d_phi) / R
        d_BZ_d_Z = (-(R - R0) * d_Q0_d_Z + Q1 + Z * d_Q1_d_Z) / R
        
        return np.array([[d_BR_d_R, d_BR_d_phi, d_BR_d_Z],
                         [d_Bphi_d_R, d_Bphi_d_phi, d_Bphi_d_Z],
                         [d_BZ_d_R, d_BZ_d_phi, d_BZ_d_Z]])

    def Q0_Q1(self, R, phi, Z):
        """
        This subroutine is only for testing. It returns the intermediate expressions Q0 and Q1.
        """
        R0 = self.R0
        B0 = self.B0
        theta = np.arctan2(Z, R - R0)

        r = np.sqrt(Z * Z + (R - R0) * (R - R0))
        psi = 0.5 * B0 * r * r
        jarr = np.arange(len(self.iotaj)) + 1

        temp = self.ms * self.eps * (r ** (self.ms - 2))
        sines = np.sin(self.ms * theta - phi)
        coses = np.cos(self.ms * theta - phi)
        cos_terms = np.sum(temp * coses)
        Q0 = -cos_terms + B0 * np.sum(self.iotaj * jarr * (psi ** (jarr - 1)))
        Q1 = np.sum(temp * sines)
        return Q0, Q1
