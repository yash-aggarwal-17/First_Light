### A simple file to store constants and definitions

import numpy as np
import scipy
import matplotlib.pyplot as plt

"""Basic constants"""

h_planck = 4.135667696e-15 # eV s
cs = 29979245800 # cm /s 
rydberg_energy = 13.6056923 # eV
lyAlpha_energy = 10.198810993784 # eV
kB = 8.617343e-5 # eV / K
G = 6.674e-8 # dyne / cm^2 / g^2
pi = 3.14159265
e = 2.718281828
sigma_th = 6.65245873e-25 # Thomson cross section in cm^2
rec = 2.81794003e-13 # classical electron radius in cm

"""Define masses"""

mp = 1.67262171e-24 # grams
mp_eV = 938.272029e+6 # eV

me = 9.1093827e-28 # grams
me_eV = 5.10998918e+5 # eV

"""Conversions"""

pc_to_cm = 3.0856776e+18 # cm
msol_to_grams = 1.9891e+33 # grams
eV_to_ergs = 1.60217653e-12 # ergs
GeV_To_InvSec = 1.52e+24 # sec

"""Black body spectra"""

def n_BB (Tr, en):
    return 8*pi*en**2 / (h_planck * cs)**3 / (np.exp(en/(kB * Tr)) -1 ) # number / eV /cm^3

def u_BB (Tr):
    return 8*pi**5 / (15 * (h_planck*cs)**3) * (kB*Tr)**4 # eV / cm^3

"""Cosmology""" 

# In acccordance to Planck 2018
h=0.674
H0=67.4/(pc_to_cm*10)
Omega_c = 0.12 / h**2
Omega_b = 0.0224 / h**2
Omega_m = Omega_c + Omega_b
Omega_lambda = 1 - Omega_m
Yp = 0.245

# Critical density
RhoCrit = 3*H0**2/(8*pi*G) * cs**2/eV_to_ergs # eV cm^-3

# Define Critical overdensity
Delta_c = 200

# Define Hydrogen number density
nH0 = RhoCrit*Omega_b*(1 - Yp)/mp_eV

# Define Helium number density
nHe0 = RhoCrit*Omega_b*(Yp)/mp_eV

# Define total number of atoms density
nA0 = nH0 + nHe0

## Define Hubble function

def Hubble(z):
    return H0*(Omega_lambda + Omega_m*(1+z)**3)**(1/2)

# Define IGM, CMB, and virial temperatures

def T_IGM(z):
    return 1/40 * (1+z)**2

def T_CMB(z):
    return 2.725*(1+z)

def Tvir(z, mhalo):

    fac = (
            1.98e+4 * (1.22/0.6) *
            (Omega_m / Omega_t_m(z) * Delta_c / (18*pi**2) )**(1/3) *
            (mhalo/ 10**8)**(2/3) *
            (1+z)/10 * h**(2/3)
          )
    return fac


"""Define cosmological density parameters at time t"""

def Omega_t_m(z):
    return Omega_m*(1+z)**3 / (Hubble(z)/H0)**2

def Omega_t_lambda(z):
    return Omega_lambda / (Hubble(z)/H0)**2


"""Define growth factor"""

def growth_fac(z):
    
    # Define the constant denominator
    const = Omega_m*( Omega_m**(4/7) - Omega_lambda + (1 + Omega_m/2)*(1 + Omega_lambda/70) )**(-1)

    func = Omega_t_m(z)*( Omega_t_m(z)**(4/7) - Omega_t_lambda(z) + (1 + Omega_t_m(z)/2)*(1 + Omega_t_lambda(z)/70) )**(-1)

    return (func/const/(1+z))


"""
Collisional rate coefficients. units cm^3 / s
"""

# case B Hydrogen Recombination
def case_B(T):
    return 2.54e-13*(T/10**4)**(-0.8163)
    
# Electron attachment to H. Taken from Hirata 2006. Valid for T<=10^(4) K
def C_Hminus(T):
    return 3e-16*(T/300)**0.95 * e**(-T/9320)

# H2 formation via H minus
def C_H2(T):
    return 1.5e-9 * (T/300)**(-0.1)

"""
Define Lyman line energies
"""
def lyman_np_level(n):
    return rydberg_energy*(1 - 1/n**2)

"""
Define cross section
"""
def sigma_Hm(en):
    en = np.asanyarray(en)

    result = np.zeros_like(en, dtype=float)

    val = 7.928e+5 * h_planck**1.5
    
    mask = en>=0.755
    result[mask] = val*(en[mask] - 0.755)**1.5 / en[mask]**3
    
    return result

