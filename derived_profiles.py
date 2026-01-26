import numpy as np
import scipy.special

# Helper wrapper for Sage compatibility
def dilog(x):
    return scipy.special.spence(1 - x)

# ================= NFW =================
def NFW_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return Rs*rho0/(r*(r/Rs + 1)**2)

def NFW_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -4*(np.pi*Rs**4*np.log(Rs) + (np.pi*Rs**3*np.log(Rs) + np.pi*Rs**3)*r - (np.pi*Rs**4 + np.pi*Rs**3*r)*np.log(Rs + r))*rho0/(Rs + r)

def NFW_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 2*np.sqrt(-(np.pi*G*Rs**4*np.log(Rs) + (np.pi*G*Rs**3*np.log(Rs) + np.pi*G*Rs**3)*r - (np.pi*G*Rs**4 + np.pi*G*Rs**3*r)*np.log(Rs + r))*rho0/(Rs*r + r**2))

def NFW_potential(r, rho0, Rs, G=4.301e-6):
    # Potential
    return -4*(np.pi*G*Rs**3*np.log(Rs + r) - np.pi*G*Rs**3*np.log(Rs))*rho0/r

# ================= superNFW =================
def superNFW_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return Rs*rho0/(r*(r/Rs + 1)**5)

def superNFW_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 1/3*(6*np.pi*Rs**5*r**2 + 4*np.pi*Rs**4*r**3 + np.pi*Rs**3*r**4)*rho0/(Rs**4 + 4*Rs**3*r + 6*Rs**2*r**2 + 4*Rs*r**3 + r**4)

def superNFW_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt((6*np.pi*G*Rs**5*r + 4*np.pi*G*Rs**4*r**2 + np.pi*G*Rs**3*r**3)*rho0/(Rs**4 + 4*Rs**3*r + 6*Rs**2*r**2 + 4*Rs*r**3 + r**4))

def superNFW_potential(r, rho0, Rs, G=4.301e-6):
    # Potential
    return -1/3*(3*np.pi*G*Rs**5 + 3*np.pi*G*Rs**4*r + np.pi*G*Rs**3*r**2)*rho0/(Rs**3 + 3*Rs**2*r + 3*Rs*r**2 + r**3)

def superNFW_sigma2(r, rho0, Rs, G=4.301e-6):
    # Radial Velocity Dispersion
    return -1/840*(4421*np.pi*G*Rs**8*r + 23048*np.pi*G*Rs**7*r**2 + 57288*np.pi*G*Rs**6*r**3 + 83216*np.pi*G*Rs**5*r**4 + 74620*np.pi*G*Rs**4*r**5 + 40880*np.pi*G*Rs**3*r**6 + 12600*np.pi*G*Rs**2*r**7 + 1680*np.pi*G*Rs*r**8 - 1680*(np.pi*G*Rs**8*r + 8*np.pi*G*Rs**7*r**2 + 28*np.pi*G*Rs**6*r**3 + 56*np.pi*G*Rs**5*r**4 + 70*np.pi*G*Rs**4*r**5 + 56*np.pi*G*Rs**3*r**6 + 28*np.pi*G*Rs**2*r**7 + 8*np.pi*G*Rs*r**8 + np.pi*G*r**9)*np.log(Rs + r) + 1680*(np.pi*G*Rs**8*r + 8*np.pi*G*Rs**7*r**2 + 28*np.pi*G*Rs**6*r**3 + 56*np.pi*G*Rs**5*r**4 + 70*np.pi*G*Rs**4*r**5 + 56*np.pi*G*Rs**3*r**6 + 28*np.pi*G*Rs**2*r**7 + 8*np.pi*G*Rs*r**8 + np.pi*G*r**9)*np.log(r))*rho0/(Rs**7 + 3*Rs**6*r + 3*Rs**5*r**2 + Rs**4*r**3)

