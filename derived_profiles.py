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

def NFW_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return +Infinity

def NFW_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 2*np.sqrt(-(np.pi*G*Rs**4*np.log(Rs) + (np.pi*G*Rs**3*np.log(Rs) + np.pi*G*Rs**3)*r - (np.pi*G*Rs**4 + np.pi*G*Rs**3*r)*np.log(Rs + r))*rho0/(Rs*r + r**2))

def NFW_potential_inf(r, rho0, Rs, G=4.301e-6):
    # Potential (wo ref. pt.)
    return -4*(np.pi*G*Rs**3*np.log(Rs + r) - np.pi*G*Rs**3*np.log(Rs))*rho0/r

def NFW_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return -4*(np.pi*G*Rs**3*np.log(Rs + r) - np.pi*G*Rs**3*np.log(Rs) - np.pi*G*Rs**2*r)*rho0/r

# ================= superNFW =================
def superNFW_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return Rs*rho0/(r*(r/Rs + 1)**5)

def superNFW_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 1/3*(6*np.pi*Rs**5*r**2 + 4*np.pi*Rs**4*r**3 + np.pi*Rs**3*r**4)*rho0/(Rs**4 + 4*Rs**3*r + 6*Rs**2*r**2 + 4*Rs*r**3 + r**4)

def superNFW_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return 1/3*np.pi*Rs**3*rho0

def superNFW_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt((6*np.pi*G*Rs**5*r + 4*np.pi*G*Rs**4*r**2 + np.pi*G*Rs**3*r**3)*rho0/(Rs**4 + 4*Rs**3*r + 6*Rs**2*r**2 + 4*Rs*r**3 + r**4))

def superNFW_potential_inf(r, rho0, Rs, G=4.301e-6):
    # Potential (wo ref. pt.)
    return -1/3*(3*np.pi*G*Rs**5 + 3*np.pi*G*Rs**4*r + np.pi*G*Rs**3*r**2)*rho0/(Rs**3 + 3*Rs**2*r + 3*Rs*r**2 + r**3)

def superNFW_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return 1/3*(6*np.pi*G*Rs**4*r + 8*np.pi*G*Rs**3*r**2 + 3*np.pi*G*Rs**2*r**3)*rho0/(Rs**3 + 3*Rs**2*r + 3*Rs*r**2 + r**3)

def superNFW_sigma2(r, rho0, Rs, G=4.301e-6):
    # Radial Velocity Dispersion
    return -1/840*(4421*np.pi*G*Rs**8*r + 23048*np.pi*G*Rs**7*r**2 + 57288*np.pi*G*Rs**6*r**3 + 83216*np.pi*G*Rs**5*r**4 + 74620*np.pi*G*Rs**4*r**5 + 40880*np.pi*G*Rs**3*r**6 + 12600*np.pi*G*Rs**2*r**7 + 1680*np.pi*G*Rs*r**8 - 1680*(np.pi*G*Rs**8*r + 8*np.pi*G*Rs**7*r**2 + 28*np.pi*G*Rs**6*r**3 + 56*np.pi*G*Rs**5*r**4 + 70*np.pi*G*Rs**4*r**5 + 56*np.pi*G*Rs**3*r**6 + 28*np.pi*G*Rs**2*r**7 + 8*np.pi*G*Rs*r**8 + np.pi*G*r**9)*np.log(Rs + r) + 1680*(np.pi*G*Rs**8*r + 8*np.pi*G*Rs**7*r**2 + 28*np.pi*G*Rs**6*r**3 + 56*np.pi*G*Rs**5*r**4 + 70*np.pi*G*Rs**4*r**5 + 56*np.pi*G*Rs**3*r**6 + 28*np.pi*G*Rs**2*r**7 + 8*np.pi*G*Rs*r**8 + np.pi*G*r**9)*np.log(r))*rho0/(Rs**7 + 3*Rs**6*r + 3*Rs**5*r**2 + Rs**4*r**3)

# ================= pISO =================
def pISO_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0/(r**2/Rs**2 + 1)

def pISO_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -4*(np.pi*Rs**3*arcnp.tan(r/Rs) - np.pi*Rs**2*r)*rho0

def pISO_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return +Infinity

def pISO_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 2*np.sqrt(-(np.pi*G*Rs**3*arcnp.tan(r/Rs) - np.pi*G*Rs**2*r)*rho0/r)

def pISO_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return 2*(2*np.pi*G*Rs**3*arcnp.tan(r/Rs) + np.pi*G*Rs**2*r*np.log(Rs**2 + r**2) - 2*(np.pi*G*Rs**2*np.log(Rs) + np.pi*G*Rs**2)*r)*rho0/r

def pISO_sigma2(r, rho0, Rs, G=4.301e-6):
    # Radial Velocity Dispersion
    return 1/2*(np.pi**3*G*Rs**2*r + np.pi**3*G*r**3 - 4*(np.pi*G*Rs**2*r + np.pi*G*r**3)*arcnp.tan(r/Rs)**2 - 8*(np.pi*G*Rs**3 + np.pi*G*Rs*r**2)*arcnp.tan(r/Rs))*rho0/r

# ================= pISO1 =================
def pISO1_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return 1/(r**2 + 1)

def pISO1_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 4*np.pi*r - 4*np.pi*arcnp.tan(r)

def pISO1_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return +Infinity

def pISO1_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 2*np.sqrt((np.pi*G*r - np.pi*G*arcnp.tan(r))/r)

def pISO1_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return 2*(np.pi*G*r*np.log(r**2 + 1) - 2*np.pi*G*r + 2*np.pi*G*arcnp.tan(r))/r

def pISO1_sigma2(r, rho0, Rs, G=4.301e-6):
    # Radial Velocity Dispersion
    return 1/2*(np.pi**3*G*r**3 + np.pi**3*G*r - 4*(np.pi*G*r**3 + np.pi*G*r)*arcnp.tan(r)**2 - 8*(np.pi*G*r**2 + np.pi*G)*arcnp.tan(r))/r

# ================= Burkert =================
def Burkert_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return Rs**3*rho0/((Rs**2 + r**2)*(Rs + r))

def Burkert_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -(2*np.pi*Rs**3*arcnp.tan(r/Rs) - np.pi*Rs**3*np.log(Rs**2 + r**2) - 2*np.pi*Rs**3*np.log(Rs + r) + 4*np.pi*Rs**3*np.log(Rs))*rho0

def Burkert_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return +Infinity

def Burkert_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(-(2*np.pi*G*Rs**3*arcnp.tan(r/Rs) - np.pi*G*Rs**3*np.log(Rs**2 + r**2) - 2*np.pi*G*Rs**3*np.log(Rs + r) + 4*np.pi*G*Rs**3*np.log(Rs))*rho0/r)

def Burkert_potential_inf(r, rho0, Rs, G=4.301e-6):
    # Potential (wo ref. pt.)
    return -(np.pi**2*G*Rs**2*r - 4*np.pi*G*Rs**3*np.log(Rs) - 2*(np.pi*G*Rs**3 + np.pi*G*Rs**2*r)*arcnp.tan(r/Rs) + (np.pi*G*Rs**3 - np.pi*G*Rs**2*r)*np.log(Rs**2 + r**2) + 2*(np.pi*G*Rs**3 + np.pi*G*Rs**2*r)*np.log(Rs + r))*rho0/r

def Burkert_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return (4*np.pi*G*Rs**3*np.log(Rs) + 2*(np.pi*G*Rs**3 + np.pi*G*Rs**2*r)*arcnp.tan(r/Rs) - (np.pi*G*Rs**3 - np.pi*G*Rs**2*r)*np.log(Rs**2 + r**2) - 2*(np.pi*G*Rs**3 + np.pi*G*Rs**2*r)*np.log(Rs + r))*rho0/r

# ================= Lucky13 =================
def Lucky13_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0/(r/Rs + 1)**3

def Lucky13_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -2*(2*np.pi*Rs**5*np.log(Rs) + (2*np.pi*Rs**3*np.log(Rs) + 3*np.pi*Rs**3)*r**2 + 2*(2*np.pi*Rs**4*np.log(Rs) + np.pi*Rs**4)*r - 2*(np.pi*Rs**5 + 2*np.pi*Rs**4*r + np.pi*Rs**3*r**2)*np.log(Rs + r))*rho0/(Rs**2 + 2*Rs*r + r**2)

def Lucky13_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return +Infinity

def Lucky13_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-(2*np.pi*G*Rs**5*np.log(Rs) + (2*np.pi*G*Rs**3*np.log(Rs) + 3*np.pi*G*Rs**3)*r**2 + 2*(2*np.pi*G*Rs**4*np.log(Rs) + np.pi*G*Rs**4)*r - 2*(np.pi*G*Rs**5 + 2*np.pi*G*Rs**4*r + np.pi*G*Rs**3*r**2)*np.log(Rs + r))*rho0/(Rs**2*r + 2*Rs*r**2 + r**3))

def Lucky13_potential_inf(r, rho0, Rs, G=4.301e-6):
    # Potential (wo ref. pt.)
    return 2*(2*np.pi*G*Rs**4*np.log(Rs) + (2*np.pi*G*Rs**3*np.log(Rs) + np.pi*G*Rs**3)*r - 2*(np.pi*G*Rs**4 + np.pi*G*Rs**3*r)*np.log(Rs + r))*rho0/(Rs*r + r**2)

def Lucky13_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return 2*(2*np.pi*G*Rs**4*np.log(Rs) + np.pi*G*Rs**2*r**2 + 2*(np.pi*G*Rs**3*np.log(Rs) + np.pi*G*Rs**3)*r - 2*(np.pi*G*Rs**4 + np.pi*G*Rs**3*r)*np.log(Rs + r))*rho0/(Rs*r + r**2)

# ================= coreEinasto =================
def coreEinasto_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*np.exp(-2*((r/Rs + rc/Rs)**alpha - 1)/alpha)

# ================= DiCintio =================
def DiCintio_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*((r/Rs)**(1/beta) + 1)**((alpha - scipy.special.gamma)*beta)/(r/Rs)**alpha

# ================= gNFW =================
def gNFW_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*(r/Rs + 1)**(scipy.special.gamma - 3)/(r/Rs)**scipy.special.gamma

# ================= Dekel-Zhao =================
def Dekel-Zhao_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*(np.sqrt(r/Rs) + 1)**(2*alpha - 7)/(r/Rs)**alpha

# ================= Exponential =================
def Exponential_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*np.exp(-r/Rs)

def Exponential_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 4*(2*np.pi*Rs**3*np.exp(r/Rs) - 2*np.pi*Rs**3 - 2*np.pi*Rs**2*r - np.pi*Rs*r**2)*rho0*np.exp(-r/Rs)

def Exponential_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return 8*np.pi*Rs**3*rho0

def Exponential_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 2*np.sqrt((2*np.pi*G*Rs**3*np.exp(r/Rs) - 2*np.pi*G*Rs**3 - 2*np.pi*G*Rs**2*r - np.pi*G*Rs*r**2)*rho0/r)*np.exp(-1/2*r/Rs)

def Exponential_potential_inf(r, rho0, Rs, G=4.301e-6):
    # Potential (wo ref. pt.)
    return -4*(2*np.pi*G*Rs**2*r*scipy.special.expi(-r/Rs)*np.exp(r/Rs) - np.pi*G*Rs**2*r - 2*(np.pi*G*Rs**2*r*scipy.special.gamma(-1, r/Rs) - np.pi*G*Rs**3)*np.exp(r/Rs))*rho0*np.exp(-r/Rs)/r

def Exponential_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return 4*(2*np.pi*G*Rs**3 + np.pi*G*Rs**2*r - (2*np.pi*G*Rs**3 - np.pi*G*Rs**2*r)*np.exp(r/Rs))*rho0*np.exp(-r/Rs)/r

def Exponential_sigma2(r, rho0, Rs, G=4.301e-6):
    # Radial Velocity Dispersion
    return 2*(4*np.pi*G*Rs**2*scipy.special.expi(-2*r/Rs)*np.exp(2*r/Rs) - 4*(2*np.pi*scipy.special.gamma(-1, 2*r/Rs) - np.pi*scipy.special.gamma(-1, r/Rs))*G*Rs**2*np.exp(2*r/Rs) - np.pi*G*Rs**2)*rho0*np.exp(-r/Rs)

# ================= Exponential1 =================
def Exponential1_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return 9.60000000000000*np.exp(-0.714285714285714*r)

def Exponential1_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return (-210.7392*np.pi - 53.76*np.pi*r**2 - 150.528*np.pi*r + 210.7392*np.pi*np.exp(5/7*r))*np.exp(-5/7*r)

def Exponential1_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return 210.7392*np.pi

def Exponential1_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt((-53.76*np.pi*G*r**2 - 150.528*np.pi*G*r + 210.7392*np.pi*G*np.exp(5/7*r) - 210.7392*np.pi*G)/r)*np.exp(-5/14*r)

def Exponential1_potential_inf(r, rho0, Rs, G=4.301e-6):
    # Potential (wo ref. pt.)
    return -(-75.264*np.pi*G*r + (210.7392*np.pi*G + (150.528*np.pi*G*scipy.special.expi(-5/7*r) - 150.528*np.pi*G*scipy.special.gamma(-1, 5/7*r))*r)*np.exp(5/7*r))*np.exp(-5/7*r)/r

def Exponential1_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return 1/2157614375*(510165393437*G*r + (510165393438*G*r - 1428463101625*G)*np.exp(5/7*r) + 1428463101625*G)*np.exp(-5/7*r)/r

def Exponential1_sigma2(r, rho0, Rs, G=4.301e-6):
    # Radial Velocity Dispersion
    return (150.528*np.pi*G*scipy.special.expi(-10/7*r) - 37.632*np.pi*G*np.exp(-1.428571428571429*r) + (-301.056*np.pi*scipy.special.gamma(-1, 10/7*r) + 150.528*np.pi*scipy.special.gamma(-1, 5/7*r))*G)*np.exp(0.7142857142857143*r)

# ================= Exponential2 =================
def Exponential2_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*np.exp(-r/(Rs_1 + Rs_2))

def Exponential2_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -4*(2*np.pi*Rs_1**3 + 6*np.pi*Rs_1**2*Rs_2 + 6*np.pi*Rs_1*Rs_2**2 + 2*np.pi*Rs_2**3 + (np.pi*Rs_1 + np.pi*Rs_2)*r**2 + 2*(np.pi*Rs_1**2 + 2*np.pi*Rs_1*Rs_2 + np.pi*Rs_2**2)*r - 2*(np.pi*Rs_1**3 + 3*np.pi*Rs_1**2*Rs_2 + 3*np.pi*Rs_1*Rs_2**2 + np.pi*Rs_2**3)*np.exp(r/(Rs_1 + Rs_2)))*rho0*np.exp(-r/(Rs_1 + Rs_2))

def Exponential2_enclosed_mass_limit(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass Limit
    return 8*(np.pi*Rs_1**3 + 3*np.pi*Rs_1**2*Rs_2 + 3*np.pi*Rs_1*Rs_2**2 + np.pi*Rs_2**3)*rho0

def Exponential2_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return 2*np.sqrt(-(2*np.pi*G*Rs_1**3 + 6*np.pi*G*Rs_1**2*Rs_2 + 6*np.pi*G*Rs_1*Rs_2**2 + 2*np.pi*G*Rs_2**3 + (np.pi*G*Rs_1 + np.pi*G*Rs_2)*r**2 + 2*(np.pi*G*Rs_1**2 + 2*np.pi*G*Rs_1*Rs_2 + np.pi*G*Rs_2**2)*r - 2*(np.pi*G*Rs_1**3 + 3*np.pi*G*Rs_1**2*Rs_2 + 3*np.pi*G*Rs_1*Rs_2**2 + np.pi*G*Rs_2**3)*np.exp(r/(Rs_1 + Rs_2)))*rho0*np.exp(-r/(Rs_1 + Rs_2))/r)

def Exponential2_potential_inf(r, rho0, Rs, G=4.301e-6):
    # Potential (wo ref. pt.)
    return -4*(2*(np.pi*G*Rs_1**2 + 2*np.pi*G*Rs_1*Rs_2 + np.pi*G*Rs_2**2)*r*scipy.special.expi(-r/(Rs_1 + Rs_2))*np.exp(r/(Rs_1 + Rs_2)) - (np.pi*G*Rs_1**2 + 2*np.pi*G*Rs_1*Rs_2 + np.pi*G*Rs_2**2)*r + 2*(np.pi*G*Rs_1**3 + 3*np.pi*G*Rs_1**2*Rs_2 + 3*np.pi*G*Rs_1*Rs_2**2 + np.pi*G*Rs_2**3 - (np.pi*G*Rs_1**2*scipy.special.gamma(-1, r/(Rs_1 + Rs_2)) + 2*np.pi*G*Rs_1*Rs_2*scipy.special.gamma(-1, r/(Rs_1 + Rs_2)) + np.pi*G*Rs_2**2*scipy.special.gamma(-1, r/(Rs_1 + Rs_2)))*r)*np.exp(r/(Rs_1 + Rs_2)))*rho0*np.exp(-r/(Rs_1 + Rs_2))/r

def Exponential2_potential_zero(r, rho0, Rs, G=4.301e-6):
    # Potential
    return 4*(2*np.pi*G*Rs_1**3 + 6*np.pi*G*Rs_1**2*Rs_2 + 6*np.pi*G*Rs_1*Rs_2**2 + 2*np.pi*G*Rs_2**3 + (np.pi*G*Rs_1**2 + 2*np.pi*G*Rs_1*Rs_2 + np.pi*G*Rs_2**2)*r - (2*np.pi*G*Rs_1**3 + 6*np.pi*G*Rs_1**2*Rs_2 + 6*np.pi*G*Rs_1*Rs_2**2 + 2*np.pi*G*Rs_2**3 - (np.pi*G*Rs_1**2 + 2*np.pi*G*Rs_1*Rs_2 + np.pi*G*Rs_2**2)*r)*np.exp(r/(Rs_1 + Rs_2)))*rho0*np.exp(-r/(Rs_1 + Rs_2))/r

def Exponential2_sigma2(r, rho0, Rs, G=4.301e-6):
    # Radial Velocity Dispersion
    return -2*(np.pi*G*Rs_1**2 + 2*np.pi*G*Rs_1*Rs_2 + np.pi*G*Rs_2**2 - 4*(np.pi*G*Rs_1**2 + 2*np.pi*G*Rs_1*Rs_2 + np.pi*G*Rs_2**2)*scipy.special.expi(-2*r/(Rs_1 + Rs_2))*np.exp(2*r/(Rs_1 + Rs_2)) + 4*((2*np.pi*scipy.special.gamma(-1, 2*r/(Rs_1 + Rs_2)) - np.pi*scipy.special.gamma(-1, r/(Rs_1 + Rs_2)))*G*Rs_1**2 + 2*(2*np.pi*scipy.special.gamma(-1, 2*r/(Rs_1 + Rs_2)) - np.pi*scipy.special.gamma(-1, r/(Rs_1 + Rs_2)))*G*Rs_1*Rs_2 + (2*np.pi*scipy.special.gamma(-1, 2*r/(Rs_1 + Rs_2)) - np.pi*scipy.special.gamma(-1, r/(Rs_1 + Rs_2)))*G*Rs_2**2)*np.exp(2*r/(Rs_1 + Rs_2)))*rho0*np.exp(-r/(Rs_1 + Rs_2))

