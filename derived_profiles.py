import numpy as np
import scipy.special

# Helper wrapper for Sage compatibility
def dilog(x):
    return scipy.special.spence(1 - x)

# ================= PhySO_0 =================
def PhySO_0_density(r, rho0, Rs, G):
    # Density
    return (-c0 + 1.00000000000000*rs0**2/r**2)**c0*rho0

# ================= PhySO_1 =================
def PhySO_1_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0*e**(-np.sqrt(r/rs0))/r

def PhySO_1_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 8*(6*np.pi*c0*rho0*rs0**3*e**(np.sqrt(r/rs0)) - 3*np.pi*c0*r*rho0*rs0**2 - (np.pi*c0*rho0*(r/rs0)**(3/2) + 6*np.pi*c0*rho0*np.sqrt(r/rs0) + 6*np.pi*c0*rho0)*rs0**3)*e**(-np.sqrt(r/rs0))

def PhySO_1_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(2)*np.sqrt((6*np.pi*G*c0*rho0*rs0**3*e**(np.sqrt(r/rs0)) - 3*np.pi*G*c0*r*rho0*rs0**2 - (np.pi*G*c0*rho0*(r/rs0)**(3/2) + 6*np.pi*G*c0*rho0*np.sqrt(r/rs0) + 6*np.pi*G*c0*rho0)*rs0**3)/r)*e**(-1/2*np.sqrt(r/rs0))

def PhySO_1_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -32*(np.pi*scipy.special.gamma(-1, 2*np.sqrt(r/rs0)) + 6*np.pi*scipy.special.gamma(-2, 2*np.sqrt(r/rs0)) + 24*np.pi*scipy.special.gamma(-3, 2*np.sqrt(r/rs0)) + 48*np.pi*scipy.special.gamma(-4, 2*np.sqrt(r/rs0)) - 3*np.pi*scipy.special.gamma(-4, np.sqrt(r/rs0)))*G*c0*r*rho0*rs0*e**(np.sqrt(r/rs0))

def PhySO_1_potential(r, rho0, Rs, G):
    # Potential
    return 16*(np.pi*G*c0*r*rho0*rs0**2 - 3*(np.pi*G*c0*rho0*rs0**3 + (np.pi*G*c0*r*rho0*scipy.special.expi(-np.sqrt(r/rs0)) - 2*(np.pi*scipy.special.gamma(-1, np.sqrt(r/rs0)) + np.pi*scipy.special.gamma(-2, np.sqrt(r/rs0)))*G*c0*r*rho0)*rs0**2)*e**(np.sqrt(r/rs0)))*e**(-np.sqrt(r/rs0))/r

# ================= PhySO_2 =================
def PhySO_2_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*e**(-c0)/r**2 + rho0

def PhySO_2_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*r**3*rho0*e**c0 - 3*np.pi*r*rho0*rs0**2)*e**(-c0)

def PhySO_2_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0*e**c0 - 3*np.pi*G*rho0*rs0**2)*e**(-1/2*c0)

# ================= PhySO_3 =================
def PhySO_3_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*c0*rho0**1.00000000000000*rs0**2/r**2 - rho0

def PhySO_3_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_3_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_4 =================
def PhySO_4_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*rho0*rs0*e**(c0 - np.sqrt(-r/rs0))/r

def PhySO_4_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -(48.0*np.pi*rho0*rs0**3*e**(c0 + np.sqrt(-r/rs0)) + 24.0*np.pi*r*rho0*rs0**2*e**c0 + (-8.0*np.pi*rho0*(-r/rs0)**(3/2)*e**c0 - 48.0*np.pi*rho0*np.sqrt(-r/rs0)*e**c0 - 48.0*np.pi*rho0*e**c0)*rs0**3)*e**(-np.sqrt(-r/rs0))

def PhySO_4_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(-(48.0*np.pi*G*rho0*rs0**3*e**(c0 + np.sqrt(-r/rs0)) + 24.0*np.pi*G*r*rho0*rs0**2*e**c0 + (-8.0*np.pi*G*rho0*(-r/rs0)**(3/2)*e**c0 - 48.0*np.pi*G*rho0*np.sqrt(-r/rs0)*e**c0 - 48.0*np.pi*G*rho0*e**c0)*rs0**3)*e**(-np.sqrt(-r/rs0))/r)

def PhySO_4_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return (32.0*np.pi*scipy.special.gamma(-1, 2*np.sqrt(-r/rs0)) + 192.0*np.pi*scipy.special.gamma(-2, 2*np.sqrt(-r/rs0)) + 768.0*np.pi*scipy.special.gamma(-3, 2*np.sqrt(-r/rs0)) + 1536.0*np.pi*scipy.special.gamma(-4, 2*np.sqrt(-r/rs0)) - 96.0*np.pi*scipy.special.gamma(-4, np.sqrt(-r/rs0)))*G*r*rho0*rs0*e**(c0 + np.sqrt(-r/rs0))

# ================= PhySO_5 =================
def PhySO_5_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rs0**2/(r**2*(c0/rho0)**1.00000000000000)

def PhySO_5_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rs0**2/(c0/rho0)**1.0

def PhySO_5_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rs0**2/(c0/rho0)**1.0)

# ================= PhySO_6 =================
def PhySO_6_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/(c0*r**2)

def PhySO_6_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_6_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_7 =================
def PhySO_7_density(r, rho0, Rs, G):
    # Density
    return rho0 - rho0*rs0**2/(c0*r**2)

def PhySO_7_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_7_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_8 =================
def PhySO_8_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rho0*rs0**2/(c0*r**2)

def PhySO_8_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_8_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_9 =================
def PhySO_9_density(r, rho0, Rs, G):
    # Density
    return -rho0 + c0*rho0*rs0/(r**2*(r/rs0**2 - 1/rs0))

def PhySO_9_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*rho0*rs0**3*np.log(rs0) + 4*np.pi*c0*rho0*rs0**3*np.log(abs(-r + rs0)) - 4/3*np.pi*r**3*rho0

def PhySO_9_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(3*np.pi*G*c0*rho0*rs0**3*np.log(rs0) - 3*np.pi*G*c0*rho0*rs0**3*np.log(abs(-r + rs0)) + np.pi*G*r**3*rho0)/r)

def PhySO_9_surface_density(r, rho0, Rs, G):
    # Surface Density
    return +Infinity

# ================= PhySO_10 =================
def PhySO_10_density(r, rho0, Rs, G):
    # Density
    return -(c0*rs0**2/r - r)*rho0/r

def PhySO_10_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_10_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_11 =================
def PhySO_11_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*c0*rho0*rs0**2/r**2 + rho0

def PhySO_11_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_11_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_12 =================
def PhySO_12_density(r, rho0, Rs, G):
    # Density
    return -rho0 - (c0 - 1)*rho0*rs0**2/(c0*r**2)

def PhySO_12_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*(np.pi - np.pi*c0)*r*rho0*rs0**2)/c0

def PhySO_12_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 + 3*(np.pi*G*c0 - np.pi*G)*rho0*rs0**2)/c0)

# ================= PhySO_13 =================
def PhySO_13_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_13_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_13_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_14 =================
def PhySO_14_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_14_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_14_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_15 =================
def PhySO_15_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*c0*rho0*rs0**2/r**2 + rho0

def PhySO_15_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_15_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_16 =================
def PhySO_16_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0/(r/rs0)**(1.00000000000000*c0)

def PhySO_16_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*((3*np.pi - np.pi*c0)*r**3*rho0*(r/rs0)**c0 - 3*np.pi*r**3*rho0)/((c0 - 3)*(r/rs0)**c0)

def PhySO_16_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(3*np.pi*G*r**2*rho0 + (np.pi*G*c0 - 3*np.pi*G)*r**2*rho0*(r/rs0)**c0)/(c0 - 3))/(r/rs0)**(1/2*c0)

# ================= PhySO_17 =================
def PhySO_17_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0**2*e**(c0*r/rs0)/r**2 + rho0

def PhySO_17_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0*r**3*rho0 + 3*np.pi*rho0*rs0**3*e**(c0*r/rs0) - 3*np.pi*rho0*rs0**3)/c0

def PhySO_17_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((np.pi*G*c0*r**3*rho0 + 3*np.pi*G*rho0*rs0**3*e**(c0*r/rs0) - 3*np.pi*G*rho0*rs0**3)/(c0*r))

# ================= PhySO_18 =================
def PhySO_18_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0**2/r**2 - 0.632120558828558)

def PhySO_18_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -71418824/84737187*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_18_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/28245729*np.sqrt(9415243)*np.sqrt(-17854706*np.pi*G*r**2*rho0 + 84737187*np.pi*G*rho0*rs0**2)

# ================= PhySO_19 =================
def PhySO_19_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_19_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_19_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_20 =================
def PhySO_20_density(r, rho0, Rs, G):
    # Density
    return (c0 + rs0/r)*c0*rho0*rs0/r - rho0

def PhySO_20_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0**2*r**2*rho0*rs0 + 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_20_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(3*np.pi*G*c0**2*r*rho0*rs0 + 6*np.pi*G*c0*rho0*rs0**2 - 2*np.pi*G*r**2*rho0)

# ================= PhySO_21 =================
def PhySO_21_density(r, rho0, Rs, G):
    # Density
    return (c0*rho0/rs0 + rho0*rs0/(c0*r**2))*rs0 - rho0

def PhySO_21_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*((np.pi*c0**2 - np.pi*c0)*r**3*rho0 + 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_21_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((3*np.pi*G*rho0*rs0**2 + (np.pi*G*c0**2 - np.pi*G*c0)*r**2*rho0)/c0)

# ================= PhySO_22 =================
def PhySO_22_density(r, rho0, Rs, G):
    # Density
    return c0**2*(-1.00000000000000*r + 1.00000000000000*rs0**2/r)*rho0/r

def PhySO_22_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*c0**2*r**3*rho0 + 4*np.pi*c0**2*r*rho0*rs0**2

def PhySO_22_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*c0**2*r**2*rho0 + 3*np.pi*G*c0**2*rho0*rs0**2)

# ================= PhySO_23 =================
def PhySO_23_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*e**(rs0/((c0*rs0 + r)*c0)) - rho0

def PhySO_23_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2/3*(2*np.pi*c0**2*r**3*rho0 + (np.pi + 6*np.pi*c0**4 - 6*np.pi*c0**2)*rho0*rs0**3*scipy.special.expi(rs0/(c0**2*rs0 + c0*r)) - (6*np.pi*c0**4*scipy.special.expi(c0**(-2)) - 6*np.pi*c0**2*scipy.special.expi(c0**(-2)) + np.pi*scipy.special.expi(c0**(-2)) - (2*np.pi*c0**6 - 5*np.pi*c0**4 + np.pi*c0**2)*e**(c0**(-2)))*rho0*rs0**3 - (2*np.pi*c0**3*r**3*rho0 + np.pi*c0**2*r**2*rho0*rs0 - (4*np.pi*c0**3 - np.pi*c0)*r*rho0*rs0**2 + (2*np.pi*c0**6 - 5*np.pi*c0**4 + np.pi*c0**2)*rho0*rs0**3)*e**(rs0/(c0**2*rs0 + c0*r)))/c0**2

def PhySO_23_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(-(2*np.pi*G*c0**2*r**3*rho0 + (6*np.pi*G*c0**4 - 6*np.pi*G*c0**2 + np.pi*G)*rho0*rs0**3*scipy.special.expi(rs0/(c0**2*rs0 + c0*r)) - (6*np.pi*G*c0**4*scipy.special.expi(c0**(-2)) - 6*np.pi*G*c0**2*scipy.special.expi(c0**(-2)) + np.pi*G*scipy.special.expi(c0**(-2)) - (2*np.pi*G*c0**6 - 5*np.pi*G*c0**4 + np.pi*G*c0**2)*e**(c0**(-2)))*rho0*rs0**3 - (2*np.pi*G*c0**3*r**3*rho0 + np.pi*G*c0**2*r**2*rho0*rs0 - (4*np.pi*G*c0**3 - np.pi*G*c0)*r*rho0*rs0**2 + (2*np.pi*G*c0**6 - 5*np.pi*G*c0**4 + np.pi*G*c0**2)*rho0*rs0**3)*e**(rs0/(c0**2*rs0 + c0*r)))/(c0**2*r))

# ================= PhySO_24 =================
def PhySO_24_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0*e**(c0*rs0/(r + rs0))/r

def PhySO_24_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*(np.pi*c0**2 - 2*np.pi*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(r + rs0)) + 2*((np.pi*c0**2 - 2*np.pi*c0)*scipy.special.expi(c0) + (np.pi - np.pi*c0)*e**c0)*rho0*rs0**3 + 2*(np.pi*c0*r*rho0*rs0**2 + np.pi*r**2*rho0*rs0 - (np.pi - np.pi*c0)*rho0*rs0**3)*e**(c0*rs0/(r + rs0))

def PhySO_24_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-((np.pi*G*c0**2 - 2*np.pi*G*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(r + rs0)) - ((np.pi*G*c0**2 - 2*np.pi*G*c0)*scipy.special.expi(c0) - (np.pi*G*c0 - np.pi*G)*e**c0)*rho0*rs0**3 - (np.pi*G*c0*r*rho0*rs0**2 + np.pi*G*r**2*rho0*rs0 + (np.pi*G*c0 - np.pi*G)*rho0*rs0**3)*e**(c0*rs0/(r + rs0)))/r)

# ================= PhySO_25 =================
def PhySO_25_density(r, rho0, Rs, G):
    # Density
    return rho0**1.00000000000000*rs0**2*e**(-2*r/rs0)/(c0**2*r**2)

def PhySO_25_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*rho0*rs0**3*e**(2*r/rs0) - np.pi*rho0*rs0**3)*e**(-2*r/rs0)/c0**2

def PhySO_25_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*rho0*rs0**3*e**(2*r/rs0) - np.pi*G*rho0*rs0**3)/r)*e**(-r/rs0)/c0

def PhySO_25_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -16*(8*np.pi*scipy.special.gamma(-3, 4*r/rs0) - np.pi*scipy.special.gamma(-3, 2*r/rs0))*G*r**2*rho0*e**(2*r/rs0)/c0**2

def PhySO_25_potential(r, rho0, Rs, G):
    # Potential
    return 2*(2*np.pi*G*r*rho0*rs0**2*scipy.special.gamma(-1, 2*r/rs0) - np.pi*G*rho0*rs0**3)/(c0**2*r)

# ================= PhySO_26 =================
def PhySO_26_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0**2*e**(-r/(c0*rs0))/r**2

def PhySO_26_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*(np.pi*c0*rho0*rs0**3*e**(r/(c0*rs0)) - np.pi*c0*rho0*rs0**3)*e**(-r/(c0*rs0))

def PhySO_26_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt((np.pi*G*c0*rho0*rs0**3*e**(r/(c0*rs0)) - np.pi*G*c0*rho0*rs0**3)/r)*e**(-1/2*r/(c0*rs0))

def PhySO_26_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -4*(8*np.pi*scipy.special.gamma(-3, 2*r/(c0*rs0)) - np.pi*scipy.special.gamma(-3, r/(c0*rs0)))*G*r**2*rho0*e**(r/(c0*rs0))/c0**2

def PhySO_26_potential(r, rho0, Rs, G):
    # Potential
    return -4*(np.pi*G*c0*rho0*rs0**3 - np.pi*G*r*rho0*rs0**2*scipy.special.gamma(-1, r/(c0*rs0)))/r

# ================= PhySO_27 =================
def PhySO_27_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0**2*e**(-c0*r/rs0)/r**2

def PhySO_27_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*(np.pi*rho0*rs0**3*e**(c0*r/rs0) - np.pi*rho0*rs0**3)*e**(-c0*r/rs0)/c0

def PhySO_27_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt((np.pi*G*rho0*rs0**3*e**(c0*r/rs0) - np.pi*G*rho0*rs0**3)/(c0*r))*e**(-1/2*c0*r/rs0)

def PhySO_27_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -4*(8*np.pi*scipy.special.gamma(-3, 2*c0*r/rs0) - np.pi*scipy.special.gamma(-3, c0*r/rs0))*G*c0**2*r**2*rho0*e**(c0*r/rs0)

def PhySO_27_potential(r, rho0, Rs, G):
    # Potential
    return 4*(np.pi*G*c0*r*rho0*rs0**2*scipy.special.gamma(-1, c0*r/rs0) - np.pi*G*rho0*rs0**3)/(c0*r)

# ================= PhySO_28 =================
def PhySO_28_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*e**(c0*r/rs0)/r**2

def PhySO_28_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*(np.pi*rho0*rs0**3*e**(c0*r/rs0) - np.pi*rho0*rs0**3)/c0

def PhySO_28_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-(np.pi*G*rho0*rs0**3*e**(c0*r/rs0) - np.pi*G*rho0*rs0**3)/(c0*r))

# ================= PhySO_29 =================
def PhySO_29_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*e**(-r/rs0)/r**2

def PhySO_29_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*(np.pi*rho0*rs0**3*e**(r/rs0) - np.pi*rho0*rs0**3)*e**(-r/rs0)

def PhySO_29_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-(np.pi*G*rho0*rs0**3*e**(r/rs0) - np.pi*G*rho0*rs0**3)*e**(-r/rs0)/r)

def PhySO_29_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return 4*(8*np.pi*scipy.special.gamma(-3, 2*r/rs0) - np.pi*scipy.special.gamma(-3, r/rs0))*G*r**2*rho0*e**(r/rs0)

def PhySO_29_potential(r, rho0, Rs, G):
    # Potential
    return -4*(np.pi*G*r*rho0*rs0**2*scipy.special.gamma(-1, r/rs0) - np.pi*G*rho0*rs0**3)/r

# ================= PhySO_30 =================
def PhySO_30_density(r, rho0, Rs, G):
    # Density
    return (c0 + 1.00000000000000*rs0/r)*rho0*rs0/r + rho0

def PhySO_30_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 + 4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_30_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(3*np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*r**2*rho0 + 6*np.pi*G*rho0*rs0**2)

# ================= PhySO_31 =================
def PhySO_31_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(2*np.log(c0*(1.00000000000000*r - 1.00000000000000*rs0)/r) + 2.00000000000000)

def PhySO_31_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 9.8520747985742*np.pi*c0**2*r**3*rho0 - 29.5562243957226*np.pi*c0**2*r**2*rho0*rs0 + 29.5562243957226*np.pi*c0**2*r*rho0*rs0**2

def PhySO_31_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(9.8520747985742*np.pi*G*c0**2*r**2*rho0 - 29.5562243957226*np.pi*G*c0**2*r*rho0*rs0 + 29.5562243957226*np.pi*G*c0**2*rho0*rs0**2)

# ================= PhySO_32 =================
def PhySO_32_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0/r - 1.00000000000000)**2

def PhySO_32_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_32_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*r*rho0*rs0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_33 =================
def PhySO_33_density(r, rho0, Rs, G):
    # Density
    return (1.00000000000000*c0**2*rs0 - 1.00000000000000*r)**2*rho0**1.00000000000000/r**2

def PhySO_33_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4.0*np.pi*c0**4*r*rho0*rs0**2 - 4.0*np.pi*c0**2*r**2*rho0*rs0 + 1.333333333333333*np.pi*r**3*rho0

def PhySO_33_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(4.0*np.pi*G*c0**4*rho0*rs0**2 - 4.0*np.pi*G*c0**2*r*rho0*rs0 + 1.333333333333333*np.pi*G*r**2*rho0)

# ================= PhySO_34 =================
def PhySO_34_density(r, rho0, Rs, G):
    # Density
    return (c0*rho0*rs0/r - rho0)**2/rho0**1.00000000000000

def PhySO_34_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0**2*r*rho0*rs0**2 - 4*np.pi*c0*r**2*rho0*rs0 + 4/3*np.pi*r**3*rho0

def PhySO_34_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0**2*rho0*rs0**2 - 3*np.pi*G*c0*r*rho0*rs0 + np.pi*G*r**2*rho0)

# ================= PhySO_35 =================
def PhySO_35_density(r, rho0, Rs, G):
    # Density
    return ((c0 - e**2)*r - rs0)**2*rho0/r**2

def PhySO_35_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0**2 - 2*np.pi*c0*e**2 + np.pi*e**4)*r**3*rho0 - 4*(np.pi*c0 - np.pi*e**2)*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_35_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*rho0*rs0**2 + (np.pi*G*c0**2 - 2*np.pi*G*c0*e**2 + np.pi*G*e**4)*r**2*rho0 - 3*(np.pi*G*c0 - np.pi*G*e**2)*r*rho0*rs0)

# ================= PhySO_36 =================
def PhySO_36_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0/r - e**(0.500000000000000*c0 - 0.500000000000000))**2

def PhySO_36_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 0.4905059215619232*np.pi*r**3*rho0*e**(1.0*c0) - 2.426122638850534*np.pi*r**2*rho0*rs0*e**(0.5*c0) + 4.0*np.pi*r*rho0*rs0**2

def PhySO_36_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(0.4905059215619232*np.pi*G*r**2*rho0*e**(1.0*c0) - 2.426122638850534*np.pi*G*r*rho0*rs0*e**(0.5*c0) + 4.0*np.pi*G*rho0*rs0**2)

# ================= PhySO_37 =================
def PhySO_37_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0 - rs0**2/r)/r

def PhySO_37_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*r**2*rho0*rs0 - 4*np.pi*r*rho0*rs0**2

def PhySO_37_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_38 =================
def PhySO_38_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*(1.00000000000000/r - 1/rs0)/r

def PhySO_38_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*r**2*rho0*rs0 - 4*np.pi*r*rho0*rs0**2

def PhySO_38_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_39 =================
def PhySO_39_density(r, rho0, Rs, G):
    # Density
    return (c0**2*rs0/r - c0)*rho0*rs0/r

def PhySO_39_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0**2*r*rho0*rs0**2 - 2*np.pi*c0*r**2*rho0*rs0

def PhySO_39_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(2*np.pi*G*c0**2*rho0*rs0**2 - np.pi*G*c0*r*rho0*rs0)

# ================= PhySO_40 =================
def PhySO_40_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*(r - rs0)*rho0*rs0/(c0*r**2)

def PhySO_40_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return (2.0*np.pi*r**2*rho0*rs0 - 4.0*np.pi*r*rho0*rs0**2)/c0

def PhySO_40_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt((2.0*np.pi*G*r*rho0*rs0 - 4.0*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_41 =================
def PhySO_41_density(r, rho0, Rs, G):
    # Density
    return (c0 + rs0/r)*rho0*rs0/(c0*r)

def PhySO_41_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*c0*r**2*rho0*rs0 + 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_41_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_42 =================
def PhySO_42_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0*(-1.00000000000000*rs0/r + 1.00000000000000)/(c0*r)

def PhySO_42_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*r**2*rho0*rs0 - 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_42_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_43 =================
def PhySO_43_density(r, rho0, Rs, G):
    # Density
    return (r - rs0)*rho0*rs0/(c0*r**2)

def PhySO_43_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*r**2*rho0*rs0 - 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_43_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_44 =================
def PhySO_44_density(r, rho0, Rs, G):
    # Density
    return -(c0 - rs0/r)*rho0*rs0/r

def PhySO_44_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_44_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_45 =================
def PhySO_45_density(r, rho0, Rs, G):
    # Density
    return -(c0*r - rs0)*c0*rho0*rs0*e**(-c0)/r**2

def PhySO_45_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*(np.pi*c0**2*r**2*rho0*rs0 - 2*np.pi*c0*r*rho0*rs0**2)*e**(-c0)

def PhySO_45_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-(np.pi*G*c0**2*r*rho0*rs0 - 2*np.pi*G*c0*rho0*rs0**2)*e**(-c0))

# ================= PhySO_46 =================
def PhySO_46_density(r, rho0, Rs, G):
    # Density
    return c0*(r + rs0/c0)*rho0*rs0/r**2

def PhySO_46_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_46_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_47 =================
def PhySO_47_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0*np.log(c0*e**(2.00000000000000*rs0/r))/r

def PhySO_47_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4272943/680060*r**2*rho0*rs0*np.log(c0) - 4272943/170015*r*rho0*rs0**2

def PhySO_47_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/340030*np.sqrt(4272943)*np.sqrt(170015)*np.sqrt(-G*r*rho0*rs0*np.log(c0) - 4*G*rho0*rs0**2)

def PhySO_47_surface_density(r, rho0, Rs, G):
    # Surface Density
    return +Infinity

# ================= PhySO_48 =================
def PhySO_48_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*(c0 - rs0/r)**(1/c0)*rho0

def PhySO_48_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4.0*np.pi*c0*r**(-1/c0 + 3)*rho0*rs0**(1/c0)*e**(I*np.pi/c0)*hypergeometric((-1/c0, (3*c0 - 1)/c0), ((4*c0 - 1)/c0,), c0*r/rs0)/(3*c0 - 1)

def PhySO_48_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2.0*np.sqrt(np.pi*G*c0*r**(-1/c0 + 2)*rho0*rs0**(1/c0)*hypergeometric((-1/c0, (3*c0 - 1)/c0), ((4*c0 - 1)/c0,), c0*r/rs0)/(3*c0 - 1))*e**(1/2*I*np.pi/c0)

# ================= PhySO_49 =================
def PhySO_49_density(r, rho0, Rs, G):
    # Density
    return (c0 - 1.00000000000000*c0*rs0/r)**(1/c0)*rho0

