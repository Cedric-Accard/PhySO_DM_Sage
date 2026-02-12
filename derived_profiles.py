import numpy as np
import scipy.special

# Helper wrapper for Sage compatibility
def dilog(x):
    return scipy.special.spence(1 - x)

# ================= PhySO_0 =================
def PhySO_0_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return 1.00000000000000*rho0*rs0*(rs0/r - 1/c0)/r

def PhySO_0_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return (4.0*np.pi*c0*r*rho0*rs0**2 - 2.0*np.pi*r**2*rho0*rs0)/c0

def PhySO_0_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt((4.0*np.pi*G*c0*rho0*rs0**2 - 2.0*np.pi*G*r*rho0*rs0)/c0)

# ================= PhySO_1 =================
def PhySO_1_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return c0**2*(rho0 - r*rho0/rs0)*rs0**2/r**2

def PhySO_1_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -2*np.pi*c0**2*r**2*rho0*rs0 + 4*np.pi*c0**2*r*rho0*rs0**2

def PhySO_1_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0**2*r*rho0*rs0 + 2*np.pi*G*c0**2*rho0*rs0**2)

# ================= PhySO_2 =================
def PhySO_2_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -(c0**2*rs0**2/r - rs0)*rho0/r

def PhySO_2_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -4*np.pi*c0**2*r*rho0*rs0**2 + 2*np.pi*r**2*rho0*rs0

def PhySO_2_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-2*np.pi*G*c0**2*rho0*rs0**2 + np.pi*G*r*rho0*rs0)

# ================= PhySO_3 =================
def PhySO_3_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return (c0*rs0 + 1.00000000000000*rs0**2/r)*rho0/r

def PhySO_3_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_3_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_4 =================
def PhySO_4_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return (c0 - rs0/(c0*r))*rho0*rs0/r

def PhySO_4_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*(np.pi*c0**2*r**2*rho0*rs0 - 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_4_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*c0**2*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_5 =================
def PhySO_5_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return (c0 - rs0/r)*rho0*rs0/r

def PhySO_5_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 - 4*np.pi*r*rho0*rs0**2

def PhySO_5_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_6 =================
def PhySO_6_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -(c0*rho0/rs0 - rho0/r)*rs0**2/r

def PhySO_6_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_6_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_7 =================
def PhySO_7_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -(c0*rs0*e/r - c0)*rho0*rs0/r

def PhySO_7_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2*e + 2*np.pi*c0*r**2*rho0*rs0

def PhySO_7_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-2*np.pi*G*c0*rho0*rs0**2*e + np.pi*G*c0*r*rho0*rs0)

# ================= PhySO_8 =================
def PhySO_8_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return (c0 + 1.00000000000000*rs0/r)*rho0*rs0/r

def PhySO_8_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_8_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_9 =================
def PhySO_9_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*rs0*(rs0/(c0*r) + 1)/r

def PhySO_9_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*(np.pi*c0*r**2*rho0*rs0 + 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_9_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_10 =================
def PhySO_10_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -(r - rs0)*rho0*rs0/(c0*r**2)

def PhySO_10_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -2*(np.pi*r**2*rho0*rs0 - 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_10_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-(np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_11 =================
def PhySO_11_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return c0*rho0*(rs0 - rs0**2/r)/r

def PhySO_11_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 - 4*np.pi*c0*r*rho0*rs0**2

def PhySO_11_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 - 2*np.pi*G*c0*rho0*rs0**2)

# ================= PhySO_12 =================
def PhySO_12_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -(c0 - rs0/r)*rho0*rs0/r

def PhySO_12_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_12_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_13 =================
def PhySO_13_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -c0**2*rs0**2*(rho0/r - rho0/rs0)/r

def PhySO_13_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*np.pi*c0**2*r**2*rho0*rs0 - 4*np.pi*c0**2*r*rho0*rs0**2

def PhySO_13_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0**2*r*rho0*rs0 - 2*np.pi*G*c0**2*rho0*rs0**2)

# ================= PhySO_14 =================
def PhySO_14_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -c0*rho0*(rs0 - rs0**2/r)/r

def PhySO_14_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*c0*r*rho0*rs0**2

def PhySO_14_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*c0*rho0*rs0**2)

# ================= PhySO_15 =================
def PhySO_15_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return rho0*(rs0**2/r - rs0/c0)/(c0*r)

def PhySO_15_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*(2*np.pi*c0*r*rho0*rs0**2 - np.pi*r**2*rho0*rs0)/c0**2

def PhySO_15_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(2*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r*rho0*rs0)/c0

# ================= PhySO_16 =================
def PhySO_16_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -(-1.00000000000000*r*np.np.exp(-c0) + rs0)*c0*rho0*rs0/r**2

def PhySO_16_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -(4*np.pi*c0*r*rho0*rs0**2*np.expc0 - 2.0*np.pi*c0*r**2*rho0*rs0)*np.np.exp(-c0)

def PhySO_16_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(-(4*np.pi*G*c0*rho0*rs0**2*np.expc0 - 2.0*np.pi*G*c0*r*rho0*rs0)*np.np.exp(-c0))

# ================= PhySO_17 =================
def PhySO_17_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return -1.00000000000000*(c0 + rs0/r)*rho0*rs0/r

def PhySO_17_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return -2.0*np.pi*c0*r**2*rho0*rs0 - 4.0*np.pi*r*rho0*rs0**2

def PhySO_17_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(-2.0*np.pi*G*c0*r*rho0*rs0 - 4.0*np.pi*G*rho0*rs0**2)

# ================= PhySO_18 =================
def PhySO_18_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return (c0 + rs0/r)*rho0*rs0/r

def PhySO_18_mass(r, rho0, Rs, G=4.301e-6):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_18_circular_velocity(r, rho0, Rs, G=4.301e-6):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_19 =================
def PhySO_19_density(r, rho0, Rs, G=4.301e-6):
    # Density
    return 1.00000000000000*rho0*np.np.exp(0.500000000000000*c0 + 0.500000000000000*np.log(rs0/r - 1.00000000000000)/c0)

