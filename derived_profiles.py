import numpy as np
import scipy.special

# Helper wrapper for Sage compatibility
def dilog(x):
    return scipy.special.spence(1 - x)

# ================= PhySO_0 =================
def PhySO_0_density(r, rho0, Rs, G):
    # Density
    return -c0*rho0*(r/rs0)**c0

def PhySO_0_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r**(c0 + 3)*rho0/((c0 + 3)*rs0**c0)

def PhySO_0_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-np.pi*G*c0*r**(c0 + 2)*rho0/((c0 + 3)*rs0**c0))

# ================= PhySO_1 =================
def PhySO_1_density(r, rho0, Rs, G):
    # Density
    return rho0/(r/(c0*rs0))**(1/c0)

def PhySO_1_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r**3*rho0/((3*c0 - 1)*(r/(c0*rs0))**(1/c0))

def PhySO_1_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(np.pi*G*c0*rho0/(3*c0 - 1))*r/(r/(c0*rs0))**(1/2/c0)

# ================= PhySO_2 =================
def PhySO_2_density(r, rho0, Rs, G):
    # Density
    return -rho0/(-r/rs0)**(1.00000000000000*c0)

def PhySO_2_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*r**3*rho0/((c0 - 3)*(-r/rs0)**c0)

def PhySO_2_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*r*np.sqrt(np.pi*G*rho0/((c0 - 3)*(-r/rs0)**c0))

# ================= PhySO_3 =================
def PhySO_3_density(r, rho0, Rs, G):
    # Density
    return -rho0*(-rs0/r)**c0

# ================= PhySO_4 =================
def PhySO_4_density(r, rho0, Rs, G):
    # Density
    return rho0/(c0*rs0/r)**(1.00000000000000*c0)

# ================= PhySO_5 =================
def PhySO_5_density(r, rho0, Rs, G):
    # Density
    return -rho0*(-1.00000000000000*c0*r/rs0)**c0

def PhySO_5_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*r**3*rho0*(-c0*r/rs0)**c0/(c0 + 3)

def PhySO_5_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-np.pi*G*r**2*rho0*(-c0*r/rs0)**c0/(c0 + 3))

# ================= PhySO_6 =================
def PhySO_6_density(r, rho0, Rs, G):
    # Density
    return -rho0*(-c0*r/rs0)**c0

def PhySO_6_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*r**3*rho0*(-c0*r/rs0)**c0/(c0 + 3)

def PhySO_6_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-np.pi*G*r**2*rho0*(-c0*r/rs0)**c0/(c0 + 3))

# ================= PhySO_7 =================
def PhySO_7_density(r, rho0, Rs, G):
    # Density
    return -rho0/((c0 + 1)*r/rs0)**(1/c0)

def PhySO_7_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r**3*rho0/((3*c0 - 1)*((c0 + 1)*r/rs0)**(1/c0))

def PhySO_7_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-np.pi*G*c0*r**2*rho0/((3*c0 - 1)*((c0 + 1)*r/rs0)**(1/c0)))

# ================= PhySO_8 =================
def PhySO_8_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0/r)**(c0**(-2))

# ================= PhySO_9 =================
def PhySO_9_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0/r)**(1/c0)

# ================= PhySO_10 =================
def PhySO_10_density(r, rho0, Rs, G):
    # Density
    return -rho0*e**(-c0 + np.log(r/rs0)/c0)

def PhySO_10_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r**3*rho0*(r/rs0)**(1/c0)*e**(-c0)/(3*c0 + 1)

def PhySO_10_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-np.pi*G*c0*r**2*rho0*(r/rs0)**(1/c0)*e**(-c0)/(3*c0 + 1))

# ================= PhySO_11 =================
def PhySO_11_density(r, rho0, Rs, G):
    # Density
    return -rho0/(2*r/rs0)**(1/c0)

def PhySO_11_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r**3*rho0/((3*c0 - 1)*(2*r/rs0)**(1/c0))

def PhySO_11_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-np.pi*G*c0*r**2*rho0/((3*c0 - 1)*(2*r/rs0)**(1/c0)))

# ================= PhySO_12 =================
def PhySO_12_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(-c0*np.log(1.00000000000000*r/rs0) - c0)

# ================= PhySO_13 =================
def PhySO_13_density(r, rho0, Rs, G):
    # Density
    return rho0*(1.00000000000000*(c0**2*rs0 + 1.00000000000000*rs0)/r)**(1.00000000000000*c0)

# ================= PhySO_14 =================
def PhySO_14_density(r, rho0, Rs, G):
    # Density
    return rho0*(c0*rs0/r)**c0

# ================= PhySO_15 =================
def PhySO_15_density(r, rho0, Rs, G):
    # Density
    return rho0*(-c0*rs0/r)**c0/c0

# ================= PhySO_16 =================
def PhySO_16_density(r, rho0, Rs, G):
    # Density
    return rho0*(1.00000000000000*r/(c0*rs0))**c0

def PhySO_16_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*r**(c0 + 3)*rho0/((c0 + 3)*c0**c0*rs0**c0)

def PhySO_16_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(np.pi*G*r**(c0 + 2)*rho0/(c0 + 3))/(c0**(1/2*c0)*rs0**(1/2*c0))

# ================= PhySO_17 =================
def PhySO_17_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(-c0*np.log(-r/rs0) + c0)

def PhySO_17_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*r**3*rho0*e**c0/((c0 - 3)*(-r/rs0)**c0)

def PhySO_17_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-np.pi*G*r**2*rho0*e**c0/((c0 - 3)*(-r/rs0)**c0))

# ================= PhySO_18 =================
def PhySO_18_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(c0*(np.log(rs0/r) + 1.00000000000000))

# ================= PhySO_19 =================
def PhySO_19_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*rho0*rs0*(1.00000000000000*rs0/r)**(1/c0)/r

# ================= PhySO_20 =================
def PhySO_20_density(r, rho0, Rs, G):
    # Density
    return rho0*(r/rs0)**(1.00000000000000*c0)/c0

def PhySO_20_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*r**(c0 + 3)*rho0/((c0**2 + 3*c0)*rs0**c0)

def PhySO_20_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(np.pi*G*r**(c0 + 2)*rho0/(c0**2 + 3*c0))/rs0**(1/2*c0)

# ================= PhySO_21 =================
def PhySO_21_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*rho0*(r/rs0)**(0.500000000000000*c0)

def PhySO_21_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -8.0*np.pi*r**(1/2*c0 + 3)*rho0/((c0 + 6)*rs0**(1/2*c0))

def PhySO_21_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2.82842712474619*I*np.sqrt(np.pi*G*r**(1/2*c0 + 2)*rho0/(c0 + 6))/rs0**(1/4*c0)

# ================= PhySO_22 =================
def PhySO_22_density(r, rho0, Rs, G):
    # Density
    return rho0*(-rs0/(c0*r))**c0

# ================= PhySO_23 =================
def PhySO_23_density(r, rho0, Rs, G):
    # Density
    return np.sqrt(rho0**2*(-rs0/r)**(1/c0))

# ================= PhySO_24 =================
def PhySO_24_density(r, rho0, Rs, G):
    # Density
    return rho0/(rs0/r)**c0

def PhySO_24_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*r**(c0 + 3)*rho0/((c0 + 3)*rs0**c0)

def PhySO_24_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(np.pi*G*r**(c0 + 2)*rho0/(c0 + 3))/rs0**(1/2*c0)

# ================= PhySO_25 =================
def PhySO_25_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*(1.00000000000000*c0*rs0/r)**c0

# ================= PhySO_26 =================
def PhySO_26_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(1/(c0 + r/rs0))

def PhySO_26_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2/3*(np.pi + 6*np.pi*c0**2 - 6*np.pi*c0)*rho0*rs0**3*scipy.special.expi(rs0/(c0*rs0 + r)) + 2/3*(6*np.pi*c0**2*scipy.special.expi(1/c0) - 6*np.pi*c0*scipy.special.expi(1/c0) + np.pi*scipy.special.expi(1/c0) - (2*np.pi*c0**3 - 5*np.pi*c0**2 + np.pi*c0)*e**(1/c0))*rho0*rs0**3 + 2/3*(2*np.pi*r**3*rho0 + np.pi*r**2*rho0*rs0 + (np.pi - 4*np.pi*c0)*r*rho0*rs0**2 + (2*np.pi*c0**3 - 5*np.pi*c0**2 + np.pi*c0)*rho0*rs0**3)*e**(rs0/(c0*rs0 + r))

def PhySO_26_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(-((6*np.pi*G*c0**2 - 6*np.pi*G*c0 + np.pi*G)*rho0*rs0**3*scipy.special.expi(rs0/(c0*rs0 + r)) - (6*np.pi*G*c0**2*scipy.special.expi(1/c0) - 6*np.pi*G*c0*scipy.special.expi(1/c0) + np.pi*G*scipy.special.expi(1/c0) - (2*np.pi*G*c0**3 - 5*np.pi*G*c0**2 + np.pi*G*c0)*e**(1/c0))*rho0*rs0**3 - (2*np.pi*G*r**3*rho0 + np.pi*G*r**2*rho0*rs0 - (4*np.pi*G*c0 - np.pi*G)*r*rho0*rs0**2 + (2*np.pi*G*c0**3 - 5*np.pi*G*c0**2 + np.pi*G*c0)*rho0*rs0**3)*e**(rs0/(c0*rs0 + r)))/r)

# ================= PhySO_27 =================
def PhySO_27_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*rho0*rs0*np.log(-r/rs0)**2/(c0**2*r)

def PhySO_27_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -(2*np.pi*r**2*rho0*np.log(-r/rs0)**2 - 2*np.pi*r**2*rho0*np.log(-r/rs0) + np.pi*r**2*rho0)*rs0/c0**2

def PhySO_27_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(-(2*np.pi*G*r*rho0*np.log(-r/rs0)**2 - 2*np.pi*G*r*rho0*np.log(-r/rs0) + np.pi*G*r*rho0)*rs0/c0**2)

# ================= PhySO_28 =================
def PhySO_28_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*c0*rho0*e**(1/np.log(1.00000000000000*c0 + r/rs0) - 1.00000000000000)

# ================= PhySO_29 =================
def PhySO_29_density(r, rho0, Rs, G):
    # Density
    return -rho0*e**(-c0**2/(c0 + rs0/r))

# ================= PhySO_30 =================
def PhySO_30_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(rs0**2/(1.00000000000000*r + 1.00000000000000*rs0/c0)**2)

# ================= PhySO_31 =================
def PhySO_31_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(-np.sqrt(c0**3*r/rs0) + rs0/r)

# ================= PhySO_32 =================
def PhySO_32_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(-1.00000000000000*c0**2*r/(r + rs0))

# ================= PhySO_33 =================
def PhySO_33_density(r, rho0, Rs, G):
    # Density
    return rho0*(e**(-c0*rs0/(r + rs0)) + 1.00000000000000)/c0

def PhySO_33_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4272943/2040180*((c0**3 + 6*c0**2 + 6*c0)*rho0*rs0**3*scipy.special.expi(-c0*rs0/(r + rs0))*e**(c0 + c0*rs0/(r + rs0)) - c0*r**2*rho0*rs0*e**c0 + (c0**2 + 4*c0)*r*rho0*rs0**2*e**c0 + (c0**2 + 5*c0 + 2)*rho0*rs0**3*e**c0 + 2*r**3*rho0*e**c0 - ((c0**2 + (c0**3*scipy.special.expi(-c0) + 6*c0**2*scipy.special.expi(-c0) + 6*c0*scipy.special.expi(-c0))*e**c0 + 5*c0 + 2)*rho0*rs0**3 - 2*r**3*rho0*e**c0)*e**(c0*rs0/(r + rs0)))*e**(-c0 - c0*rs0/(r + rs0))/c0

def PhySO_33_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/1020090*np.sqrt(4272943)*np.sqrt(510045)*np.sqrt(((G*c0**3 + 6*G*c0**2 + 6*G*c0)*rho0*rs0**3*scipy.special.expi(-c0*rs0/(r + rs0))*e**(c0 + c0*rs0/(r + rs0)) - G*c0*r**2*rho0*rs0*e**c0 + 2*G*r**3*rho0*e**c0 + (G*c0**2 + 4*G*c0)*r*rho0*rs0**2*e**c0 + (G*c0**2 + 5*G*c0 + 2*G)*rho0*rs0**3*e**c0 + (2*G*r**3*rho0*e**c0 - (G*c0**2 + 5*G*c0 + (G*c0**3*scipy.special.expi(-c0) + 6*G*c0**2*scipy.special.expi(-c0) + 6*G*c0*scipy.special.expi(-c0))*e**c0 + 2*G)*rho0*rs0**3)*e**(c0*rs0/(r + rs0)))/(c0*r))*e**(-1/2*c0 - 1/2*c0*rs0/(r + rs0))

# ================= PhySO_34 =================
def PhySO_34_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/r**2

def PhySO_34_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_34_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_35 =================
def PhySO_35_density(r, rho0, Rs, G):
    # Density
    return rho0 - rho0*rs0**2/r**2

def PhySO_35_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r*rho0*rs0**2

def PhySO_35_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_36 =================
def PhySO_36_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rho0*rs0**2/r**2

def PhySO_36_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_36_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_37 =================
def PhySO_37_density(r, rho0, Rs, G):
    # Density
    return rho0 - rho0*rs0**2/r**2

def PhySO_37_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r*rho0*rs0**2

def PhySO_37_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_38 =================
def PhySO_38_density(r, rho0, Rs, G):
    # Density
    return rho0 - rho0*rs0**2/r**2

def PhySO_38_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r*rho0*rs0**2

def PhySO_38_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_39 =================
def PhySO_39_density(r, rho0, Rs, G):
    # Density
    return rho0 - rho0*rs0**2/r**2

def PhySO_39_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r*rho0*rs0**2

def PhySO_39_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_40 =================
def PhySO_40_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/r**2

def PhySO_40_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_40_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_41 =================
def PhySO_41_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/r**2

def PhySO_41_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_41_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_42 =================
def PhySO_42_density(r, rho0, Rs, G):
    # Density
    return rho0 - 1.00000000000000*rho0*rs0**2/r**2

def PhySO_42_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r*rho0*rs0**2

def PhySO_42_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_43 =================
def PhySO_43_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*rho0*e**(rs0/(r + rs0/c0))

def PhySO_43_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -10838702/5175099*((c0**3 - 6*c0**2 + 6*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(c0*r + rs0)) - ((c0**3 - 6*c0**2 + 6*c0)*scipy.special.expi(c0) - (c0**2 - 5*c0 + 2)*e**c0)*rho0*rs0**3 - (2*c0**3*r**3*rho0 + c0**3*r**2*rho0*rs0 + (c0**3 - 4*c0**2)*r*rho0*rs0**2 + (c0**2 - 5*c0 + 2)*rho0*rs0**3)*e**(c0*rs0/(c0*r + rs0)))/c0**3

def PhySO_43_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 7/1725033*np.sqrt(575011)*np.sqrt(221198)*np.sqrt(-((G*c0**3 - 6*G*c0**2 + 6*G*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(c0*r + rs0)) - ((G*c0**3 - 6*G*c0**2 + 6*G*c0)*scipy.special.expi(c0) - (G*c0**2 - 5*G*c0 + 2*G)*e**c0)*rho0*rs0**3 - (2*G*c0**3*r**3*rho0 + G*c0**3*r**2*rho0*rs0 + (G*c0**3 - 4*G*c0**2)*r*rho0*rs0**2 + (G*c0**2 - 5*G*c0 + 2*G)*rho0*rs0**3)*e**(c0*rs0/(c0*r + rs0)))/(c0**3*r))

# ================= PhySO_44 =================
def PhySO_44_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(c0*rs0/(r + 1.00000000000000*rs0))

def PhySO_44_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2/3*(np.pi*c0**3 - 6*np.pi*c0**2 + 6*np.pi*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(r + rs0)) + 2/3*((np.pi*c0**3 - 6*np.pi*c0**2 + 6*np.pi*c0)*scipy.special.expi(c0) - (2*np.pi + np.pi*c0**2 - 5*np.pi*c0)*e**c0)*rho0*rs0**3 + 2/3*(np.pi*c0*r**2*rho0*rs0 + 2*np.pi*r**3*rho0 + (np.pi*c0**2 - 4*np.pi*c0)*r*rho0*rs0**2 + (2*np.pi + np.pi*c0**2 - 5*np.pi*c0)*rho0*rs0**3)*e**(c0*rs0/(r + rs0))

def PhySO_44_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(-((np.pi*G*c0**3 - 6*np.pi*G*c0**2 + 6*np.pi*G*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(r + rs0)) - ((np.pi*G*c0**3 - 6*np.pi*G*c0**2 + 6*np.pi*G*c0)*scipy.special.expi(c0) - (np.pi*G*c0**2 - 5*np.pi*G*c0 + 2*np.pi*G)*e**c0)*rho0*rs0**3 - (np.pi*G*c0*r**2*rho0*rs0 + 2*np.pi*G*r**3*rho0 + (np.pi*G*c0**2 - 4*np.pi*G*c0)*r*rho0*rs0**2 + (np.pi*G*c0**2 - 5*np.pi*G*c0 + 2*np.pi*G)*rho0*rs0**3)*e**(c0*rs0/(r + rs0)))/r)

# ================= PhySO_45 =================
def PhySO_45_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*rho0*e**(-c0*rs0/(2.00000000000000*r + 1.00000000000000*rs0))

def PhySO_45_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 5419351/20700396*((c0**3 + 6*c0**2 + 6*c0)*rho0*rs0**3*scipy.special.expi(-c0*rs0/(2*r + rs0))*e**(c0 + c0*rs0/(2*r + rs0)) - 4*c0*r**2*rho0*rs0*e**c0 + 2*(c0**2 + 4*c0)*r*rho0*rs0**2*e**c0 + (c0**2 + 5*c0 + 2)*rho0*rs0**3*e**c0 - (c0**2 + (c0**3*scipy.special.expi(-c0) + 6*c0**2*scipy.special.expi(-c0) + 6*c0*scipy.special.expi(-c0))*e**c0 + 5*c0 + 2)*rho0*rs0**3*e**(c0*rs0/(2*r + rs0)) + 16*r**3*rho0*e**c0)*e**(-c0 - c0*rs0/(2*r + rs0))

def PhySO_45_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 7/3450066*np.sqrt(575011)*np.sqrt(110599)*np.sqrt(((G*c0**3 + 6*G*c0**2 + 6*G*c0)*rho0*rs0**3*scipy.special.expi(-c0*rs0/(2*r + rs0))*e**(c0 + c0*rs0/(2*r + rs0)) - 4*G*c0*r**2*rho0*rs0*e**c0 + 16*G*r**3*rho0*e**c0 + 2*(G*c0**2 + 4*G*c0)*r*rho0*rs0**2*e**c0 + (G*c0**2 + 5*G*c0 + 2*G)*rho0*rs0**3*e**c0 - (G*c0**2 + 5*G*c0 + (G*c0**3*scipy.special.expi(-c0) + 6*G*c0**2*scipy.special.expi(-c0) + 6*G*c0*scipy.special.expi(-c0))*e**c0 + 2*G)*rho0*rs0**3*e**(c0*rs0/(2*r + rs0)))/r)*e**(-1/2*c0 - 1/2*c0*rs0/(2*r + rs0))

# ================= PhySO_46 =================
def PhySO_46_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0*e**(-np.sqrt(-r/rs0))/r

def PhySO_46_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 8*(6*np.pi*rho0*rs0**3*e**(np.sqrt(-r/rs0)) + 3*np.pi*r*rho0*rs0**2 - (np.pi*rho0*(-r/rs0)**(3/2) + 6*np.pi*rho0*np.sqrt(-r/rs0) + 6*np.pi*rho0)*rs0**3)*e**(-np.sqrt(-r/rs0))

def PhySO_46_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(2)*np.sqrt((6*np.pi*G*rho0*rs0**3*e**(np.sqrt(-r/rs0)) + 3*np.pi*G*r*rho0*rs0**2 - (np.pi*G*rho0*(-r/rs0)**(3/2) + 6*np.pi*G*rho0*np.sqrt(-r/rs0) + 6*np.pi*G*rho0)*rs0**3)/r)*e**(-1/2*np.sqrt(-r/rs0))

def PhySO_46_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -32*(np.pi*scipy.special.gamma(-1, 2*np.sqrt(-r/rs0)) + 6*np.pi*scipy.special.gamma(-2, 2*np.sqrt(-r/rs0)) + 24*np.pi*scipy.special.gamma(-3, 2*np.sqrt(-r/rs0)) + 48*np.pi*scipy.special.gamma(-4, 2*np.sqrt(-r/rs0)) - 3*np.pi*scipy.special.gamma(-4, np.sqrt(-r/rs0)))*G*r*rho0*rs0*e**(np.sqrt(-r/rs0))

# ================= PhySO_47 =================
def PhySO_47_density(r, rho0, Rs, G):
    # Density
    return -rho0*e**(-1.00000000000000*(c0 + np.log(-r**2/rs0**2))*c0)

def PhySO_47_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*r**3*rho0*e**(-c0**2)/((2*c0 - 3)*(-r**2/rs0**2)**c0)

def PhySO_47_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*r*np.sqrt(np.pi*G*rho0/((2*c0 - 3)*(-r**2/rs0**2)**c0))*e**(-1/2*c0**2)

# ================= PhySO_48 =================
def PhySO_48_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0*(rs0/r - r**2/(c0**4*rs0**2))/(c0*r)

def PhySO_48_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return (4*np.pi*c0**4*r*rho0*rs0**3 - np.pi*r**4*rho0)/(c0**5*rs0)

def PhySO_48_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt((4*np.pi*G*c0**4*rho0*rs0**3 - np.pi*G*r**3*rho0)/(c0**5*rs0))

# ================= PhySO_49 =================
def PhySO_49_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*(r - rs0)*rho0*rs0/r**2

def PhySO_49_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*r**2*rho0*rs0 - 4.0*np.pi*r*rho0*rs0**2

def PhySO_49_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2*np.pi*G*r*rho0*rs0 - 4.0*np.pi*G*rho0*rs0**2)

# ================= PhySO_50 =================
def PhySO_50_density(r, rho0, Rs, G):
    # Density
    return (-c0 + 1.00000000000000*rs0**2/r**2)**c0*rho0

# ================= PhySO_51 =================
def PhySO_51_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0*e**(-np.sqrt(r/rs0))/r

def PhySO_51_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 8*(6*np.pi*c0*rho0*rs0**3*e**(np.sqrt(r/rs0)) - 3*np.pi*c0*r*rho0*rs0**2 - (np.pi*c0*rho0*(r/rs0)**(3/2) + 6*np.pi*c0*rho0*np.sqrt(r/rs0) + 6*np.pi*c0*rho0)*rs0**3)*e**(-np.sqrt(r/rs0))

def PhySO_51_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(2)*np.sqrt((6*np.pi*G*c0*rho0*rs0**3*e**(np.sqrt(r/rs0)) - 3*np.pi*G*c0*r*rho0*rs0**2 - (np.pi*G*c0*rho0*(r/rs0)**(3/2) + 6*np.pi*G*c0*rho0*np.sqrt(r/rs0) + 6*np.pi*G*c0*rho0)*rs0**3)/r)*e**(-1/2*np.sqrt(r/rs0))

def PhySO_51_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -32*(np.pi*scipy.special.gamma(-1, 2*np.sqrt(r/rs0)) + 6*np.pi*scipy.special.gamma(-2, 2*np.sqrt(r/rs0)) + 24*np.pi*scipy.special.gamma(-3, 2*np.sqrt(r/rs0)) + 48*np.pi*scipy.special.gamma(-4, 2*np.sqrt(r/rs0)) - 3*np.pi*scipy.special.gamma(-4, np.sqrt(r/rs0)))*G*c0*r*rho0*rs0*e**(np.sqrt(r/rs0))

def PhySO_51_potential(r, rho0, Rs, G):
    # Potential
    return 16*(np.pi*G*c0*r*rho0*rs0**2 - 3*(np.pi*G*c0*rho0*rs0**3 + (np.pi*G*c0*r*rho0*scipy.special.expi(-np.sqrt(r/rs0)) - 2*(np.pi*scipy.special.gamma(-1, np.sqrt(r/rs0)) + np.pi*scipy.special.gamma(-2, np.sqrt(r/rs0)))*G*c0*r*rho0)*rs0**2)*e**(np.sqrt(r/rs0)))*e**(-np.sqrt(r/rs0))/r

# ================= PhySO_52 =================
def PhySO_52_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*e**(-c0)/r**2 + rho0

def PhySO_52_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*r**3*rho0*e**c0 - 3*np.pi*r*rho0*rs0**2)*e**(-c0)

def PhySO_52_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0*e**c0 - 3*np.pi*G*rho0*rs0**2)*e**(-1/2*c0)

# ================= PhySO_53 =================
def PhySO_53_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*c0*rho0**1.00000000000000*rs0**2/r**2 - rho0

def PhySO_53_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_53_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_54 =================
def PhySO_54_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*rho0*rs0*e**(c0 - np.sqrt(-r/rs0))/r

def PhySO_54_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -(48.0*np.pi*rho0*rs0**3*e**(c0 + np.sqrt(-r/rs0)) + 24.0*np.pi*r*rho0*rs0**2*e**c0 + (-8.0*np.pi*rho0*(-r/rs0)**(3/2)*e**c0 - 48.0*np.pi*rho0*np.sqrt(-r/rs0)*e**c0 - 48.0*np.pi*rho0*e**c0)*rs0**3)*e**(-np.sqrt(-r/rs0))

def PhySO_54_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(-(48.0*np.pi*G*rho0*rs0**3*e**(c0 + np.sqrt(-r/rs0)) + 24.0*np.pi*G*r*rho0*rs0**2*e**c0 + (-8.0*np.pi*G*rho0*(-r/rs0)**(3/2)*e**c0 - 48.0*np.pi*G*rho0*np.sqrt(-r/rs0)*e**c0 - 48.0*np.pi*G*rho0*e**c0)*rs0**3)*e**(-np.sqrt(-r/rs0))/r)

def PhySO_54_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return (32.0*np.pi*scipy.special.gamma(-1, 2*np.sqrt(-r/rs0)) + 192.0*np.pi*scipy.special.gamma(-2, 2*np.sqrt(-r/rs0)) + 768.0*np.pi*scipy.special.gamma(-3, 2*np.sqrt(-r/rs0)) + 1536.0*np.pi*scipy.special.gamma(-4, 2*np.sqrt(-r/rs0)) - 96.0*np.pi*scipy.special.gamma(-4, np.sqrt(-r/rs0)))*G*r*rho0*rs0*e**(c0 + np.sqrt(-r/rs0))

# ================= PhySO_55 =================
def PhySO_55_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rs0**2/(r**2*(c0/rho0)**1.00000000000000)

def PhySO_55_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2/c0

def PhySO_55_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_56 =================
def PhySO_56_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/(c0*r**2)

def PhySO_56_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_56_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_57 =================
def PhySO_57_density(r, rho0, Rs, G):
    # Density
    return rho0 - rho0*rs0**2/(c0*r**2)

def PhySO_57_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_57_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_58 =================
def PhySO_58_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rho0*rs0**2/(c0*r**2)

def PhySO_58_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_58_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_59 =================
def PhySO_59_density(r, rho0, Rs, G):
    # Density
    return -rho0 + c0*rho0*rs0/(r**2*(r/rs0**2 - 1/rs0))

def PhySO_59_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*rho0*rs0**3*np.log(rs0) + 4*np.pi*c0*rho0*rs0**3*np.log(abs(-r + rs0)) - 4/3*np.pi*r**3*rho0

def PhySO_59_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(3*np.pi*G*c0*rho0*rs0**3*np.log(rs0) - 3*np.pi*G*c0*rho0*rs0**3*np.log(abs(-r + rs0)) + np.pi*G*r**3*rho0)/r)

def PhySO_59_surface_density(r, rho0, Rs, G):
    # Surface Density
    return +Infinity

# ================= PhySO_60 =================
def PhySO_60_density(r, rho0, Rs, G):
    # Density
    return -(c0*rs0**2/r - r)*rho0/r

def PhySO_60_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_60_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_61 =================
def PhySO_61_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*c0*rho0*rs0**2/r**2 + rho0

def PhySO_61_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_61_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_62 =================
def PhySO_62_density(r, rho0, Rs, G):
    # Density
    return -rho0 - (c0 - 1)*rho0*rs0**2/(c0*r**2)

def PhySO_62_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*(np.pi - np.pi*c0)*r*rho0*rs0**2)/c0

def PhySO_62_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 + 3*(np.pi*G*c0 - np.pi*G)*rho0*rs0**2)/c0)

# ================= PhySO_63 =================
def PhySO_63_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_63_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_63_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_64 =================
def PhySO_64_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_64_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_64_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_65 =================
def PhySO_65_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*c0*rho0*rs0**2/r**2 + rho0

def PhySO_65_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_65_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_66 =================
def PhySO_66_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0/(r/rs0)**(1.00000000000000*c0)

def PhySO_66_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*((3*np.pi - np.pi*c0)*r**3*rho0*(r/rs0)**c0 - 3*np.pi*r**3*rho0)/((c0 - 3)*(r/rs0)**c0)

def PhySO_66_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(3*np.pi*G*r**2*rho0 + (np.pi*G*c0 - 3*np.pi*G)*r**2*rho0*(r/rs0)**c0)/(c0 - 3))/(r/rs0)**(1/2*c0)

# ================= PhySO_67 =================
def PhySO_67_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0**2*e**(c0*r/rs0)/r**2 + rho0

def PhySO_67_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0*r**3*rho0 + 3*np.pi*rho0*rs0**3*e**(c0*r/rs0) - 3*np.pi*rho0*rs0**3)/c0

def PhySO_67_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((np.pi*G*c0*r**3*rho0 + 3*np.pi*G*rho0*rs0**3*e**(c0*r/rs0) - 3*np.pi*G*rho0*rs0**3)/(c0*r))

# ================= PhySO_68 =================
def PhySO_68_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0**2/r**2 - 0.632120558828558)

def PhySO_68_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -71418824/84737187*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_68_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/28245729*np.sqrt(9415243)*np.sqrt(-17854706*np.pi*G*r**2*rho0 + 84737187*np.pi*G*rho0*rs0**2)

# ================= PhySO_69 =================
def PhySO_69_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_69_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_69_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_70 =================
def PhySO_70_density(r, rho0, Rs, G):
    # Density
    return (c0 + rs0/r)*c0*rho0*rs0/r - rho0

def PhySO_70_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0**2*r**2*rho0*rs0 + 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_70_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(3*np.pi*G*c0**2*r*rho0*rs0 + 6*np.pi*G*c0*rho0*rs0**2 - 2*np.pi*G*r**2*rho0)

# ================= PhySO_71 =================
def PhySO_71_density(r, rho0, Rs, G):
    # Density
    return (c0*rho0/rs0 + rho0*rs0/(c0*r**2))*rs0 - rho0

def PhySO_71_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*((np.pi*c0**2 - np.pi*c0)*r**3*rho0 + 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_71_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((3*np.pi*G*rho0*rs0**2 + (np.pi*G*c0**2 - np.pi*G*c0)*r**2*rho0)/c0)

# ================= PhySO_72 =================
def PhySO_72_density(r, rho0, Rs, G):
    # Density
    return c0**2*(-1.00000000000000*r + 1.00000000000000*rs0**2/r)*rho0/r

def PhySO_72_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*c0**2*r**3*rho0 + 4*np.pi*c0**2*r*rho0*rs0**2

def PhySO_72_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*c0**2*r**2*rho0 + 3*np.pi*G*c0**2*rho0*rs0**2)

# ================= PhySO_73 =================
def PhySO_73_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*e**(rs0/((c0*rs0 + r)*c0)) - rho0

def PhySO_73_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2/3*(2*np.pi*c0**2*r**3*rho0 + (np.pi + 6*np.pi*c0**4 - 6*np.pi*c0**2)*rho0*rs0**3*scipy.special.expi(rs0/(c0**2*rs0 + c0*r)) - (6*np.pi*c0**4*scipy.special.expi(c0**(-2)) - 6*np.pi*c0**2*scipy.special.expi(c0**(-2)) + np.pi*scipy.special.expi(c0**(-2)) - (2*np.pi*c0**6 - 5*np.pi*c0**4 + np.pi*c0**2)*e**(c0**(-2)))*rho0*rs0**3 - (2*np.pi*c0**3*r**3*rho0 + np.pi*c0**2*r**2*rho0*rs0 - (4*np.pi*c0**3 - np.pi*c0)*r*rho0*rs0**2 + (2*np.pi*c0**6 - 5*np.pi*c0**4 + np.pi*c0**2)*rho0*rs0**3)*e**(rs0/(c0**2*rs0 + c0*r)))/c0**2

def PhySO_73_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(-(2*np.pi*G*c0**2*r**3*rho0 + (6*np.pi*G*c0**4 - 6*np.pi*G*c0**2 + np.pi*G)*rho0*rs0**3*scipy.special.expi(rs0/(c0**2*rs0 + c0*r)) - (6*np.pi*G*c0**4*scipy.special.expi(c0**(-2)) - 6*np.pi*G*c0**2*scipy.special.expi(c0**(-2)) + np.pi*G*scipy.special.expi(c0**(-2)) - (2*np.pi*G*c0**6 - 5*np.pi*G*c0**4 + np.pi*G*c0**2)*e**(c0**(-2)))*rho0*rs0**3 - (2*np.pi*G*c0**3*r**3*rho0 + np.pi*G*c0**2*r**2*rho0*rs0 - (4*np.pi*G*c0**3 - np.pi*G*c0)*r*rho0*rs0**2 + (2*np.pi*G*c0**6 - 5*np.pi*G*c0**4 + np.pi*G*c0**2)*rho0*rs0**3)*e**(rs0/(c0**2*rs0 + c0*r)))/(c0**2*r))

# ================= PhySO_74 =================
def PhySO_74_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0*e**(c0*rs0/(r + rs0))/r

def PhySO_74_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*(np.pi*c0**2 - 2*np.pi*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(r + rs0)) + 2*((np.pi*c0**2 - 2*np.pi*c0)*scipy.special.expi(c0) + (np.pi - np.pi*c0)*e**c0)*rho0*rs0**3 + 2*(np.pi*c0*r*rho0*rs0**2 + np.pi*r**2*rho0*rs0 - (np.pi - np.pi*c0)*rho0*rs0**3)*e**(c0*rs0/(r + rs0))

def PhySO_74_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-((np.pi*G*c0**2 - 2*np.pi*G*c0)*rho0*rs0**3*scipy.special.expi(c0*rs0/(r + rs0)) - ((np.pi*G*c0**2 - 2*np.pi*G*c0)*scipy.special.expi(c0) - (np.pi*G*c0 - np.pi*G)*e**c0)*rho0*rs0**3 - (np.pi*G*c0*r*rho0*rs0**2 + np.pi*G*r**2*rho0*rs0 + (np.pi*G*c0 - np.pi*G)*rho0*rs0**3)*e**(c0*rs0/(r + rs0)))/r)

# ================= PhySO_75 =================
def PhySO_75_density(r, rho0, Rs, G):
    # Density
    return rho0**1.00000000000000*rs0**2*e**(-2*r/rs0)/(c0**2*r**2)

def PhySO_75_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*rho0*rs0**3*e**(2*r/rs0) - np.pi*rho0*rs0**3)*e**(-2*r/rs0)/c0**2

def PhySO_75_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*rho0*rs0**3*e**(2*r/rs0) - np.pi*G*rho0*rs0**3)/r)*e**(-r/rs0)/c0

def PhySO_75_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -16*(8*np.pi*scipy.special.gamma(-3, 4*r/rs0) - np.pi*scipy.special.gamma(-3, 2*r/rs0))*G*r**2*rho0*e**(2*r/rs0)/c0**2

def PhySO_75_potential(r, rho0, Rs, G):
    # Potential
    return 2*(2*np.pi*G*r*rho0*rs0**2*scipy.special.gamma(-1, 2*r/rs0) - np.pi*G*rho0*rs0**3)/(c0**2*r)

# ================= PhySO_76 =================
def PhySO_76_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0**2*e**(-r/(c0*rs0))/r**2

def PhySO_76_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*(np.pi*c0*rho0*rs0**3*e**(r/(c0*rs0)) - np.pi*c0*rho0*rs0**3)*e**(-r/(c0*rs0))

def PhySO_76_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt((np.pi*G*c0*rho0*rs0**3*e**(r/(c0*rs0)) - np.pi*G*c0*rho0*rs0**3)/r)*e**(-1/2*r/(c0*rs0))

def PhySO_76_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -4*(8*np.pi*scipy.special.gamma(-3, 2*r/(c0*rs0)) - np.pi*scipy.special.gamma(-3, r/(c0*rs0)))*G*r**2*rho0*e**(r/(c0*rs0))/c0**2

def PhySO_76_potential(r, rho0, Rs, G):
    # Potential
    return -4*(np.pi*G*c0*rho0*rs0**3 - np.pi*G*r*rho0*rs0**2*scipy.special.gamma(-1, r/(c0*rs0)))/r

# ================= PhySO_77 =================
def PhySO_77_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0**2*e**(-c0*r/rs0)/r**2

def PhySO_77_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*(np.pi*rho0*rs0**3*e**(c0*r/rs0) - np.pi*rho0*rs0**3)*e**(-c0*r/rs0)/c0

def PhySO_77_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt((np.pi*G*rho0*rs0**3*e**(c0*r/rs0) - np.pi*G*rho0*rs0**3)/(c0*r))*e**(-1/2*c0*r/rs0)

def PhySO_77_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -4*(8*np.pi*scipy.special.gamma(-3, 2*c0*r/rs0) - np.pi*scipy.special.gamma(-3, c0*r/rs0))*G*c0**2*r**2*rho0*e**(c0*r/rs0)

def PhySO_77_potential(r, rho0, Rs, G):
    # Potential
    return 4*(np.pi*G*c0*r*rho0*rs0**2*scipy.special.gamma(-1, c0*r/rs0) - np.pi*G*rho0*rs0**3)/(c0*r)

# ================= PhySO_78 =================
def PhySO_78_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*e**(c0*r/rs0)/r**2

def PhySO_78_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*(np.pi*rho0*rs0**3*e**(c0*r/rs0) - np.pi*rho0*rs0**3)/c0

def PhySO_78_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-(np.pi*G*rho0*rs0**3*e**(c0*r/rs0) - np.pi*G*rho0*rs0**3)/(c0*r))

# ================= PhySO_79 =================
def PhySO_79_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*e**(-r/rs0)/r**2

def PhySO_79_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*(np.pi*rho0*rs0**3*e**(r/rs0) - np.pi*rho0*rs0**3)*e**(-r/rs0)

def PhySO_79_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(-(np.pi*G*rho0*rs0**3*e**(r/rs0) - np.pi*G*rho0*rs0**3)*e**(-r/rs0)/r)

def PhySO_79_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return 4*(8*np.pi*scipy.special.gamma(-3, 2*r/rs0) - np.pi*scipy.special.gamma(-3, r/rs0))*G*r**2*rho0*e**(r/rs0)

def PhySO_79_potential(r, rho0, Rs, G):
    # Potential
    return -4*(np.pi*G*r*rho0*rs0**2*scipy.special.gamma(-1, r/rs0) - np.pi*G*rho0*rs0**3)/r

# ================= PhySO_80 =================
def PhySO_80_density(r, rho0, Rs, G):
    # Density
    return (c0 + 1.00000000000000*rs0/r)*rho0*rs0/r + rho0

def PhySO_80_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 + 4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_80_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/3*np.sqrt(3)*np.sqrt(2)*np.sqrt(3*np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*r**2*rho0 + 6*np.pi*G*rho0*rs0**2)

# ================= PhySO_81 =================
def PhySO_81_density(r, rho0, Rs, G):
    # Density
    return rho0*e**(2*np.log(c0*(1.00000000000000*r - 1.00000000000000*rs0)/r) + 2.00000000000000)

def PhySO_81_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 9.8520747985742*np.pi*c0**2*r**3*rho0 - 29.5562243957226*np.pi*c0**2*r**2*rho0*rs0 + 29.5562243957226*np.pi*c0**2*r*rho0*rs0**2

def PhySO_81_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(9.8520747985742*np.pi*G*c0**2*r**2*rho0 - 29.5562243957226*np.pi*G*c0**2*r*rho0*rs0 + 29.5562243957226*np.pi*G*c0**2*rho0*rs0**2)

# ================= PhySO_82 =================
def PhySO_82_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0/r - 1.00000000000000)**2

def PhySO_82_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_82_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*r*rho0*rs0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_83 =================
def PhySO_83_density(r, rho0, Rs, G):
    # Density
    return (1.00000000000000*c0**2*rs0 - 1.00000000000000*r)**2*rho0**1.00000000000000/r**2

def PhySO_83_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0**4*r*rho0*rs0**2 - 4.0*np.pi*c0**2*r**2*rho0*rs0 + 1.333333333333333*np.pi*r**3*rho0

def PhySO_83_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(4*np.pi*G*c0**4*rho0*rs0**2 - 4.0*np.pi*G*c0**2*r*rho0*rs0 + 1.333333333333333*np.pi*G*r**2*rho0)

# ================= PhySO_84 =================
def PhySO_84_density(r, rho0, Rs, G):
    # Density
    return (c0*rho0*rs0/r - rho0)**2/rho0**1.00000000000000

def PhySO_84_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0**2*r*rho0*rs0**2 - 4*np.pi*c0*r**2*rho0*rs0 + 4/3*np.pi*r**3*rho0

def PhySO_84_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0**2*rho0*rs0**2 - 3*np.pi*G*c0*r*rho0*rs0 + np.pi*G*r**2*rho0)

# ================= PhySO_85 =================
def PhySO_85_density(r, rho0, Rs, G):
    # Density
    return ((c0 - e**2)*r - rs0)**2*rho0/r**2

def PhySO_85_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0**2 - 2*np.pi*c0*e**2 + np.pi*e**4)*r**3*rho0 - 4*(np.pi*c0 - np.pi*e**2)*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_85_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*rho0*rs0**2 + (np.pi*G*c0**2 - 2*np.pi*G*c0*e**2 + np.pi*G*e**4)*r**2*rho0 - 3*(np.pi*G*c0 - np.pi*G*e**2)*r*rho0*rs0)

# ================= PhySO_86 =================
def PhySO_86_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0/r - e**(0.500000000000000*c0 - 0.500000000000000))**2

def PhySO_86_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2.426122638850534*np.pi*r**2*rho0*rs0*e**(0.5*c0) + 0.4905059215619232*np.pi*r**3*rho0*e**c0 + 4.0*np.pi*r*rho0*rs0**2

def PhySO_86_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(-2.426122638850534*np.pi*G*r*rho0*rs0*e**(0.5*c0) + 0.4905059215619232*np.pi*G*r**2*rho0*e**c0 + 4.0*np.pi*G*rho0*rs0**2)

# ================= PhySO_87 =================
def PhySO_87_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0 - rs0**2/r)/r

def PhySO_87_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*r**2*rho0*rs0 - 4*np.pi*r*rho0*rs0**2

def PhySO_87_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_88 =================
def PhySO_88_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*(1.00000000000000/r - 1/rs0)/r

def PhySO_88_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*r**2*rho0*rs0 - 4*np.pi*r*rho0*rs0**2

def PhySO_88_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_89 =================
def PhySO_89_density(r, rho0, Rs, G):
    # Density
    return (c0**2*rs0/r - c0)*rho0*rs0/r

def PhySO_89_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0**2*r*rho0*rs0**2 - 2*np.pi*c0*r**2*rho0*rs0

def PhySO_89_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(2*np.pi*G*c0**2*rho0*rs0**2 - np.pi*G*c0*r*rho0*rs0)

# ================= PhySO_90 =================
def PhySO_90_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*(r - rs0)*rho0*rs0/(c0*r**2)

def PhySO_90_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return (2*np.pi*r**2*rho0*rs0 - 4.0*np.pi*r*rho0*rs0**2)/c0

def PhySO_90_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt((2*np.pi*G*r*rho0*rs0 - 4.0*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_91 =================
def PhySO_91_density(r, rho0, Rs, G):
    # Density
    return (c0 + rs0/r)*rho0*rs0/(c0*r)

def PhySO_91_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*c0*r**2*rho0*rs0 + 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_91_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_92 =================
def PhySO_92_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0*(-1.00000000000000*rs0/r + 1.00000000000000)/(c0*r)

def PhySO_92_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*r**2*rho0*rs0 - 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_92_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_93 =================
def PhySO_93_density(r, rho0, Rs, G):
    # Density
    return (r - rs0)*rho0*rs0/(c0*r**2)

def PhySO_93_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*(np.pi*r**2*rho0*rs0 - 2*np.pi*r*rho0*rs0**2)/c0

def PhySO_93_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt((np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_94 =================
def PhySO_94_density(r, rho0, Rs, G):
    # Density
    return -(c0 - rs0/r)*rho0*rs0/r

def PhySO_94_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_94_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_95 =================
def PhySO_95_density(r, rho0, Rs, G):
    # Density
    return -(c0*r - rs0)*c0*rho0*rs0*e**(-c0)/r**2

def PhySO_95_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*(np.pi*c0**2*r**2*rho0*rs0 - 2*np.pi*c0*r*rho0*rs0**2)*e**(-c0)

def PhySO_95_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-(np.pi*G*c0**2*r*rho0*rs0 - 2*np.pi*G*c0*rho0*rs0**2)*e**(-c0))

# ================= PhySO_96 =================
def PhySO_96_density(r, rho0, Rs, G):
    # Density
    return c0*(r + rs0/c0)*rho0*rs0/r**2

def PhySO_96_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_96_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_97 =================
def PhySO_97_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0*np.log(c0*e**(2.00000000000000*rs0/r))/r

def PhySO_97_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4272943/680060*r**2*rho0*rs0*np.log(c0) - 4272943/170015*r*rho0*rs0**2

def PhySO_97_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 1/340030*np.sqrt(4272943)*np.sqrt(170015)*np.sqrt(-G*r*rho0*rs0*np.log(c0) - 4*G*rho0*rs0**2)

def PhySO_97_surface_density(r, rho0, Rs, G):
    # Surface Density
    return +Infinity

# ================= PhySO_98 =================
def PhySO_98_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*(c0 - rs0/r)**(1/c0)*rho0

def PhySO_98_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r**(-1/c0 + 3)*rho0*rs0**(1/c0)*e**(I*np.pi/c0)*hypergeometric((-1/c0, (3*c0 - 1)/c0), ((4*c0 - 1)/c0,), c0*r/rs0)/(3*c0 - 1)

def PhySO_98_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2*np.sqrt(np.pi*G*c0*r**(-1/c0 + 2)*rho0*rs0**(1/c0)*hypergeometric((-1/c0, (3*c0 - 1)/c0), ((4*c0 - 1)/c0,), c0*r/rs0)/(3*c0 - 1))*e**(1/2*I*np.pi/c0)

# ================= PhySO_99 =================
def PhySO_99_density(r, rho0, Rs, G):
    # Density
    return (c0 - 1.00000000000000*c0*rs0/r)**(1/c0)*rho0

