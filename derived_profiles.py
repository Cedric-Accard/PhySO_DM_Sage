import numpy as np
import scipy.special

# Helper wrapper for Sage compatibility
def dilog(x):
    return scipy.special.spence(1 - x)

# ================= PhySO_0 =================
def PhySO_0_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rho0*rs0**2/r**2

def PhySO_0_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_0_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_1 =================
def PhySO_1_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*rho0*rs0/(c0*r*(7.38905609893065*r**2/rs0**2 + 1.00000000000000))

def PhySO_1_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 0.2706705664732254*np.pi*rho0*rs0**3*np.log((7.38905609893065*r**2 + 1.0*rs0**2)/rs0**2)/c0

def PhySO_1_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 0.520260095022889*np.sqrt(np.pi*G*rho0*rs0**3*np.log((7.38905609893065*r**2 + 1.0*rs0**2)/rs0**2)/(c0*r))

def PhySO_1_surface_density(r, rho0, Rs, G):
    # Surface Density
    return -2091716122475/2462863044*(rho0*rs0**2*np.log((165417469811022409873132816*R**4 + 179094561033273389971758733*R**2*rs0**2 + 24237813143575609401840835*rs0**4 - 19782263938339608*(5329837*R**2*rs0 + 1442630*rs0**3)*np.sqrt(5329837*R**2 + 721315*rs0**2))/(165417469811022409873132816*R**4 + 429963280627*R**2*rs0**2 + 58189202365*rs0**4)) - rho0*rs0**2*np.log((25722944606792*R**2 + 3481221994040*rs0**2 - 4098918063*np.sqrt(5329837*R**2 + 721315*rs0**2)*rs0)/(25722944606792*R**2 + 3481221994040*rs0**2 + 4098918063*np.sqrt(5329837*R**2 + 721315*rs0**2)*rs0)))*np.sqrt(5329837*R**2 + 721315*rs0**2)/(5329837*R**2*c0 + 721315*c0*rs0**2)

# ================= PhySO_2 =================
def PhySO_2_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rho0*rs0**2/r**2

def PhySO_2_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_2_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_3 =================
def PhySO_3_density(r, rho0, Rs, G):
    # Density
    return -c0*rho0*rs0/(r*(r**2/rs0**2 + 1.00000000000000))

def PhySO_3_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*np.pi*c0*rho0*rs0**3*np.log((1.0*r**2 + 1.0*rs0**2)/rs0**2)

def PhySO_3_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0*rho0*rs0**3*np.log((1.0*r**2 + 1.0*rs0**2)/rs0**2)/r)

def PhySO_3_surface_density(r, rho0, Rs, G):
    # Surface Density
    return -(c0*rho0*rs0**2*np.log((R**2 + rs0**2 - np.sqrt(R**2 + rs0**2)*rs0)/(R**2 + rs0**2 + np.sqrt(R**2 + rs0**2)*rs0)) - c0*rho0*rs0**2*np.log((R**4 + 8*R**2*rs0**2 + 8*rs0**4 - 4*(R**2*rs0 + 2*rs0**3)*np.sqrt(R**2 + rs0**2))/R**4))/np.sqrt(R**2 + rs0**2)

def PhySO_3_average_surface_density(r, rho0, Rs, G):
    # Average Surface Density
    return 2*(2*c0*rho0*rs0**3*(np.log(2) - np.log(R)) + 2*c0*rho0*rs0**3*np.log(rs0) - (c0*rho0*rs0**2*np.log((R**2 + 2*rs0**2 - 2*np.sqrt(R**2 + rs0**2)*rs0)/R**2) - c0*rho0*rs0**2*np.log((R**4 + 8*R**2*rs0**2 + 8*rs0**4 - 4*(R**2*rs0 + 2*rs0**3)*np.sqrt(R**2 + rs0**2))/R**4))*np.sqrt(R**2 + rs0**2))/R**2

# ================= PhySO_4 =================
def PhySO_4_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rho0*rs0**2/r**2

def PhySO_4_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_4_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_5 =================
def PhySO_5_density(r, rho0, Rs, G):
    # Density
    return -rho0 + 1.00000000000000*rho0*rs0**2/r**2

def PhySO_5_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_5_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_6 =================
def PhySO_6_density(r, rho0, Rs, G):
    # Density
    return rho0 - rho0*rs0**2/r**2

def PhySO_6_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r*rho0*rs0**2

def PhySO_6_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_7 =================
def PhySO_7_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/r**2

def PhySO_7_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_7_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_8 =================
def PhySO_8_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/r**2

def PhySO_8_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_8_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_9 =================
def PhySO_9_density(r, rho0, Rs, G):
    # Density
    return (c0 - rs0**2*e**(c0*r/rs0)/r**2)*rho0

def PhySO_9_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0**2*r**3*rho0 - 3*np.pi*rho0*rs0**3*e**(c0*r/rs0) + 3*np.pi*rho0*rs0**3)/c0

def PhySO_9_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((np.pi*G*c0**2*r**3*rho0 - 3*np.pi*G*rho0*rs0**3*e**(c0*r/rs0) + 3*np.pi*G*rho0*rs0**3)/(c0*r))

# ================= PhySO_10 =================
def PhySO_10_density(r, rho0, Rs, G):
    # Density
    return -rho0*e**(np.log(1.00000000000000*(r + rs0)/(c0*rs0))**2)

# ================= PhySO_11 =================
def PhySO_11_density(r, rho0, Rs, G):
    # Density
    return rho0 - 2.71828182845905*rho0*rs0**2/(c0*r**2)

def PhySO_11_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/31173069*(10391023*np.pi*c0*r**3*rho0 - 84737187*np.pi*r*rho0*rs0**2)/c0

def PhySO_11_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/31173069*np.sqrt(31173069)*np.sqrt((10391023*np.pi*G*c0*r**2*rho0 - 84737187*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_12 =================
def PhySO_12_density(r, rho0, Rs, G):
    # Density
    return (1.00000000000000*c0*e**(-np.sqrt(c0*r/rs0)) + 1.00000000000000)**2*rho0 - rho0

def PhySO_12_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return (-10.0*np.pi*c0**3*r**2*rho0*rs0 - 30.0*np.pi*c0**2*r*rho0*rs0**2 + (1920.0*np.pi + 15.0*np.pi*c0)*rho0*rs0**3*e**(2*np.sqrt(c0*r/rs0)) + (-4.0*np.pi*c0*rho0*(c0*r/rs0)**(5/2) - 20.0*np.pi*c0*rho0*(c0*r/rs0)**(3/2) - 30.0*np.pi*c0*rho0*np.sqrt(c0*r/rs0) - 15.0*np.pi*c0*rho0)*rs0**3 + (-80.0*np.pi*c0**2*r**2*rho0*rs0 - 960.0*np.pi*c0*r*rho0*rs0**2 + (-16.0*np.pi*rho0*(c0*r/rs0)**(5/2) - 320.0*np.pi*rho0*(c0*r/rs0)**(3/2) - 1920.0*np.pi*rho0*np.sqrt(c0*r/rs0) - 1920.0*np.pi*rho0)*rs0**3)*e**(np.sqrt(c0*r/rs0)))*e**(-2*np.sqrt(c0*r/rs0))/c0**2

def PhySO_12_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt((-10.0*np.pi*G*c0**3*r**2*rho0*rs0 - 30.0*np.pi*G*c0**2*r*rho0*rs0**2 + (15.0*np.pi*G*c0 + 1920.0*np.pi*G)*rho0*rs0**3*e**(2*np.sqrt(c0*r/rs0)) + (-4.0*np.pi*G*c0*rho0*(c0*r/rs0)**(5/2) - 20.0*np.pi*G*c0*rho0*(c0*r/rs0)**(3/2) - 30.0*np.pi*G*c0*rho0*np.sqrt(c0*r/rs0) - 15.0*np.pi*G*c0*rho0)*rs0**3 + (-80.0*np.pi*G*c0**2*r**2*rho0*rs0 - 960.0*np.pi*G*c0*r*rho0*rs0**2 + (-16.0*np.pi*G*rho0*(c0*r/rs0)**(5/2) - 320.0*np.pi*G*rho0*(c0*r/rs0)**(3/2) - 1920.0*np.pi*G*rho0*np.sqrt(c0*r/rs0) - 1920.0*np.pi*G*rho0)*rs0**3)*e**(np.sqrt(c0*r/rs0)))/r)*e**(-np.sqrt(c0*r/rs0))/c0

def PhySO_12_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return -(-(1.01195085757989e+21)*G*c0**3*r*rho0*rs0**2*scipy.special.expi(-4*np.sqrt(c0*r*rs0)/rs0)*e**(2*np.sqrt(c0*r/rs0) + 4*np.sqrt(c0*r*rs0)/rs0) - (1.30035685212073e+23)*G*c0**2*r*rho0*rs0**2*scipy.special.expi(-3*np.sqrt(c0*r*rs0)/rs0)*e**(2*np.sqrt(c0*r/rs0) + 4*np.sqrt(c0*r*rs0)/rs0) + ((5.05975428789945e+20)*G*c0**2 + (6.47648548916414e+22)*G*c0)*r*rho0*rs0**2*scipy.special.expi(-np.sqrt(c0*r*rs0)/rs0)*e**(2*np.sqrt(c0*r/rs0) + 4*np.sqrt(c0*r*rs0)/rs0) + ((1.01195085757989e+21)*G*c0**3 + (1.29529709783283e+23)*G*c0**2 - (6.47648548916414e+22)*G*c0)*r*rho0*rs0**2*scipy.special.expi(-2*np.sqrt(c0*r*rs0)/rs0)*e**(2*np.sqrt(c0*r/rs0) + 4*np.sqrt(c0*r*rs0)/rs0) + (-(5.05975428789945e+20)*G*c0 - (6.47648548916414e+22)*G)*rho0*rs0**3*e**(2*np.sqrt(c0*r/rs0) + 3*np.sqrt(c0*r*rs0)/rs0) + ((3.37316952560632e+19)*G*c0**4*r**2*rho0*rs0 + (1.93957247722363e+20)*G*c0**3*r*rho0*rs0**2 + (2.52987714420474e+20)*G*c0**2*rho0*rs0**3)*e**(2*np.sqrt(c0*r/rs0)) + ((5.39707123868518e+20)*G*c0**2*r**2*rho0*rs0 + (1.24132638541986e+22)*G*c0*r*rho0*rs0**2 + (-(2.52987714420474e+20)*G*c0**2 - (3.23824274458207e+22)*G*c0 + (6.47648548916414e+22)*G)*rho0*rs0**3)*e**(2*np.sqrt(c0*r/rs0) + 2*np.sqrt(c0*r*rs0)/rs0) + ((2.69853562048505e+20)*G*c0**3*r**2*rho0*rs0 + (4.48256750283252e+21)*G*c0**2*r*rho0*rs0**2 + (3.28884028746107e+22)*G*c0*rho0*rs0**3)*e**(2*np.sqrt(c0*r/rs0) + np.sqrt(c0*r*rs0)/rs0) + ((1.0119508576819e+20)*G*c0**3*r*rho0*rs0*e**(2*np.sqrt(c0*r/rs0)) + ((5.05975428789945e+20)*G*c0 + (6.47648548916414e+22)*G)*rho0*rs0**2*e**(2*np.sqrt(c0*r/rs0) + 3*np.sqrt(c0*r*rs0)/rs0) + ((3.23824274451679e+21)*G*c0*r*rho0*rs0 + ((5.05975428789945e+20)*G*c0**2 + (6.47648548916414e+22)*G*c0)*rho0*rs0**2)*e**(2*np.sqrt(c0*r/rs0) + 2*np.sqrt(c0*r*rs0)/rs0) + ((1.30429221655962e+21)*G*c0**2*r*rho0*rs0 - (3.18764520167043e+22)*G*c0*rho0*rs0**2)*e**(2*np.sqrt(c0*r/rs0) + np.sqrt(c0*r*rs0)/rs0))*np.sqrt(c0*r*rs0))*e**(-4*np.sqrt(c0*r*rs0)/rs0)/((5.36856603887193e+18)*c0**3*r + (1.07371320777439e+19)*c0**2*r*e**(np.sqrt(c0*r/rs0)))

def PhySO_12_potential(r, rho0, Rs, G):
    # Potential
    return -(-4.0*np.pi*G*c0**3*r**2*rho0*rs0 + (-14.0*np.pi*G*c0**2*r*rho0*np.sqrt(c0*r/rs0) - 27.0*np.pi*G*c0**2*r*rho0)*rs0**2 + ((15.0*np.pi*G*c0 + 1920.0*np.pi*G)*rho0*rs0**3 + (60.0*np.pi*G*c0**2*r*rho0*scipy.special.expi(-2*np.sqrt(c0*r/rs0)) + 1920.0*np.pi*G*c0*r*rho0*scipy.special.expi(-np.sqrt(c0*r/rs0)) + ((-120.0*np.pi*scipy.special.gamma(-1, 2*np.sqrt(c0*r/rs0)) - 120.0*np.pi*scipy.special.gamma(-2, 2*np.sqrt(c0*r/rs0)))*G*c0**2 + (-3840.0*np.pi*scipy.special.gamma(-1, np.sqrt(c0*r/rs0)) - 3840.0*np.pi*scipy.special.gamma(-2, np.sqrt(c0*r/rs0)))*G*c0)*r*rho0)*rs0**2)*e**(2*np.sqrt(c0*r/rs0)) + (-32.0*np.pi*G*c0**2*r**2*rho0*rs0 + (-224.0*np.pi*G*c0*r*rho0*np.sqrt(c0*r/rs0) - 864.0*np.pi*G*c0*r*rho0)*rs0**2)*e**(np.sqrt(c0*r/rs0)))*e**(-2*np.sqrt(c0*r/rs0))/(c0**2*r)

# ================= PhySO_13 =================
def PhySO_13_density(r, rho0, Rs, G):
    # Density
    return rho0 - 1.00000000000000*rho0*rs0**2/(c0*r**2)

def PhySO_13_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_13_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_14 =================
def PhySO_14_density(r, rho0, Rs, G):
    # Density
    return rho0 - 1.00000000000000*rho0*rs0**2/(c0*r**2)

def PhySO_14_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_14_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt((np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_15 =================
def PhySO_15_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/(c0*r**2)

def PhySO_15_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_15_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_16 =================
def PhySO_16_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/(c0*r**2)

def PhySO_16_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_16_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_17 =================
def PhySO_17_density(r, rho0, Rs, G):
    # Density
    return -rho0 + rho0*rs0**2/(c0*r**2)

def PhySO_17_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*(np.pi*c0*r**3*rho0 - 3*np.pi*r*rho0*rs0**2)/c0

def PhySO_17_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-(np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)/c0)

# ================= PhySO_18 =================
def PhySO_18_density(r, rho0, Rs, G):
    # Density
    return -c0*rho0*rs0**2/r**2 + rho0

def PhySO_18_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_18_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_19 =================
def PhySO_19_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*c0*rho0*rs0**2/r**2 - rho0

def PhySO_19_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_19_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_20 =================
def PhySO_20_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*c0*rho0*rs0**2/r**2 - rho0

def PhySO_20_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_20_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_21 =================
def PhySO_21_density(r, rho0, Rs, G):
    # Density
    return -c0*rho0*rs0**2/r**2 + rho0

def PhySO_21_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 4/3*np.pi*r**3*rho0

def PhySO_21_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-3*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r**2*rho0)

# ================= PhySO_22 =================
def PhySO_22_density(r, rho0, Rs, G):
    # Density
    return -(c0 - 1.00000000000000)*rho0*rs0**2/r**2 - rho0

def PhySO_22_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*r**3*rho0 + 4*(np.pi - np.pi*c0)*r*rho0*rs0**2

def PhySO_22_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*r**2*rho0 - 3*(np.pi*G*c0 - np.pi*G)*rho0*rs0**2)

# ================= PhySO_23 =================
def PhySO_23_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*rho0*rs0**2/(r**2*(1.00000000000000*r/rs0 + 1.00000000000000))

def PhySO_23_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4.0*np.pi*rho0*rs0**3*np.log((1.0*r + 1.0*rs0)/rs0)

def PhySO_23_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2.0*np.sqrt(-np.pi*G*rho0*rs0**3*np.log((1.0*r + 1.0*rs0)/rs0)/r)

def PhySO_23_surface_density(r, rho0, Rs, G):
    # Surface Density
    return -(np.pi*R**2*rho0*rs0**2 - np.pi*rho0*rs0**4 - 4*(R*rho0*rs0**2*np.arctan((R + rs0)/np.sqrt(R**2 - rs0**2)) - R*rho0*rs0**2*np.arctan(rs0/np.sqrt(R**2 - rs0**2)))*np.sqrt(R**2 - rs0**2))/(R**3 - R*rs0**2)

# ================= PhySO_24 =================
def PhySO_24_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_24_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_24_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_25 =================
def PhySO_25_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*c0*rho0*rs0**2/r**2 - rho0

def PhySO_25_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_25_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_26 =================
def PhySO_26_density(r, rho0, Rs, G):
    # Density
    return c0*(rho0 + 1.00000000000000*rho0*rs0**2/(c0*r**2))

def PhySO_26_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*c0*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_26_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*c0*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_27 =================
def PhySO_27_density(r, rho0, Rs, G):
    # Density
    return c0*rho0*rs0**2/r**2 - rho0

def PhySO_27_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 - 4/3*np.pi*r**3*rho0

def PhySO_27_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0*rho0*rs0**2 - np.pi*G*r**2*rho0)

# ================= PhySO_28 =================
def PhySO_28_density(r, rho0, Rs, G):
    # Density
    return -(c0*r + 1.00000000000000*rs0**2/r)*rho0/r

def PhySO_28_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4/3*np.pi*c0*r**3*rho0 - 4*np.pi*r*rho0*rs0**2

def PhySO_28_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(-np.pi*G*c0*r**2*rho0 - 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_29 =================
def PhySO_29_density(r, rho0, Rs, G):
    # Density
    return (c0 + rs0**2/r**2)*rho0

def PhySO_29_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*c0*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_29_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*c0*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_30 =================
def PhySO_30_density(r, rho0, Rs, G):
    # Density
    return (c0 + 1.00000000000000*rs0**2/r**2)*rho0

def PhySO_30_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*c0*r**3*rho0 + 4*np.pi*r*rho0*rs0**2

def PhySO_30_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*c0*r**2*rho0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_31 =================
def PhySO_31_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*e**(1.00000000000000*c0*r/rs0)/r**2

def PhySO_31_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return (-4.0*np.pi*rho0*rs0**3*e**(1.0*c0*r/rs0) + 4.0*np.pi*rho0*rs0**3)/c0

def PhySO_31_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt((-4.0*np.pi*G*rho0*rs0**3*e**(1.0*c0*r/rs0) + 4.0*np.pi*G*rho0*rs0**3)/(c0*r))

# ================= PhySO_32 =================
def PhySO_32_density(r, rho0, Rs, G):
    # Density
    return rho0*rs0**2*e**(-1.00000000000000*r/rs0)/r**2

def PhySO_32_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4.0*np.pi*rho0*rs0**3*e**(-1.0*r/rs0) + 4.0*np.pi*rho0*rs0**3

def PhySO_32_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt((-4.0*np.pi*G*rho0*rs0**3*e**(-1.0*r/rs0) + 4.0*np.pi*G*rho0*rs0**3)/r)

def PhySO_32_sigma2(r, rho0, Rs, G):
    # Radial Velocity Dispersion
    return (-32.0*np.pi*scipy.special.gamma(-3, 2*r/rs0) + 4.0*np.pi*scipy.special.gamma(-3, r/rs0))*G*r**2*rho0*e**(1.0*r/rs0)

def PhySO_32_potential(r, rho0, Rs, G):
    # Potential
    return -(-4.0*np.pi*G*r*rho0*rs0**2*scipy.special.gamma(-1, r/rs0) + 4.0*np.pi*G*rho0*rs0**3)/r

# ================= PhySO_33 =================
def PhySO_33_density(r, rho0, Rs, G):
    # Density
    return -c0*rho0*rs0*np.log(1.00000000000000*r*e**(rs0/r)/(c0*rs0))/r

def PhySO_33_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2.0*np.pi*c0*r**2*rho0*rs0*np.log(1.0*r*e**(rs0/r)/(c0*rs0)) + 1.0*np.pi*c0*r**2*rho0*rs0 - 2.0*np.pi*c0*r*rho0*rs0**2

def PhySO_33_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(-2.0*np.pi*G*c0*r*rho0*rs0*np.log(1.0*r*e**(rs0/r)/(c0*rs0)) + 1.0*np.pi*G*c0*r*rho0*rs0 - 2.0*np.pi*G*c0*rho0*rs0**2)

# ================= PhySO_34 =================
def PhySO_34_density(r, rho0, Rs, G):
    # Density
    return (c0 - rs0/r)*rho0*rs0/r

def PhySO_34_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 - 4*np.pi*r*rho0*rs0**2

def PhySO_34_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_35 =================
def PhySO_35_density(r, rho0, Rs, G):
    # Density
    return -(rho0 - rho0*rs0/r)*rs0*(1/r - 1/rs0)

def PhySO_35_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_35_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*r*rho0*rs0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_36 =================
def PhySO_36_density(r, rho0, Rs, G):
    # Density
    return (np.sqrt(rho0) - 1.00000000000000*np.sqrt(rho0)*rs0/r)**2

def PhySO_36_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*np.pi*r**3*rho0 - 4*np.pi*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_36_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*r**2*rho0 - 3*np.pi*G*r*rho0*rs0 + 3*np.pi*G*rho0*rs0**2)

# ================= PhySO_37 =================
def PhySO_37_density(r, rho0, Rs, G):
    # Density
    return rho0*(1.00000000000000*c0*rs0/r - 1.00000000000000)**2

def PhySO_37_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0**2*r*rho0*rs0**2 - 4*np.pi*c0*r**2*rho0*rs0 + 4/3*np.pi*r**3*rho0

def PhySO_37_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*c0**2*rho0*rs0**2 - 3*np.pi*G*c0*r*rho0*rs0 + np.pi*G*r**2*rho0)

# ================= PhySO_38 =================
def PhySO_38_density(r, rho0, Rs, G):
    # Density
    return rho0*(rs0*e**(c0**2)/r - 1.00000000000000)**2

def PhySO_38_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*r*rho0*rs0**2*e**(2*c0**2) - 4*np.pi*r**2*rho0*rs0*e**(c0**2) + 4/3*np.pi*r**3*rho0

def PhySO_38_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(3*np.pi*G*rho0*rs0**2*e**(2*c0**2) - 3*np.pi*G*r*rho0*rs0*e**(c0**2) + np.pi*G*r**2*rho0)

# ================= PhySO_39 =================
def PhySO_39_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0*(1.00000000000000*rs0/r - 1.00000000000000)/r

def PhySO_39_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*r**2*rho0*rs0 - 4*np.pi*r*rho0*rs0**2

def PhySO_39_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*r*rho0*rs0 - 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_40 =================
def PhySO_40_density(r, rho0, Rs, G):
    # Density
    return (c0*r - rs0/c0)**2*rho0/r**2

def PhySO_40_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4/3*(np.pi*c0**4*r**3*rho0 - 3*np.pi*c0**2*r**2*rho0*rs0 + 3*np.pi*r*rho0*rs0**2)/c0**2

def PhySO_40_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return 2/3*np.sqrt(3)*np.sqrt(np.pi*G*c0**4*r**2*rho0 - 3*np.pi*G*c0**2*r*rho0*rs0 + 3*np.pi*G*rho0*rs0**2)/c0

# ================= PhySO_41 =================
def PhySO_41_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*rho0*rs0*(rs0/r - 1.00000000000000)/r

def PhySO_41_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2.0*np.pi*r**2*rho0*rs0 + 4.0*np.pi*r*rho0*rs0**2

def PhySO_41_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(-2.0*np.pi*G*r*rho0*rs0 + 4.0*np.pi*G*rho0*rs0**2)

# ================= PhySO_42 =================
def PhySO_42_density(r, rho0, Rs, G):
    # Density
    return 1.00000000000000*rho0*rs0**2*(1/r - 1/rs0)/r

def PhySO_42_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2.0*np.pi*r**2*rho0*rs0 + 4.0*np.pi*r*rho0*rs0**2

def PhySO_42_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(-2.0*np.pi*G*r*rho0*rs0 + 4.0*np.pi*G*rho0*rs0**2)

# ================= PhySO_43 =================
def PhySO_43_density(r, rho0, Rs, G):
    # Density
    return (c0*rho0*rs0/r - rho0)*c0*rs0/r

def PhySO_43_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0**2*r*rho0*rs0**2 - 2*np.pi*c0*r**2*rho0*rs0

def PhySO_43_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(2*np.pi*G*c0**2*rho0*rs0**2 - np.pi*G*c0*r*rho0*rs0)

# ================= PhySO_44 =================
def PhySO_44_density(r, rho0, Rs, G):
    # Density
    return -(c0 - rs0/(c0*r))*c0*rho0*rs0/r

def PhySO_44_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -2*np.pi*c0**2*r**2*rho0*rs0 + 4*np.pi*r*rho0*rs0**2

def PhySO_44_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-np.pi*G*c0**2*r*rho0*rs0 + 2*np.pi*G*rho0*rs0**2)

# ================= PhySO_45 =================
def PhySO_45_density(r, rho0, Rs, G):
    # Density
    return (c0*(r + rs0)/r + 1.00000000000000)*rho0*rs0/r

def PhySO_45_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 4*np.pi*c0*r*rho0*rs0**2 + 2*(np.pi + np.pi*c0)*r**2*rho0*rs0

def PhySO_45_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(2*np.pi*G*c0*rho0*rs0**2 + (np.pi*G*c0 + np.pi*G)*r*rho0*rs0)

# ================= PhySO_46 =================
def PhySO_46_density(r, rho0, Rs, G):
    # Density
    return -1.00000000000000*rho0*rs0*(rs0/r - 1/c0)/r

def PhySO_46_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return (-4.0*np.pi*c0*r*rho0*rs0**2 + 2.0*np.pi*r**2*rho0*rs0)/c0

def PhySO_46_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt((-4.0*np.pi*G*c0*rho0*rs0**2 + 2.0*np.pi*G*r*rho0*rs0)/c0)

# ================= PhySO_47 =================
def PhySO_47_density(r, rho0, Rs, G):
    # Density
    return (c0 - 1.00000000000000*c0*rs0/r)*rho0*rs0/r

def PhySO_47_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 - 4*np.pi*c0*r*rho0*rs0**2

def PhySO_47_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 - 2*np.pi*G*c0*rho0*rs0**2)

# ================= PhySO_48 =================
def PhySO_48_density(r, rho0, Rs, G):
    # Density
    return -rho0*rs0**2*(c0/r - 1/rs0)/r

def PhySO_48_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return -4*np.pi*c0*r*rho0*rs0**2 + 2*np.pi*r**2*rho0*rs0

def PhySO_48_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(-2*np.pi*G*c0*rho0*rs0**2 + np.pi*G*r*rho0*rs0)

# ================= PhySO_49 =================
def PhySO_49_density(r, rho0, Rs, G):
    # Density
    return c0*(r - rs0)*rho0*rs0/r**2

def PhySO_49_mass(r, rho0, Rs, G):
    # Enclosed Mass
    return 2*np.pi*c0*r**2*rho0*rs0 - 4*np.pi*c0*r*rho0*rs0**2

def PhySO_49_circular_velocity(r, rho0, Rs, G):
    # Circular Velocity
    return np.sqrt(2)*np.sqrt(np.pi*G*c0*r*rho0*rs0 - 2*np.pi*G*c0*rho0*rs0**2)

