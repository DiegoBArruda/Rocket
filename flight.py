import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Dados de voo

acceleration = []
mdot = []
force = []
arrasto = []

gamma = 1.4
R = 287.05
def interpolar_valor(x_input):
    x = np.loadtxt("mach.txt")
    y = np.loadtxt("cd.txt")
    return np.interp(x_input, x, y)

class flight():
    '''
    Class that simulates the flight of a rocket
    Internal objects:
    - rocket: rocket object
        - body: rocket body object
            - diameter: rocket diameter
            - length: rocket length
            - Cd: rocket drag coefficient
            - A: rocket cross-sectional area
        - chamber: rocket chamber object
            - P1: chamber pressure
            - P2: nozzle pressure
            - P3: ambient pressure
            - A_t: nozzle throat area
        - nozzle: rocket nozzle object
    - target: altitude target
    - env: environment object
    '''
    gamma = 1.4
    R = 287.05
    def __init__(self, rocket, target):
        self.parachute = False
        self.t_start = 0
        self.rocket = rocket
        self.target = target
        # Initialize the simulation
        self.rocket.altitude = 0
        self.rocket.v = 0
        self.rocket.acc = 0

        # Initialize the environment
        self.env = environment(0, 0, 0, 0)
    
    


    def force_update(self):
        # Evaluate the thrust, drag, and mass
        self.thrust_update()
        self.drag_update()

        thrust = self.rocket.thrust
        drag = self.rocket.drag
        mass = self.rocket.mass

        # Calculate the force
        force = thrust - drag - mass * 9.81
        self.rocket.force = force

    def pressure_update(self):
        rho_b = self.rocket.chamber.rho_comb
        n = 0.4
        alpha = 5e-6
        T_burn = 3300
        A_t = self.rocket.nozzle.A_t
        A_b = self.rocket.chamber.A_b

        c = np.sqrt(gamma * R * T_burn)/(gamma * (2/(gamma+1))**((gamma+1)/(2*(gamma-1))))
        P = (rho_b * alpha * A_b * c/ A_t)**(1/(1-n))
        self.rocket.chamber.P = P

        mdot = rho_b * A_b * alpha * P**n
        self.rocket.chamber.mdot = mdot

    def thrust_update(self):
        self.pressure_update()
        P_1 = self.rocket.chamber.P # Chamber pressure
        A_t = self.rocket.nozzle.A_t # Nozzle throat area
        P_2 = self.rocket.nozzle.Pe # Nozzle pressure (optimal)
        A2 = self.rocket.nozzle.Ae # Nozzle exit area
        P_3 = self.env.P # Ambient pressure

        # Calculate the thrust
        if self.rocket.mass_comb > 0:
            thrust = A_t * P_1 * np.sqrt(2 * gamma / (gamma - 1) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)) * (1- (P_2 / P_1) ** ((gamma - 1) / gamma))) + (P_2-P_3)*A2
        else:
            thrust = 0
            self.rocket.mass_comb  = 0
            self.rocket.mdot = 0
            self.rocket.chamber.mdot = 0
        self.rocket.thrust = thrust

    def drag_update(self):
        #Evaluate cd
        self.cd_update()

        # Calculate the drag
        if self.parachute:
            drag = 0.5 * 0.3 * 0.75 * self.env.rho * np.abs(self.rocket.v)*self.rocket.v
        else:
            drag = 0.5 * self.rocket.Cd * self.rocket.A * self.env.rho * np.abs(self.rocket.v)*self.rocket.v
        self.rocket.drag = drag

    def cd_update(self):
        # Evaluate the Mach number
        mach = self.rocket.v / self.env.a

        # Interpolate the Cd value
        self.rocket.Cd0 = interpolar_valor(mach)
        self.rocket.deltaCD = 0.004 *(self.rocket.length/self.rocket.diameter)*(self.rocket.length/self.rocket.diameter-5)
        self.rocket.Cd = self.rocket.Cd0 + self.rocket.deltaCD
        

    def env_update(self):
        h = self.rocket.altitude

        # Evaluate troposphere
        if h < 11000:
            T = (15.04+273.15) - 0.00649 * h
            P = 101.29 * ((T / 288.08) ** 5.256)
        
        # Evaluate stratosphere
        elif h < 25000:
            T = -56.46+273.15
            P = 22.65 * np.exp(1.73 - 0.000157 * h)

        # Evaluate high stratosphere
        elif h < 50000:
            T = (-131.21 + 273.15) + 0.00299 * h
            P = 2.488 * ((T / 216.6) ** -11.388)
        
        else:
            print("Error: Altitude out of range")
        
        rho = P / (0.2869 * T)

        # Update the environment
        self.env.T = T
        self.env.P = P * 1000
        self.env.rho = rho
        self.env.a = np.sqrt(gamma * R * self.env.T)


    def mass_update(self):
        self.rocket.mass = self.rocket.mass_comb + self.rocket.mass_struc

    def simulation(self, t, y):
        position = y[0]
        velocity = y[1]
        mass = y[2]
        # update all variables
        self.env_update()
        self.force_update()
        self.mass_update()
        if self.rocket.v < 0 and position < 500 and self.t_start == 0:
            self.t_start = t
            self.parachute = True
        if position < 0 and velocity < 0:
            position = 0
            velocity = 0
        self.rocket.altitude = position
        self.rocket.mass_comb = mass
        self.rocket.acc = self.rocket.force / self.rocket.mass
        self.rocket.v = velocity

        # Evaluate the mass flow rate
        mdot = self.rocket.chamber.mdot * -1

        # Update the velocity
        return self.rocket.v, self.rocket.acc, mdot
    






def pressure_chamber_calculation(M, Pamb):
    return Pamb/((1 + (gamma - 1) / 2 * M**2) ** (-gamma / (gamma - 1)))

def area_ratio_calculation(M):
    return np.sqrt(1/M**2 * (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M**2)) ** ((gamma + 1) / (gamma - 1)))


class chamber():
    def __init__(self, mdot, A_b, rho_comb, P):
        self.mdot = mdot
        self.A_b = A_b
        self.rho_comb = rho_comb
        self.P = P


class nozzle():
    def __init__(self, A_t, Pe, Ae):
        self.A_t = A_t
        self.Pe = Pe
        self.Ae = Ae

class rocket():
    def __init__(self, diameter, length, mass_comb, mass_struc, chamber, nozzle, rho_comb):
        self.diameter = diameter
        self.length = length
        self.mass_comb = mass_comb
        self.mass_struc = mass_struc
        self.chamber = chamber
        self.nozzle = nozzle
        self.Cd = 0
        self.Cd0 = 0
        self.deltaCD = 0
        self.A = np.pi * self.diameter ** 2 / 4
        self.altitude = 0
        self.v = 0
        self.acc = 0
        self.thrust = 0
        self.drag = 0
        self.mass = self.mass_comb + self.mass_struc
        self.rho = rho_comb



def rocket_assembly(Mach, Poptim):
    gamma = 1.4
    R = 287.05
    # Rocket body
    diameter = 0.5
    Abody = np.pi * diameter ** 2 / 4
    rho_comb = 1800 # Density of the rocket fuel (kg/m^3)
    mass_comb = 4 # Mass of the rocket fuel (kg)
    length = mass_comb / (rho_comb * Abody)
    mass_struc =  0.5 + 0.1*length*2   

    n = 0.4
    alpha = 5e-6
    T_burn = 3300

    #chamber
    A_b = Abody 
    P_exit = Poptim
    P0 = pressure_chamber_calculation(Mach, Poptim)
    mdot = rho_comb * A_b * alpha * P0**n
    c = np.sqrt(gamma * R * T_burn)/(gamma * (2/(gamma+1))**((gamma+1)/(2*(gamma-1))))
    A_t = mdot * c / P0
    T_burn = 3300
    
    

    # Rocket nozzle
    A_nozzle = area_ratio_calculation(Mach) * A_t
    # Rocket

    # Target
    target = 1000
    chamber1 = chamber(mdot, A_b, rho_comb, P0)
    return rocket(diameter, length, mass_comb, mass_struc, chamber1, nozzle(A_t, P_exit, A_nozzle), rho_comb), target

class environment():
    def __init__(self, T, P, rho, a):
        self.T = T
        self.P = P
        self.rho = rho
        self.a = a

def main():
    rocket, target = rocket_assembly(2, 101325)
    flight_simulation = flight(rocket, target)
    dt = 0.01
    position = 0
    velocity = 0
    mass = 4
    tempo = 0
    # Simulate with solve_ivp
    sol = solve_ivp(flight_simulation.simulation, [0, 150], [position, velocity, mass])
    time = sol.t
    altitude = sol.y[0]
    velocity = sol.y[1]
    mass = sol.y[2]

    # Plot the results
    plt.figure()
    plt.plot(time, altitude)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude vs Time")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, velocity)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs Time")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, mass)
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (kg)")
    plt.title("Mass vs Time")
    plt.grid()
    plt.show()

    print("Fim")
if __name__ == "__main__":
    main()