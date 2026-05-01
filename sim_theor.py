import numpy as np
import matplotlib.pyplot as plt

def theoretical_motion(t, dt, D):
    n_timesteps = int(t/dt)
    sdev = np.sqrt(2*D*dt)
    x_step = np.random.normal(0, sdev, n_timesteps)
    x = np.cumsum(x_step)
    y_step = np.random.normal(0, sdev, n_timesteps)
    y = np.cumsum(y_step)
    return x, y

# sample for now
t = 10
dt = 10**(-2)
k = 1.380649 * 10**-23 #boltzmann
T = 100 #K
eta = 1 * 10**(-5) # can be computed later 
a = 1* 10**(-2)
D = k*T/(6*np.pi*eta*a)

x, y  = theoretical_motion(t, dt, D)

plt.plot(x, y)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title(f'Theoretical Brownian Motion \n with t = {t}s, dt = {dt}s, D = {D:.2e} m^2/s')
plt.show()