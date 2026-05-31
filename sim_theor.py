import numpy as np
import matplotlib.pyplot as plt

def theoretical_motion(t, dt, D):
    n_timesteps = int(t/dt)
    sdev = np.sqrt(2*D*dt)
    x_step = np.random.normal(0, sdev, n_timesteps)
    x = np.cumsum(x_step)
    y_step = np.random.normal(0, sdev, n_timesteps)
    y = np.cumsum(y_step)
    z_step = np.random.normal(0, sdev, n_timesteps)
    z = np.cumsum(z_step)
    return x, y, z

# sample for now
t = 10
dt = 10**(-2)
k = 1.380649 * 10**-23 #boltzmann
T = 100 #K
eta = 1 * 10**(-5) # can be computed later 
a = 10**(-6) # of the big particle
def eta(T):
    """argon using sutherland formula"""
    return 2.1*10**(-5) * (T/273.15)**(3/2) * (273.15 + 165)/(T + 165)
def D(T, a):
    return k*T/(6*np.pi*eta(T)*a)

x, y, z  = theoretical_motion(t, dt, D(T, a))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot3D(x, y, z)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.grid()
plt.title(f'Theoretical Brownian Motion over {t}s, dt = {dt}s, \n with T = {T}K and a = {a:.2e} m')
plt.show()