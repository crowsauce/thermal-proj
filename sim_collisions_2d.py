import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy

m_s = 6.63 * 10**(-26) # Ar kg
r_s = 1.88 * 10**(-10) # Ar m
k = 1.380649 * 10**-23
def m_big(a):
    return 4/3 * np.pi * a**3 * 2500 # kg
lj_e = 143.78 * k # Ar J
lj_s = 3.3237 * 10**(-10) # Ar m

r_s_nm = 1.88 * 10**(-1) # Ar nm
lj_s_nm = 3.3237 * 10**(-1) # Ar nm
def m_big_nm(a):
    return 4/3 * np.pi * a**3 * 2500 * 10**(-27) # kg from nm

p_atm = 101325 # Pa
def density(m_particle, T, atm_frac): # assumes ideal gas law
    return atm_frac*p_atm * m_particle / (k * T)
def n_given_rho_v(rho, box_size, m_particle):
    return int(rho * (2*box_size)**2 / m_particle)
#idk if these still work lowk

class Particle:
    def __init__(self, x, y, v_x, v_y, a_x, a_y, radius, m):
        self.r = np.array((x, y))
        self.v = np.array((v_x, v_y))
        self.a = np.array((a_x, a_y))
        self.radius = radius
        self.m = m

    @property
    def x(self):
        """check the x component of position"""
        return self.r[0]
    @property
    def y(self):
        """check the y component of position"""
        return self.r[1]
    @property
    def v_x(self):
        """check the x component of velocity"""
        return self.v[0]
    @property
    def v_y(self):
        """check the y component of velocity"""
        return self.v[1]
    @property
    def a_x(self):
        """check the x component of acceleration"""
        return self.a[0]
    @property
    def a_y(self):
        """check the y component of acceleration"""
        return self.a[1]
    @x.setter
    def x(self, value):
        """set the x component of position"""
        self.r[0] = value
    @y.setter
    def y(self, value):
        """set the y component of position"""
        self.r[1] = value
    @v_x.setter
    def v_x(self, value):
        """set the x component of velocity"""
        self.v[0] = value
    @v_y.setter
    def v_y(self, value):
        """set the y component of velocity"""
        self.v[1] = value
    @a_x.setter
    def a_x(self, value):
        """set the x component of acceleration"""
        self.a[0] = value
    @a_y.setter
    def a_y(self, value):
        """set the y component of acceleration"""
        self.a[1] = value
    def overlaps(self, other):
        """check if overlapping"""
        return np.linalg.norm(self.r - other.r) < self.radius + other.radius

def collide(p1, p2): # perfectly elastic, prollynot using ts!
    v1_final = p1.v - 2 * p2.m / (p1.m + p2.m) * np.dot(p1.v - p2.v, p1.r - p2.r) / np.linalg.norm(p1.r - p2.r)**2 * (p1.r - p2.r)
    v2_final = p2.v - 2 * p1.m / (p1.m + p2.m) * np.dot(p2.v - p1.v, p2.r - p1.r) / np.linalg.norm(p1.r - p2.r)**2 * (p2.r - p1.r)
    return v1_final, v2_final

def reflect_wall(p, box_size):
    if p.x + p.radius > box_size:
        p.x = box_size - p.radius
        p.v_x *= -1
    if -p.x + p.radius > box_size:
        p.x = -box_size + p.radius
        p.v_x *= -1
    if p.y + p.radius > box_size:
        p.y = box_size - p.radius
        p.v_y *= -1
    if -p.y + p.radius > box_size:
        p.y = -box_size + p.radius
        p.v_y *= -1

def initialize_particles(n, box_size, small_radius, small_mass, brownian_radius, brownian_mass, T):
    k = 1.380649 * 10**-23 #boltzmann
    particles = []
    for _ in range(n):
        x = np.random.uniform( -(box_size - small_radius), (box_size - small_radius)) # maybe do something to prevent overlaps with other particles
        y = np.random.uniform( -(box_size - small_radius), (box_size - small_radius))
        r = scipy.stats.uniform_direction.rvs(2)*scipy.stats.maxwell.rvs(scale=np.sqrt(k*T/small_mass), size=1)[0]
        v_x, v_y = r[0], r[1]
        a_x = 0
        a_y = 0
        particles.append(Particle(x, y, v_x, v_y, a_x, a_y, small_radius, small_mass))
    r_brownian = scipy.stats.uniform_direction.rvs(2)*scipy.stats.maxwell.rvs(scale=np.sqrt(k*T/brownian_mass), size=1)[0]
    bv_x, bv_y = r_brownian[0], r_brownian[1]
    particles.append(Particle(0, 0, 0,0, 0, 0, brownian_radius, brownian_mass))
    return particles

#in m
def step_sim_elastic(particles, dt, box_size):
    """go by one timestep"""
    for i, p1 in enumerate(particles):
        for j, p2 in enumerate(particles):
            if i < j and p1.overlaps(p2):
                v1_final, v2_final = collide(p1, p2) # perfectly elastic
                p1.v = v1_final
                p2.v = v2_final
        reflect_wall(p1, box_size)
        p1.r = p1.r + p1.v * dt    # will update p2 when it gets to j in the first loop i think

def lj(p1, p2):
    r = np.linalg.norm(p1.r - p2.r)
    if r < 5*lj_s and 0 < r: # cutoff distance
        F = 24*lj_e/lj_s * (2*(lj_s/r)**13 - (lj_s/r)**7) * (p1.r - p2.r)/r
    else:
        F = np.array((0, 0))
    return F

def step_sim_lj(particles, dt, box_size):
    """go by one timestep"""
    for p1 in particles:
        p1.r = p1.r + p1.v * dt
        p1.v = p1.v + p1.a * dt
    for p1 in particles[:-1]: # lj for fluid
        total_f = np.array((0, 0))
        for p2 in particles[:-1]:
            total_f = total_f + lj(p1, p2)
        p1.a = total_f / p1.m
    for p1 in particles[:-1]: # elastic for brownian
        if p1.overlaps(particles[-1]):
            v1_final, v_brownian_final = collide(p1, particles[-1])
            p1.v = v1_final
            particles[-1].v = v_brownian_final
        reflect_wall(p1, box_size)
    
def run_sim(particles, dt, box_size, n_steps):
    brownian_positions_x = []
    brownian_positions_y = []
    for _ in range(n_steps):
        step_sim_lj(particles, dt, box_size)
        brownian_positions_x.append(particles[-1].x)
        brownian_positions_y.append(particles[-1].y)
    x, y = brownian_positions_x, brownian_positions_y
    return x, y

def animate_sim(particles, dt, box_size, n_frames, interval=30):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)
    patches = []
    for p in particles: #use enumerate to change colour on brownian particle
        patch = plt.Circle((p.x, p.y), p.radius, fc="C0", ec="k", alpha=0.8, animated=True)
        ax.add_patch(patch)
        patches.append(patch)
    def init():
        for patch, p in zip(patches, particles):
            patch.center = (p.x, p.y)
        return patches
    def update(frame):
        step_sim_lj(particles, dt, box_size)
        for patch, p in zip(patches, particles):
            patch.center = (p.x, p.y)
        return patches
    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=interval, blit=True,)
    return anim #looks like its overlapping bc its in 3d


def test_anim():
    particles = initialize_particles(100, 10**(-4), r_s, m_s, 10**(-6), m_big(10**(-6)), 10)
    anim = animate_sim(particles, dt=0.1, box_size=10**(-4), n_frames=200)
    plt.show()

def test_anim_notrealistic():
    particles = initialize_particles(100, 10, 0.1, 10**(-20), 1, 10**(-15), 300)
    anim = animate_sim(particles, dt=0.1, box_size=10, n_frames=200)
    plt.show()

#nm
def lj_nm(p1, p2):
    r = np.linalg.norm(p1.r - p2.r)
    if r < 5*lj_s_nm and 0 < r: # cutoff distance
        F = 24*lj_e/lj_s_nm * (2*(lj_s_nm/r)**13 - (lj_s_nm/r)**7) * (p1.r - p2.r)/r
    else:
        F = np.array((0, 0))
    return F

def step_sim_lj_nm(particles, dt, box_size):
    """go by one timestep"""
    for p1 in particles:
        p1.r = p1.r + p1.v * dt
        p1.v = p1.v + p1.a * dt
    for p1 in particles:
        total_f = np.array((0, 0))
        for p2 in particles:
            total_f = total_f + lj_nm(p1, p2)
        p1.a = total_f / p1.m
        reflect_wall(p1, box_size)

def run_sim_nm(particles, dt, box_size, n_steps):
    brownian_positions_x = []
    brownian_positions_y = []
    for n in range(n_steps):
        step_sim_lj_nm(particles, dt, box_size)
        brownian_positions_x.append(particles[-1].x)
        brownian_positions_y.append(particles[-1].y)
    x, y = brownian_positions_x, brownian_positions_y
    return x, y

def plot_sim(x, y, box_size, dt, n_steps):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid()
    plt.title(f'Simulated Brownian Motion over {n_steps*dt} s, dt = {dt}s, \n with box size = {2*box_size} m, particle number = {n}')
    plt.show()
    
# set variables
n, box_size, small_radius, small_mass, brownian_radius, brownian_mass, dt, n_steps = 100, 10**(-5), 10**(-8), 10**(-20), 10**(-6), 10**(-16), 10**(-7), 10**(3)

set_particles_100 = initialize_particles(n, box_size, small_radius, small_mass, brownian_radius, brownian_mass, 100)
set_particles_300 = initialize_particles(n, box_size, small_radius, small_mass, brownian_radius, brownian_mass, 300)
set_particles_1000 = initialize_particles(n, box_size, small_radius, small_mass, brownian_radius, brownian_mass, 1000)


x, y = run_sim(set_particles_1000, dt, box_size, n_steps)
plot_sim(x, y, box_size, dt, n_steps)


#test_anim_notrealistic()

def run_per_time(particles, dt, box_size, n_steps):
    disp_div_time = []
    for n in range(n_steps):
        step_sim_lj(particles, dt, box_size)
        disp_div_time.append(np.linalg.norm(particles[-1].r)/((n+1)*dt))
    return np.array([np.mean(disp_div_time[-100:]), np.std(disp_div_time[-100:]), np.mean(disp_div_time), np.std(disp_div_time)])

def data(particles, dt, box_size, n_steps):
    data_set = np.empty((0, 4))
    for _ in range(5):
        data_set = np.vstack((data_set, run_per_time(particles, dt, box_size, n_steps)))
    return data_set

#data_set_100 = data(set_particles_300, dt, box_size, n_steps)
#print(data_set_100)

