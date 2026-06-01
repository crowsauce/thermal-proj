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
    return int(rho * (2*box_size)**3 / m_particle)


class Particle:
    def __init__(self, x, y, z, v_x, v_y, v_z, a_x, a_y, a_z, radius, m):
        self.r = np.array((x, y, z))
        self.v = np.array((v_x, v_y, v_z))
        self.a = np.array((a_x, a_y, a_z))
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
    def z(self):
        """check the z component of position"""
        return self.r[2]
    @property
    def v_x(self):
        """check the x component of velocity"""
        return self.v[0]
    @property
    def v_y(self):
        """check the y component of velocity"""
        return self.v[1]
    @property
    def v_z(self):
        """check the z component of velocity"""
        return self.v[2]
    @property
    def a_x(self):
        """check the x component of acceleration"""
        return self.a[0]
    @property
    def a_y(self):
        """check the y component of acceleration"""
        return self.a[1]
    @property
    def a_z(self):
        """check the z component of acceleration"""
        return self.a[2]
    @x.setter
    def x(self, value):
        """set the x component of position"""
        self.r[0] = value
    @y.setter
    def y(self, value):
        """set the y component of position"""
        self.r[1] = value
    @z.setter
    def z(self, value):
        """set the z component of position"""
        self.r[2] = value
    @v_x.setter
    def v_x(self, value):
        """set the x component of velocity"""
        self.v[0] = value
    @v_y.setter
    def v_y(self, value):
        """set the y component of velocity"""
        self.v[1] = value
    @v_z.setter
    def v_z(self, value):
        """set the z component of velocity"""
        self.v[2] = value
    @a_x.setter
    def a_x(self, value):
        """set the x component of acceleration"""
        self.a[0] = value
    @a_y.setter
    def a_y(self, value):
        """set the y component of acceleration"""
        self.a[1] = value
    @a_z.setter
    def a_z(self, value):
        """set the z component of acceleration"""
        self.a[2] = value
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
    if p.z + p.radius > box_size:
        p.z = box_size - p.radius
        p.v_z *= -1
    if -p.z + p.radius > box_size:
        p.z = -box_size + p.radius
        p.v_z *= -1

def initialize_particles(n, box_size, small_radius, small_mass, brownian_radius, brownian_mass, T):
    k = 1.380649 * 10**-23 #boltzmann
    particles = []
    for _ in range(n):
        x = np.random.uniform( -(box_size - small_radius), (box_size - small_radius)) # maybe do something to prevent overlaps with other particles
        y = np.random.uniform( -(box_size - small_radius), (box_size - small_radius))
        z = np.random.uniform( -(box_size - small_radius), (box_size - small_radius))
        r = scipy.stats.uniform_direction.rvs(3)*scipy.stats.maxwell.rvs(scale=np.sqrt(k*T/small_mass), size=1)[0]
        v_x, v_y, v_z = r[0], r[1], r[2]
        a_x = 0
        a_y = 0
        a_z = 0
        particles.append(Particle(x, y, z, v_x, v_y, v_z, a_x, a_y, a_z, small_radius, small_mass))
    r_brownian = scipy.stats.uniform_direction.rvs(3)*scipy.stats.maxwell.rvs(scale=np.sqrt(k*T/brownian_mass), size=1)[0]
    bv_x, bv_y, bv_z = r_brownian[0], r_brownian[1], r_brownian[2]
    particles.append(Particle(0, 0, 0, bv_x, bv_y, bv_z, 0, 0, 0, brownian_radius, brownian_mass))
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
        F = np.array((0, 0, 0))
    return F

def step_sim_lj(particles, dt, box_size):
    """go by one timestep"""
    for p1 in particles:
        p1.r = p1.r + p1.v * dt
        p1.v = p1.v + p1.a * dt
    for p1 in particles[:-1]: # lj for fluid
        total_f = np.array((0, 0, 0))
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
    brownian_positions_z = []
    for _ in range(n_steps):
        step_sim_lj(particles, dt, box_size)
        brownian_positions_x.append(particles[-1].x)
        brownian_positions_y.append(particles[-1].y)
        brownian_positions_z.append(particles[-1].z)
    x, y, z = brownian_positions_x, brownian_positions_y, brownian_positions_z
    return x, y, z

# prolly not using
def animate_sim_2D(particles, dt, box_size, n_frames, interval=30):
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

def animate_sim_3D(particles, dt, box_size, n_frames, interval=30):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.set_aspect("equal")
    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)
    ax.set_zlim(-box_size, box_size)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    def x_base(p):
        return p.radius*np.outer(np.cos(u), np.sin(v))
    def y_base(p):
        return p.radius*np.outer(np.sin(u), np.sin(v))
    def z_base(p):
        return p.radius*np.outer(np.ones(np.size(u)), np.cos(v))
    spheres = []
    for p in particles:
        x = x_base(p) + p.x
        y = y_base(p) + p.y
        z = z_base(p) + p.z
        surf = ax.plot_surface(x, y, z)
        spheres.append(surf)
    def update(frame):
        step_sim_lj(particles, dt, box_size)
        for sphere in spheres:
            sphere.remove() # remove the old sphere
            spheres.clear()
            x = x_base(p) + p.x
            y = y_base(p) + p.y
            z = z_base(p) + p.z
            surf = ax.plot_surface(x, y, z)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    return anim

def test_anim():
    particles = initialize_particles(100, 10**(-4), r_s, m_s, 10**(-6), m_big(10**(-6)), 10)
    anim = animate_sim_2D(particles, dt=0.1, box_size=10**(-4), n_frames=200)
    plt.show()

def test_anim_notrealistic():
    particles = initialize_particles(100, 10, 0.1, 10**(-20), 1, 10**(-15), 300)
    anim = animate_sim_2D(particles, dt=0.1, box_size=2, n_frames=200)
    plt.show()

#using
def lj_nm(p1, p2):
    r = np.linalg.norm(p1.r - p2.r)
    if r < 5*lj_s_nm and 0 < r: # cutoff distance
        F = 24*lj_e/lj_s_nm * (2*(lj_s_nm/r)**13 - (lj_s_nm/r)**7) * (p1.r - p2.r)/r
    else:
        F = np.array((0, 0, 0))
    return F

def step_sim_lj_nm(particles, dt, box_size):
    """go by one timestep"""
    for p1 in particles:
        p1.r = p1.r + p1.v * dt
        p1.v = p1.v + p1.a * dt
    for p1 in particles:
        total_f = np.array((0, 0, 0))
        for p2 in particles:
            total_f = total_f + lj_nm(p1, p2)
        p1.a = total_f / p1.m
        reflect_wall(p1, box_size)

def run_sim_nm(particles, dt, box_size, n_steps):
    brownian_positions_x = []
    brownian_positions_y = []
    brownian_positions_z = []
    for _ in range(n_steps):
        step_sim_lj_nm(particles, dt, box_size)
        brownian_positions_x.append(particles[-1].x)
        brownian_positions_y.append(particles[-1].y)
        brownian_positions_z.append(particles[-1].z)
    x, y, z = brownian_positions_x, brownian_positions_y, brownian_positions_z
    return x, y, z

def plot_sim(x, y, z, box_size, dt, n_steps):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot3D(x, y, z)
    ax.set_xlabel('X Position (nm)')
    ax.set_ylabel('Y Position (nm)')
    ax.set_zlabel('Z Position (nm)')
    ax.grid()
    plt.title(f'Simulated Brownian Motion over {n_steps*dt} s')
    plt.show()

N = n_given_rho_v(density(m_s, 300, 10**(-6)), 10**(3)*10**(-9), m_s)
print(N)
#x, y, z = run_sim_nm(initialize_particles(N, 10**(5), r_s_nm, m_s, 10**(2), m_big_nm(10**(2)), 300), dt=0.01, box_size=10**(5), n_steps=100)
#plot_sim(x, y, z, box_size=10**(5), dt=0.01, n_steps=100)

test_anim_notrealistic()