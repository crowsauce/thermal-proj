import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Particle:
    def __init__(self, x, y, v_x, v_y, radius, m):
        self.r = np.array((x, y))
        self.v = np.array((v_x, v_y))
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

    def overlaps(self, other):
        """check if overlapping"""
        return np.linalg.norm(self.r - other.r) < self.radius + other.radius

def collide(p1, p2): # perfectly elastic
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

def initialize_particles(n, box_size, small_radius, small_mass, brownian_radius, brownian_mass):
    particles = []
    for _ in range(n):
        x = np.random.uniform( -(box_size - small_radius), (box_size - small_radius)) # maybe do something to prevent overlaps with other particles
        y = np.random.uniform( -(box_size - small_radius), (box_size - small_radius))
        v_x = np.random.normal(0, 1) # maxwell distribution later perchance
        v_y = np.random.normal(0, 1) 
        particles.append(Particle(x, y, v_x, v_y, small_radius, small_mass))
    particles.append(Particle(0, 0, 0, 0, brownian_radius, brownian_mass))
    return particles

def step_sim(particles, dt, box_size):
    """go by one timestep"""
    for i, p1 in enumerate(particles):
        for j, p2 in enumerate(particles):
            if i < j and p1.overlaps(p2):
                v1_final, v2_final = collide(p1, p2) # perfectly elastic
                p1.v = v1_final
                p2.v = v2_final
        reflect_wall(p1, box_size)
        p1.r = p1.r + p1.v * dt    # will update p2 when it gets to j in the first loop i think

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
        step_sim(particles, dt, box_size)
        for patch, p in zip(patches, particles):
            patch.center = (p.x, p.y)
        return patches
    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=interval, blit=True,)
    return anim

# test
particles = initialize_particles(100, 10, 0.1, 0.1, 1.0, 1.0)
anim = animate_sim(particles, dt=0.1, box_size=10, n_frames=200)
plt.show()