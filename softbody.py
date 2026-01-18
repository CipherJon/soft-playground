import numpy as np

import physics


class SoftBodySimulation:
    def __init__(self):
        self.particles = []
        self.constraints = []
        self.physics_engine = physics.PhysicsEngine()

        # Initialize particles and constraints
        self._initialize_softbody()

    def _initialize_softbody(self):
        # Example: Create a simple softbody with particles and constraints
        num_particles = 10
        for i in range(num_particles):
            self.particles.append(np.array([i * 0.1, 0.0, 0.0]))

        # Add constraints (e.g., springs between particles)
        for i in range(num_particles - 1):
            self.constraints.append((i, i + 1))

    def update(self):
        # Update physics
        self.physics_engine.update(self.particles, self.constraints)

    def get_particles(self):
        return self.particles
