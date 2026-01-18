import numpy as np


class PhysicsEngine:
    def __init__(self):
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.damping = 0.99

    def update(self, particles, constraints):
        # Apply gravity
        for i, particle in enumerate(particles):
            particles[i] += self.gravity * 0.01  # Simple time step

        # Apply constraints (e.g., springs)
        for i, j in constraints:
            distance = np.linalg.norm(particles[i] - particles[j])
            # Simple spring force
            force = (distance - 1.0) * 0.1  # Spring constant
            direction = (particles[j] - particles[i]) / distance
            particles[i] += force * direction * 0.01
            particles[j] -= force * direction * 0.01

        # Apply damping
        for i, particle in enumerate(particles):
            particles[i] *= self.damping
