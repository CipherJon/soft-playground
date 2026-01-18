import os
import sys
import unittest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from physics import PhysicsEngine


class TestPhysicsEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PhysicsEngine()

    def test_gravity(self):
        """Test if gravity is applied correctly."""
        particles = [np.array([0.0, 0.0, 0.0])]
        constraints = []

        initial_position = particles[0].copy()
        self.engine.update(particles, constraints)

        # Check if gravity and damping have been applied
        expected_position = (
            initial_position + self.engine.gravity * 0.01
        ) * self.engine.damping
        np.testing.assert_array_almost_equal(particles[0], expected_position)

    def test_spring_force(self):
        """Test if spring forces are applied correctly."""
        particles = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        constraints = [(0, 1)]

        initial_positions = [p.copy() for p in particles]
        self.engine.update(particles, constraints)

        # Check if particles have moved due to spring force
        for initial, updated in zip(initial_positions, particles):
            self.assertFalse(np.array_equal(initial, updated))

    def test_damping(self):
        """Test if damping is applied correctly."""
        particles = [np.array([1.0, 1.0, 1.0])]
        constraints = []

        initial_position = particles[0].copy()
        self.engine.update(particles, constraints)

        # Check if damping has been applied
        expected_position = (
            initial_position + self.engine.gravity * 0.01
        ) * self.engine.damping
        np.testing.assert_array_almost_equal(particles[0], expected_position)


if __name__ == "__main__":
    unittest.main()
