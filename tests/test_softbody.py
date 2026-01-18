import os
import sys
import unittest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from softbody import SoftBodySimulation


class TestSoftBodySimulation(unittest.TestCase):
    def setUp(self):
        self.sim = SoftBodySimulation()

    def test_initialization(self):
        """Test if the simulation initializes correctly."""
        self.assertEqual(len(self.sim.particles), 10)
        self.assertEqual(len(self.sim.constraints), 9)

    def test_particle_positions(self):
        """Test if particles are initialized with correct positions."""
        for i, particle in enumerate(self.sim.particles):
            expected_position = np.array([i * 0.1, 0.0, 0.0])
            np.testing.assert_array_equal(particle, expected_position)

    def test_constraints(self):
        """Test if constraints are correctly set between particles."""
        for i, (p1, p2) in enumerate(self.sim.constraints):
            self.assertEqual(p1, i)
            self.assertEqual(p2, i + 1)

    def test_update(self):
        """Test if the simulation updates correctly."""
        initial_positions = [p.copy() for p in self.sim.particles]
        self.sim.update()
        updated_positions = self.sim.particles

        # Check if positions have changed after update
        for initial, updated in zip(initial_positions, updated_positions):
            self.assertFalse(np.array_equal(initial, updated))


if __name__ == "__main__":
    unittest.main()
