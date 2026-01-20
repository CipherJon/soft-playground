"""
Test cases for the physics engine.
"""

import sys
import unittest
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np

from config import PhysicsConfig
from physics import Particle, PhysicsEngine


class TestPhysicsEngine(unittest.TestCase):
    def setUp(self):
        self.config = PhysicsConfig()
        self.engine = PhysicsEngine(self.config)

    def test_initialization(self):
        """Test if physics engine initializes correctly."""
        self.assertEqual(len(self.engine.particles), 0)
        self.assertEqual(len(self.engine.constraints), 0)

    def test_particle_initialization(self):
        """Test particle initialization."""
        positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        masses = [1.0, 2.0]

        self.engine.initialize_particles(positions, masses)

        self.assertEqual(len(self.engine.particles), 2)
        self.assertAlmostEqual(self.engine.get_particle_mass(0), 1.0)
        self.assertAlmostEqual(self.engine.get_particle_mass(1), 2.0)

    def test_verlet_integration(self):
        """Test Verlet integration."""
        self.config.integration_method = "verlet"
        self.engine = PhysicsEngine(self.config)

        positions = [np.array([0.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)

        initial_position = self.engine.get_positions()[0].copy()
        self.engine.update()

        updated_position = self.engine.get_positions()[0]

        # Position should change due to gravity
        self.assertFalse(np.array_equal(initial_position, updated_position))

    def test_euler_integration(self):
        """Test Euler integration."""
        self.config.integration_method = "euler"
        self.engine = PhysicsEngine(self.config)

        positions = [np.array([0.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)

        initial_position = self.engine.get_positions()[0].copy()
        self.engine.update()

        updated_position = self.engine.get_positions()[0]

        # Position should change due to gravity
        self.assertFalse(np.array_equal(initial_position, updated_position))

    def test_spring_constraints(self):
        """Test spring constraints between particles."""
        positions = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)
        self.engine.set_constraints([(0, 1)])

        initial_positions = self.engine.get_positions()
        self.engine.update()

        updated_positions = self.engine.get_positions()

        # Particles should move due to spring force
        for initial, updated in zip(initial_positions, updated_positions):
            self.assertFalse(np.array_equal(initial, updated))

    def test_fixed_particle(self):
        """Test fixed particle behavior."""
        positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)
        self.engine.set_particle_fixed(0, True)

        initial_pos_0 = self.engine.get_positions()[0].copy()
        self.engine.update()
        updated_pos_0 = self.engine.get_positions()[0]

        # Fixed particle should not move
        np.testing.assert_array_almost_equal(initial_pos_0, updated_pos_0)

    def test_mass_effects(self):
        """Test that mass affects particle behavior."""
        # Heavy particle
        positions_heavy = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        masses_heavy = [10.0, 1.0]
        self.engine.initialize_particles(positions_heavy, masses_heavy)
        self.engine.set_constraints([(0, 1)])

        # Light particle
        positions_light = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        masses_light = [1.0, 10.0]
        engine_light = PhysicsEngine(self.config)
        engine_light.initialize_particles(positions_light, masses_light)
        engine_light.set_constraints([(0, 1)])

        # Update both
        self.engine.update()
        engine_light.update()

        # Different mass distributions should result in different movements
        heavy_positions = self.engine.get_positions()
        light_positions = engine_light.get_positions()

        self.assertFalse(np.array_equal(heavy_positions[0], light_positions[0]))
        self.assertFalse(np.array_equal(heavy_positions[1], light_positions[1]))

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        positions = [np.array([0.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)

        # Test invalid mass
        with self.assertRaises(ValueError):
            self.engine.set_particle_mass(0, -1.0)

        # Test invalid index
        with self.assertRaises(IndexError):
            self.engine.set_particle_mass(1, 1.0)


if __name__ == "__main__":
    unittest.main()
