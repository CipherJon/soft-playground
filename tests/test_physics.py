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

    def test_zero_particles(self):
        """Test behavior with zero particles."""
        # Initialize with zero particles
        self.assertEqual(len(self.engine.particles), 0)
        self.assertEqual(len(self.engine.constraints), 0)

        # Test that operations on zero particles raise appropriate errors
        with self.assertRaises(RuntimeError):
            self.engine.update()  # Should raise error for uninitialized arrays

    def test_single_particle(self):
        """Test behavior with a single particle."""
        positions = [np.array([0.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)

        self.assertEqual(len(self.engine.particles), 1)

        # Single particle should move due to gravity
        initial_position = self.engine.get_positions()[0].copy()
        self.engine.update()
        updated_position = self.engine.get_positions()[0]

        self.assertFalse(np.array_equal(initial_position, updated_position))

    def test_extreme_mass_values(self):
        """Test behavior with extreme mass values."""
        positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        masses = [1e-6, 1e6]  # Very small and very large masses
        self.engine.initialize_particles(positions, masses)

        # Test that extreme masses don't cause numerical issues
        self.engine.update()
        updated_positions = self.engine.get_positions()

        # Positions should still be finite
        for pos in updated_positions:
            self.assertTrue(np.all(np.isfinite(pos)))

    def test_extreme_gravity(self):
        """Test behavior with extreme gravity values."""
        # Test with very high gravity
        self.config.gravity = (0.0, -1000.0, 0.0)
        self.engine = PhysicsEngine(self.config)

        positions = [np.array([0.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)

        initial_position = self.engine.get_positions()[0].copy()
        self.engine.update()
        updated_position = self.engine.get_positions()[0]

        # Particle should move significantly
        self.assertFalse(np.array_equal(initial_position, updated_position))

        # Position should still be finite
        self.assertTrue(np.all(np.isfinite(updated_position)))

    def test_zero_time_step(self):
        """Test behavior with zero time step."""
        self.config.time_step = 0.0
        self.engine = PhysicsEngine(self.config)

        positions = [np.array([0.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)

        initial_position = self.engine.get_positions()[0].copy()
        self.engine.update()
        updated_position = self.engine.get_positions()[0]

        # With zero time step, position should not change
        np.testing.assert_array_almost_equal(initial_position, updated_position)

    def test_invalid_constraint_indices(self):
        """Test behavior with invalid constraint indices."""
        positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)

        # Test constraint with invalid index - this should not raise an error
        # as the physics engine doesn't validate constraint indices
        self.engine.set_constraints([(0, 5)])  # Index 5 doesn't exist
        # The constraint will simply be ignored during updates

    def test_duplicate_constraints(self):
        """Test behavior with duplicate constraints."""
        positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ]
        self.engine.initialize_particles(positions)

        # Add duplicate constraints
        self.engine.set_constraints([(0, 1), (1, 2), (0, 1)])  # (0,1) is duplicated

        # Should not crash and should handle duplicates gracefully
        self.engine.update()

    def test_numerical_stability_small_time_step(self):
        """Test numerical stability with very small time step."""
        self.config.time_step = 1e-6
        self.engine = PhysicsEngine(self.config)

        positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)
        self.engine.set_constraints([(0, 1)])

        # Run multiple updates with small time step
        for _ in range(100):
            self.engine.update()

        # Positions should remain finite
        for pos in self.engine.get_positions():
            self.assertTrue(np.all(np.isfinite(pos)))

    def test_numerical_stability_large_time_step(self):
        """Test numerical stability with very large time step."""
        self.config.time_step = 1.0
        self.engine = PhysicsEngine(self.config)

        positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)
        self.engine.set_constraints([(0, 1)])

        # Run update with large time step
        self.engine.update()

        # Positions should remain finite
        for pos in self.engine.get_positions():
            self.assertTrue(np.all(np.isfinite(pos)))

    def test_extreme_spring_constant(self):
        """Test behavior with extreme spring constant."""
        self.config.spring_constant = 1000.0
        self.engine = PhysicsEngine(self.config)

        positions = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
        self.engine.initialize_particles(positions)
        self.engine.set_constraints([(0, 1)])

        # Run update with extreme spring constant
        self.engine.update()

        # Positions should remain finite
        for pos in self.engine.get_positions():
            self.assertTrue(np.all(np.isfinite(pos)))


if __name__ == "__main__":
    unittest.main()
