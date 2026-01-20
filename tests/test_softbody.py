"""
Test cases for the soft body simulation.
"""

import sys
import unittest
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import numpy as np

from config import SimulationConfigManager
from softbody import SoftBodySimulation


class TestSoftBodySimulation(unittest.TestCase):
    def setUp(self):
        self.sim = SoftBodySimulation()

    def test_initialization(self):
        """Test if the simulation initializes correctly."""
        self.assertEqual(self.sim.get_particle_count(), 3)
        self.assertEqual(len(self.sim.get_constraints()), 2)

    def test_particle_positions(self):
        """Test if particles are initialized with correct positions."""
        for i, particle in enumerate(self.sim.get_particles()):
            expected_position = np.array([i * 0.1, 0.0, 0.0])
            np.testing.assert_array_equal(particle, expected_position)

    def test_constraints(self):
        """Test if constraints are correctly set between particles."""
        for i, (p1, p2) in enumerate(self.sim.get_constraints()):
            self.assertEqual(p1, i)
            self.assertEqual(p2, i + 1)

    def test_update(self):
        """Test if the simulation updates correctly."""
        initial_positions = self.sim.get_particles()
        self.sim.update()
        updated_positions = self.sim.get_particles()

        # Check if positions have changed after update (except first particle which is fixed)
        for i, (initial, updated) in enumerate(
            zip(initial_positions, updated_positions)
        ):
            if i == 0:  # First particle is fixed
                np.testing.assert_array_almost_equal(initial, updated)
            else:
                self.assertFalse(np.array_equal(initial, updated))

    def test_custom_configuration(self):
        """Test simulation with custom configuration."""
        config = SimulationConfigManager()
        config.simulation.num_particles = 5
        config.simulation.initial_spacing = 0.2

        sim = SoftBodySimulation(config)

        self.assertEqual(sim.get_particle_count(), 5)

        # Check initial positions with custom spacing
        for i, particle in enumerate(sim.get_particles()):
            expected_position = np.array([i * 0.2, 0.0, 0.0])
            np.testing.assert_array_equal(particle, expected_position)

    def test_particle_mass(self):
        """Test particle mass management."""
        # Test getting mass
        mass = self.sim.get_particle_mass(0)
        self.assertEqual(mass, 2.0)

        # Test setting mass
        self.sim.set_particle_mass(0, 2.0)
        new_mass = self.sim.get_particle_mass(0)
        self.assertEqual(new_mass, 2.0)

        # Test invalid mass
        with self.assertRaises(ValueError):
            self.sim.set_particle_mass(0, -1.0)

        # Test invalid index
        with self.assertRaises(IndexError):
            self.sim.set_particle_mass(100, 1.0)

    def test_particle_position_setting(self):
        """Test setting particle positions."""
        new_position = np.array([5.0, 5.0, 0.0])
        self.sim.set_particle_position(0, new_position)

        updated_position = self.sim.get_particles()[0]
        np.testing.assert_array_equal(updated_position, new_position)

        # Test invalid index
        with self.assertRaises(IndexError):
            self.sim.set_particle_position(100, new_position)

    def test_zero_particles_configuration(self):
        """Test simulation with zero particles in configuration."""
        config = SimulationConfigManager()
        config.simulation.num_particles = 0

        sim = SoftBodySimulation(config)

        self.assertEqual(sim.get_particle_count(), 0)
        self.assertEqual(len(sim.get_constraints()), 0)

        # With zero particles, update should raise an error due to broadcasting issues
        with self.assertRaises(ValueError):
            sim.update()

    def test_single_particle_configuration(self):
        """Test simulation with a single particle in configuration."""
        config = SimulationConfigManager()
        config.simulation.num_particles = 1

        sim = SoftBodySimulation(config)

        self.assertEqual(sim.get_particle_count(), 1)
        self.assertEqual(len(sim.get_constraints()), 0)

        # Single particle should not move (no constraints)
        initial_position = sim.get_particles()[0].copy()
        sim.update()
        updated_position = sim.get_particles()[0]

        np.testing.assert_array_almost_equal(initial_position, updated_position)

    def test_extreme_initial_spacing(self):
        """Test simulation with extreme initial spacing values."""
        # Test with very large spacing
        config = SimulationConfigManager()
        config.simulation.num_particles = 3
        config.simulation.initial_spacing = 100.0

        sim = SoftBodySimulation(config)

        # Check that particles are initialized with large spacing
        positions = sim.get_particles()
        distance_0_1 = np.linalg.norm(positions[1] - positions[0])
        distance_1_2 = np.linalg.norm(positions[2] - positions[1])

        self.assertAlmostEqual(distance_0_1, 100.0, places=5)
        self.assertAlmostEqual(distance_1_2, 100.0, places=5)

        # Test with very small spacing
        config.simulation.initial_spacing = 0.001
        sim_small = SoftBodySimulation(config)

        positions_small = sim_small.get_particles()
        distance_small = np.linalg.norm(positions_small[1] - positions_small[0])
        self.assertAlmostEqual(distance_small, 0.001, places=6)

    def test_invalid_particle_indices(self):
        """Test behavior with invalid particle indices."""
        # Test invalid index for mass operations
        with self.assertRaises(IndexError):
            self.sim.get_particle_mass(100)

        with self.assertRaises(IndexError):
            self.sim.set_particle_mass(100, 2.0)

        # Test invalid index for position operations
        with self.assertRaises(IndexError):
            self.sim.set_particle_position(100, np.array([1.0, 1.0, 1.0]))

    def test_extreme_mass_values(self):
        """Test behavior with extreme mass values."""
        # Test with very small mass
        self.sim.set_particle_mass(1, 1e-6)
        self.assertAlmostEqual(self.sim.get_particle_mass(1), 1e-6)

        # Test with very large mass
        self.sim.set_particle_mass(1, 1e6)
        self.assertAlmostEqual(self.sim.get_particle_mass(1), 1e6)

        # Simulation should still run without crashing
        self.sim.update()

    def test_numerical_stability_extreme_config(self):
        """Test numerical stability with extreme configuration values."""
        config = SimulationConfigManager()

        # Extreme physics parameters
        config.physics.gravity = (0.0, -1000.0, 0.0)
        config.physics.spring_constant = 1000.0
        config.physics.time_step = 0.1

        # Large number of particles
        config.simulation.num_particles = 20
        config.simulation.initial_spacing = 0.1

        sim = SoftBodySimulation(config)

        # Run multiple updates
        for _ in range(10):
            sim.update()

        # All positions should remain finite
        for pos in sim.get_particles():
            self.assertTrue(np.all(np.isfinite(pos)))

    def test_fixed_particle(self):
        """Test fixed particle behavior."""
        # Get initial position of first particle (should be fixed)
        initial_position = self.sim.get_particles()[0].copy()

        # Update simulation
        self.sim.update()

        # First particle should not move (it's fixed)
        updated_position = self.sim.get_particles()[0]
        np.testing.assert_array_almost_equal(initial_position, updated_position)


if __name__ == "__main__":
    unittest.main()
