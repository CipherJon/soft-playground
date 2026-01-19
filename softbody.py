"""
Soft body simulation module.

This module implements a soft body simulation using a physics engine with Verlet integration,
spring constraints, and configurable parameters.
"""

from itertools import pairwise
from typing import List, Optional, Tuple

import numpy as np

import physics
from config import SimulationConfigManager, default_config


class SoftBodySimulation:
    """Soft body simulation with configurable physics and visualization."""

    def __init__(self, config: Optional[SimulationConfigManager] = None):
        """
        Initialize the soft body simulation.

        Args:
            config: Simulation configuration. If None, uses default configuration.
        """
        self.config = config if config is not None else default_config
        self.physics_engine = physics.PhysicsEngine(self.config.physics)

        # Initialize particles and constraints
        self._initialize_softbody()

    def _initialize_softbody(self):
        """Initialize particles and constraints based on configuration."""
        config = self.config.simulation

        # Create initial particle positions using list comprehension
        positions = [
            np.array([i * config.initial_spacing, 0.0, 0.0])
            for i in range(config.num_particles)
        ]

        # Initialize particles with masses using list comprehension
        masses = [config.particle_mass for _ in range(config.num_particles)]
        self.physics_engine.initialize_particles(positions, masses)

        # Add constraints (springs between particles) using itertools pairwise
        particle_indices = range(config.num_particles)
        constraints = list(pairwise(particle_indices))

        self.physics_engine.set_constraints(constraints)

        # Make the first particle fixed to create a pendulum-like effect
        if config.num_particles > 0:
            self.physics_engine.set_particle_fixed(0, True)

    def update(self):
        """Update the simulation by one time step."""
        self.physics_engine.update()

    def get_particles(self) -> List[np.ndarray]:
        """
        Get current positions of all particles.

        Returns:
            List of particle positions
        """
        return self.physics_engine.get_positions()

    def get_particle_count(self) -> int:
        """
        Get the number of particles in the simulation.

        Returns:
            Number of particles
        """
        return len(self.physics_engine.particles)

    def get_constraints(self) -> List[Tuple[int, int]]:
        """
        Get the constraints between particles.

        Returns:
            List of constraints as tuples (i, j)
        """
        return self.physics_engine.constraints.copy()

    def set_particle_position(self, index: int, position: np.ndarray):
        """
        Set the position of a specific particle.

        Args:
            index: Index of the particle
            position: New position for the particle

        Raises:
            IndexError: If index is out of range
        """
        if 0 <= index < len(self.physics_engine.particles):
            self.physics_engine.particles[index].position = position.copy()
            self.physics_engine.particles[index].previous_position = position.copy()
        else:
            raise IndexError(f"Particle index {index} out of range")

    def get_particle_mass(self, index: int) -> float:
        """
        Get the mass of a specific particle.

        Args:
            index: Index of the particle

        Returns:
            Mass of the particle

        Raises:
            IndexError: If index is out of range
        """
        return self.physics_engine.get_particle_mass(index)

    def set_particle_mass(self, index: int, mass: float):
        """
        Set the mass of a specific particle.

        Args:
            index: Index of the particle
            mass: New mass for the particle
        """
        return self.physics_engine.set_particle_mass(index, mass)
