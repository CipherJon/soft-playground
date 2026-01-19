"""
Physics engine for soft body simulation using Verlet integration.

This module implements a physics engine with support for gravity, damping, spring constraints,
and Verlet integration for improved stability over Euler integration.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import PhysicsConfig, default_config


@dataclass
class Particle:
    """Represents a particle with position, previous position, and mass."""

    position: np.ndarray
    previous_position: np.ndarray
    mass: float = 1.0
    is_fixed: bool = False


class PhysicsEngine:
    """Physics engine for soft body simulation."""

    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        Initialize the physics engine.

        Args:
            config: Physics configuration. If None, uses default configuration.
        """
        self.config = config if config is not None else default_config.physics
        self.particles: List[Particle] = []
        self.constraints: List[Tuple[int, int]] = []

        # Initialize integration method
        self._initialize_integration()

    def _initialize_integration(self):
        """Initialize integration method based on configuration."""
        if self.config.integration_method == "verlet":
            self._update_method = self._update_verlet
        else:
            self._update_method = self._update_euler

    def initialize_particles(
        self, positions: List[np.ndarray], masses: Optional[List[float]] = None
    ):
        """
        Initialize particles with given positions and masses.

        Args:
            positions: List of initial particle positions
            masses: List of particle masses. If None, uses default mass.
        """
        default_mass = 1.0

        self.particles = [
            Particle(
                position=pos.copy(),
                previous_position=pos.copy(),
                mass=masses[i] if masses and i < len(masses) else default_mass,
            )
            for i, pos in enumerate(positions)
        ]

    def set_constraints(self, constraints: List[Tuple[int, int]]):
        """
        Set constraints between particles.

        Args:
            constraints: List of tuples (i, j) representing constraints between particles
        """
        self.constraints = constraints

    def update(self):
        """Update the physics simulation by one time step."""
        self._update_method()

    def _update_verlet(self):
        """Update physics using Verlet integration."""
        time_step = self.config.time_step
        gravity = np.array(self.config.gravity)
        damping = self.config.damping

        # Apply Verlet integration
        for particle in self.particles:
            if particle.is_fixed:
                continue

            # Calculate acceleration (gravity for now)
            acceleration = gravity

            # Verlet integration: x_new = 2x_current - x_previous + acceleration * dt^2
            new_position = (
                2 * particle.position
                - particle.previous_position
                + acceleration * time_step * time_step
            )

            # Store current position as previous for next iteration
            particle.previous_position = particle.position.copy()
            particle.position = new_position

            # Apply damping
            particle.position *= damping

        # Apply constraints (springs)
        self._apply_spring_constraints()

    def _update_euler(self):
        """Update physics using simple Euler integration (fallback)."""
        time_step = self.config.time_step
        gravity = np.array(self.config.gravity)
        damping = self.config.damping

        # Apply Euler integration
        for particle in self.particles:
            if particle.is_fixed:
                continue

            # Simple Euler: x_new = x_current + velocity * dt
            # For simplicity, we'll use gravity as velocity
            particle.previous_position = particle.position.copy()
            particle.position += gravity * time_step

            # Apply damping
            particle.position *= damping

        # Apply constraints (springs)
        self._apply_spring_constraints()

    def _apply_spring_constraints(self):
        """Apply spring constraints between connected particles."""
        spring_constant = self.config.spring_constant
        rest_length = self.config.rest_length
        time_step = self.config.time_step

        for i, j in self.constraints:
            if i >= len(self.particles) or j >= len(self.particles):
                continue

            particle_i = self.particles[i]
            particle_j = self.particles[j]

            if particle_i.is_fixed and particle_j.is_fixed:
                continue

            # Calculate current distance
            delta = particle_j.position - particle_i.position
            current_distance = np.linalg.norm(delta)

            if current_distance == 0:
                continue

            # Calculate spring force (Hooke's law: F = -k * (x - x0))
            displacement = current_distance - rest_length
            force_magnitude = -spring_constant * displacement

            # Normalize direction vector
            direction = delta / current_distance

            # Apply force to both particles
            force_vector = force_magnitude * direction

            # Calculate accelerations based on mass
            if not particle_i.is_fixed:
                acceleration_i = force_vector / particle_i.mass
                particle_i.position += acceleration_i * time_step * time_step

            if not particle_j.is_fixed:
                acceleration_j = -force_vector / particle_j.mass
                particle_j.position += acceleration_j * time_step * time_step

    def get_positions(self) -> List[np.ndarray]:
        """
        Get current positions of all particles.

        Returns:
            List of particle positions
        """
        return [particle.position.copy() for particle in self.particles]

    def set_particle_fixed(self, index: int, is_fixed: bool):
        """
        Set whether a particle is fixed in space.

        Args:
            index: Index of the particle
            is_fixed: Whether the particle should be fixed
        """
        if 0 <= index < len(self.particles):
            self.particles[index].is_fixed = is_fixed

    def get_particle_mass(self, index: int) -> float:
        """
        Get the mass of a particle.

        Args:
            index: Index of the particle

        Returns:
            Mass of the particle

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < len(self.particles):
            raise IndexError(f"Particle index {index} out of range")
        return self.particles[index].mass

    def set_particle_mass(self, index: int, mass: float):
        """
        Set the mass of a particle.

        Args:
            index: Index of the particle
            mass: New mass for the particle

        Raises:
            IndexError: If index is out of range
            ValueError: If mass is not positive
        """
        if not 0 <= index < len(self.particles):
            raise IndexError(f"Particle index {index} out of range")
        if mass <= 0:
            raise ValueError("Mass must be positive")
        self.particles[index].mass = mass
