"""
Physics engine for soft body simulation using Verlet integration with NumPy vectorization.

This module implements a physics engine with support for gravity, damping, spring constraints,
and Verlet integration using vectorized NumPy operations for improved performance.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeGuard

import numpy as np

from config import PhysicsConfig, default_config


class Particle:
    """Lightweight particle representation that acts as a view into array storage."""

    __slots__ = ("_engine", "_index")

    def __init__(self, engine, index: int):
        """
        Initialize a particle as a view into engine's array storage.

        Args:
            engine: PhysicsEngine instance
            index: Index of this particle in the engine's arrays
        """
        self._engine = engine
        self._index = index

    @property
    def position(self) -> np.ndarray:
        """Get current position."""
        return self._engine._positions[self._index]

    @position.setter
    def position(self, value: np.ndarray):
        """Set current position."""
        self._engine._positions[self._index] = value

    @property
    def previous_position(self) -> np.ndarray:
        """Get previous position."""
        return self._engine._previous_positions[self._index]

    @previous_position.setter
    def previous_position(self, value: np.ndarray):
        """Set previous position."""
        self._engine._previous_positions[self._index] = value

    @property
    def mass(self) -> float:
        """Get mass."""
        return self._engine._masses[self._index]

    @mass.setter
    def mass(self, value: float):
        """Set mass."""
        self._engine._masses[self._index] = value

    @property
    def is_fixed(self) -> bool:
        """Get fixed status."""
        return self._engine._fixed_mask[self._index]

    @is_fixed.setter
    def is_fixed(self, value: bool):
        """Set fixed status."""
        self._engine._fixed_mask[self._index] = value


class PhysicsEngine:
    """Physics engine for soft body simulation using vectorized operations."""

    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        Initialize the physics engine.

        Args:
            config: Physics configuration. If None, uses default configuration.
        """
        self.config = config if config is not None else default_config.physics
        self.particles: List[Particle] = []
        self.constraints: List[Tuple[int, int]] = []

        # Vectorized storage for performance (primary storage)
        self._positions = None  # Shape: (n_particles, 3)
        self._previous_positions = None  # Shape: (n_particles, 3)
        self._masses = None  # Shape: (n_particles,)
        self._fixed_mask = None  # Shape: (n_particles,), boolean

        # Initialize integration method
        self._initialize_integration()

    def _initialize_integration(self):
        """Initialize integration method based on configuration."""
        if self.config.integration_method == "verlet":
            self._update_method = self._update_verlet_vectorized
        else:
            self._update_method = self._update_euler_vectorized

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
        n_particles = len(positions)

        # Initialize vectorized arrays (primary storage)
        self._positions = np.array([pos.copy() for pos in positions])
        self._previous_positions = np.array([pos.copy() for pos in positions])
        self._masses = np.array(
            [
                masses[i] if masses and i < len(masses) else default_mass
                for i in range(n_particles)
            ]
        )
        self._fixed_mask = np.zeros(n_particles, dtype=bool)

        # Initialize particles as views into the arrays
        self.particles = [Particle(self, i) for i in range(n_particles)]

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

    def _update_verlet_vectorized(self):
        """Update physics using Verlet integration with vectorized operations."""
        time_step = self.config.time_step
        gravity = np.array(self.config.gravity)
        damping = self.config.damping

        # Calculate acceleration (gravity for now)
        acceleration = gravity

        # Verlet integration: x_new = 2x_current - x_previous + acceleration * dt^2
        # Apply only to non-fixed particles
        new_positions = self._positions.copy()
        non_fixed = ~self._fixed_mask

        new_positions[non_fixed] = (
            2 * self._positions[non_fixed]
            - self._previous_positions[non_fixed]
            + acceleration * time_step * time_step
        )

        # Update previous positions and current positions
        self._previous_positions[non_fixed] = self._positions[non_fixed]
        self._positions[non_fixed] = new_positions[non_fixed]

        # Apply damping
        self._positions[non_fixed] *= damping

        # No need to sync - particles are views into arrays

        # Apply constraints (springs)
        self._apply_spring_constraints_vectorized()

    def _update_euler_vectorized(self):
        """Update physics using simple Euler integration with vectorized operations."""
        time_step = self.config.time_step
        gravity = np.array(self.config.gravity)
        damping = self.config.damping

        # Apply Euler integration
        # Simple Euler: x_new = x_current + velocity * dt
        # For simplicity, we'll use gravity as velocity
        non_fixed = ~self._fixed_mask

        # Update previous positions
        self._previous_positions[non_fixed] = self._positions[non_fixed]

        # Update positions
        self._positions[non_fixed] += gravity * time_step

        # Apply damping
        self._positions[non_fixed] *= damping

        # No need to sync - particles are views into arrays

        # Apply constraints (springs)
        self._apply_spring_constraints_vectorized()

    def _apply_spring_constraints_vectorized(self):
        """Apply spring constraints between connected particles using vectorized operations."""
        if not self.constraints:
            return

        spring_constant = self.config.spring_constant
        rest_length = self.config.rest_length
        time_step = self.config.time_step

        # Convert constraints to numpy array for vectorized operations
        constraint_indices = np.array(self.constraints)

        # Get positions of constrained particles
        pos_i = self._positions[constraint_indices[:, 0]]
        pos_j = self._positions[constraint_indices[:, 1]]

        # Calculate current distances and directions
        delta = pos_j - pos_i
        distances = np.linalg.norm(delta, axis=1)

        # Avoid division by zero and skip zero-distance constraints
        valid_mask = distances > 0
        if not np.any(valid_mask):
            return

        # Calculate spring forces for valid constraints
        displacements = distances[valid_mask] - rest_length
        force_magnitudes = -spring_constant * displacements
        directions = delta[valid_mask] / distances[valid_mask, np.newaxis]

        # Calculate force vectors
        force_vectors = force_magnitudes[:, np.newaxis] * directions

        # Get masses of constrained particles
        mass_i = self._masses[constraint_indices[valid_mask, 0]]
        mass_j = self._masses[constraint_indices[valid_mask, 1]]

        # Calculate accelerations
        acceleration_i = force_vectors / mass_i[:, np.newaxis]
        acceleration_j = -force_vectors / mass_j[:, np.newaxis]

        # Apply forces to particles (only if not fixed)
        valid_constraints = constraint_indices[valid_mask]

        for idx, (i, j) in enumerate(valid_constraints):
            if not self._fixed_mask[i]:
                self._positions[i] += acceleration_i[idx] * time_step * time_step
            if not self._fixed_mask[j]:
                self._positions[j] += acceleration_j[idx] * time_step * time_step

        # No need to sync - particles are views into arrays

    def _sync_particle_objects(self):
        """No longer needed - particles are views into arrays."""
        pass

    def get_positions(self) -> List[np.ndarray]:
        """
        Get current positions of all particles.

        Returns:
            List of particle positions
        """
        return [self._positions[i].copy() for i in range(len(self.particles))]

    def set_particle_fixed(self, index: int, is_fixed: bool):
        """
        Set whether a particle is fixed in space.

        Args:
            index: Index of the particle
            is_fixed: Whether the particle should be fixed
        """
        if 0 <= index < len(self.particles):
            self._fixed_mask[index] = is_fixed
            # Particle object will reflect this change automatically

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
        return self._masses[index]

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
        if not self._is_valid_index(index):
            raise IndexError(f"Particle index {index} out of range")
        if not self._is_valid_mass(mass):
            raise ValueError("Mass must be positive")
        self._masses[index] = mass

    def _is_valid_index(self, index: int) -> TypeGuard[int]:
        """Check if the index is valid for particle access."""
        return 0 <= index < len(self.particles)

    def _is_valid_mass(self, mass: float) -> TypeGuard[float]:
        """Check if the mass is valid (positive)."""
        return mass > 0

    def set_particle_position(self, index: int, position: np.ndarray):
        """
        Set the position of a particle.

        Args:
            index: Index of the particle
            position: New position for the particle

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < len(self.particles):
            raise IndexError(f"Particle index {index} out of range")
        self._positions[index] = position.copy()
        self._previous_positions[index] = position.copy()
