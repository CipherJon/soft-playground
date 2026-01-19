"""
Configuration module for soft body simulation parameters.

This module provides a centralized configuration system for all simulation parameters,
making it easy to adjust physics properties, visualization settings, and simulation behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class PhysicsConfig:
    """Configuration for physics engine parameters."""

    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)  # Gravity vector (x, y, z)
    damping: float = 0.99  # Damping factor (0-1)
    spring_constant: float = 0.1  # Spring stiffness
    rest_length: float = 1.0  # Rest length for springs
    time_step: float = 0.01  # Simulation time step
    integration_method: str = "verlet"  # Integration method: "euler" or "verlet"


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""

    trail_length: int = 50  # Number of points in particle trails
    particle_size: float = 50.0  # Size of particles in visualization
    axis_margin: float = 0.2  # Margin around particles in axis view
    min_axis_margin: float = 0.1  # Minimum margin regardless of particle spread
    update_interval: int = 50  # Milliseconds between visualization updates


@dataclass
class SimulationConfig:
    """Configuration for soft body simulation."""

    num_particles: int = 10  # Number of particles in the simulation
    initial_spacing: float = 0.1  # Spacing between particles in initialization
    particle_mass: float = 1.0  # Mass of each particle
    use_3d: bool = False  # Whether to use 3D visualization


@dataclass
class SimulationConfigManager:
    """Manages all simulation configuration parameters."""

    physics: PhysicsConfig = PhysicsConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    simulation: SimulationConfig = SimulationConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "physics": {
                "gravity": self.physics.gravity,
                "damping": self.physics.damping,
                "spring_constant": self.physics.spring_constant,
                "rest_length": self.physics.rest_length,
                "time_step": self.physics.time_step,
                "integration_method": self.physics.integration_method,
            },
            "visualization": {
                "trail_length": self.visualization.trail_length,
                "particle_size": self.visualization.particle_size,
                "axis_margin": self.visualization.axis_margin,
                "min_axis_margin": self.visualization.min_axis_margin,
                "update_interval": self.visualization.update_interval,
            },
            "simulation": {
                "num_particles": self.simulation.num_particles,
                "initial_spacing": self.simulation.initial_spacing,
                "particle_mass": self.simulation.particle_mass,
                "use_3d": self.simulation.use_3d,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfigManager":
        """Create configuration manager from dictionary."""
        physics_config = config_dict.get("physics", {})
        visualization_config = config_dict.get("visualization", {})
        simulation_config = config_dict.get("simulation", {})

        # Define default configurations
        default_physics_config = {
            "gravity": (0.0, -9.81, 0.0),
            "damping": 0.99,
            "spring_constant": 0.1,
            "rest_length": 1.0,
            "time_step": 0.01,
            "integration_method": "verlet",
        }

        default_visualization_config = {
            "trail_length": 50,
            "particle_size": 50.0,
            "axis_margin": 0.2,
            "min_axis_margin": 0.1,
            "update_interval": 50,
        }

        default_simulation_config = {
            "num_particles": 10,
            "initial_spacing": 0.1,
            "particle_mass": 1.0,
            "use_3d": False,
        }

        # Merge user-provided configurations with defaults using the | operator
        merged_physics_config = default_physics_config | physics_config
        merged_visualization_config = (
            default_visualization_config | visualization_config
        )
        merged_simulation_config = default_simulation_config | simulation_config

        return cls(
            physics=PhysicsConfig(
                gravity=merged_physics_config["gravity"],
                damping=merged_physics_config["damping"],
                spring_constant=merged_physics_config["spring_constant"],
                rest_length=merged_physics_config["rest_length"],
                time_step=merged_physics_config["time_step"],
                integration_method=merged_physics_config["integration_method"],
            ),
            visualization=VisualizationConfig(
                trail_length=merged_visualization_config["trail_length"],
                particle_size=merged_visualization_config["particle_size"],
                axis_margin=merged_visualization_config["axis_margin"],
                min_axis_margin=merged_visualization_config["min_axis_margin"],
                update_interval=merged_visualization_config["update_interval"],
            ),
            simulation=SimulationConfig(
                num_particles=merged_simulation_config["num_particles"],
                initial_spacing=merged_simulation_config["initial_spacing"],
                particle_mass=merged_simulation_config["particle_mass"],
                use_3d=merged_simulation_config["use_3d"],
            ),
        )


# Default configuration instance
default_config = SimulationConfigManager()
