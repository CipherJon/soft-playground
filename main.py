"""
Main entry point for the soft body simulation.

This module provides the main interface for running the soft body simulation
with visualization and interactive controls.
"""

import sys
from typing import Optional

import matplotlib.pyplot as plt

import softbody
import visualization
from config import SimulationConfigManager, default_config


def run_simulation(config: Optional[SimulationConfigManager] = None):
    """
    Run the soft body simulation with visualization.

    Args:
        config: Simulation configuration. If None, uses default configuration.
    """
    # Initialize the softbody simulation
    sim = softbody.SoftBodySimulation(config)

    # Initialize visualization
    viz = visualization.Visualization(sim, config.visualization if config else None)

    print("Soft Body Simulation Started")
    print("Controls:")
    print("  - Click and drag particles to interact")
    print("  - Pause/Resume button: Toggle simulation pause")
    print("  - Reset button: Reset simulation to initial state")
    print("  - Space key: Toggle pause")
    print("  - R key: Reset simulation")
    print("  - Close the window to exit")

    # Run the simulation loop
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nSimulation stopped due to error: {e}")
    finally:
        viz.close()
        print("Simulation ended.")


def create_custom_config() -> SimulationConfigManager:
    """
    Create a custom configuration for demonstration purposes.

    Returns:
        Custom simulation configuration
    """
    config = SimulationConfigManager()

    # Customize physics parameters
    config.physics.gravity = (0.0, -5.0, 0.0)  # Reduced gravity
    config.physics.damping = 0.98  # Slightly less damping
    config.physics.spring_constant = 0.2  # Stronger springs
    config.physics.time_step = 0.015  # Slightly larger time step

    # Customize simulation parameters
    config.simulation.num_particles = 15  # More particles
    config.simulation.initial_spacing = 0.08  # Closer spacing
    config.simulation.particle_mass = 1.2  # Slightly heavier particles

    # Customize visualization parameters
    config.visualization.trail_length = 30  # Shorter trails
    config.visualization.particle_size = 60  # Larger particles
    config.visualization.update_interval = 30  # Faster updates

    return config


def main():
    """Main entry point for the soft body simulation."""
    # Parse command line arguments
    use_custom_config = False
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        use_custom_config = True

    # Create configuration
    config = create_custom_config() if use_custom_config else None

    # Run simulation
    run_simulation(config)


if __name__ == "__main__":
    main()
