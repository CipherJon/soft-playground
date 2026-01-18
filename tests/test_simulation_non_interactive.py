#!/usr/bin/env python3
"""
Non-interactive test script for the softbody simulation.
This script runs the simulation without visualization and prints the particle positions.
"""

import softbody


def test_simulation_non_interactive():
    # Initialize the softbody simulation
    sim = softbody.SoftBodySimulation()

    # Run the simulation for a fixed number of steps
    num_steps = 100
    for step in range(num_steps):
        sim.update()

        # Print particle positions
        particles = sim.get_particles()
        print(f"Step {step + 1}:")
        for i, particle in enumerate(particles):
            print(f"  Particle {i}: {particle}")


if __name__ == "__main__":
    test_simulation_non_interactive()
