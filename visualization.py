"""
Visualization module for soft body simulation using Matplotlib.

This module provides interactive visualization of the soft body simulation with
particle trails, interactive controls, and configurable visualization parameters.
"""

import matplotlib

matplotlib.use("TkAgg")  # Set the backend explicitly

from collections import deque
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from config import VisualizationConfig, default_config


class Visualization:
    """Interactive visualization for soft body simulation."""

    def __init__(self, simulation, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualization.

        Args:
            simulation: SoftBodySimulation instance to visualize
            config: Visualization configuration. If None, uses default configuration.
        """
        self.simulation = simulation
        self.config = config if config is not None else default_config.visualization

        # Initialize matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Soft Body Simulation")

        # Initialize trails for particles
        self.trails = [
            deque(maxlen=self.config.trail_length)
            for _ in range(simulation.get_particle_count())
        ]

        # Set up plot labels and title
        self.ax.set_title("Soft Body Simulation", fontsize=16, pad=20)
        self.ax.set_xlabel("X Position", fontsize=12)
        self.ax.set_ylabel("Y Position", fontsize=12)

        # Interactive controls
        self.paused = False
        self.selected_particle = None
        self.dragging = False

        # Create control buttons
        self._create_control_buttons()

        # Connect event handlers
        self._connect_event_handlers()

        # Initialize plot elements
        self.particle_scatter = None
        self.trail_lines = []
        self.constraint_lines = []

        # Set up animation timer
        self.animation_timer = self.fig.canvas.new_timer(
            interval=self.config.update_interval
        )
        self.animation_timer.add_callback(self._animation_callback)
        self.animation_timer.start()

    def _create_control_buttons(self):
        """Create interactive control buttons."""
        # Pause/Resume button
        self.pause_ax = plt.axes((0.7, 0.05, 0.1, 0.04))
        self.pause_button = Button(self.pause_ax, "Pause")
        self.pause_button.on_clicked(self.toggle_pause)

        # Reset button
        self.reset_ax = plt.axes((0.81, 0.05, 0.1, 0.04))
        self.reset_button = Button(self.reset_ax, "Reset")
        self.reset_button.on_clicked(self.reset_simulation)

    def _connect_event_handlers(self):
        """Connect matplotlib event handlers."""
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _animation_callback(self):
        """Callback for animation timer."""
        if not self.paused:
            self.simulation.update()
        self.render()
        self.fig.canvas.draw()

    def render(self):
        """Render the current state of the simulation."""
        if not (particles := self.simulation.get_particles()):
            return

        # Extract positions using generator expressions
        x = list(p[0] for p in particles)
        y = list(p[1] for p in particles)

        # Update trails
        for i, (xi, yi) in enumerate(zip(x, y)):
            self.trails[i].append((xi, yi))

        # Clear only the necessary elements instead of entire plot
        if self.particle_scatter is None:
            # Initial render
            self._initialize_plot_elements(particles)
        else:
            # Update existing elements
            self._update_plot_elements(particles, x, y)

        # Auto-adjust axis limits
        self._adjust_axis_limits(x, y)

        # Add legend and grid
        if hasattr(self, "legend") and self.legend:
            self.legend.remove()
        self.legend = self.ax.legend(fontsize=10)
        self.ax.grid(True, alpha=0.3)

    def _initialize_plot_elements(self, particles: List[np.ndarray]):
        """Initialize plot elements for the first render."""
        x = [p[0] for p in particles]
        y = [p[1] for p in particles]

        # Plot trails
        for trail in self.trails:
            if len(trail) > 1:
                trail_x, trail_y = zip(*trail)
                (line,) = self.ax.plot(
                    trail_x, trail_y, alpha=0.3, color="gray", linewidth=1
                )
                self.trail_lines.append(line)

        # Plot particles
        self.particle_scatter = self.ax.scatter(
            x,
            y,
            s=self.config.particle_size,
            color="blue",
            label="Particles",
            picker=True,
            edgecolors="black",
            linewidth=0.5,
        )

        # Plot constraints (springs)
        constraints = self.simulation.get_constraints()
        for i, j in constraints:
            if i < len(particles) and j < len(particles):
                (line,) = self.ax.plot(
                    [x[i], x[j]],
                    [y[i], y[j]],
                    color="red",
                    alpha=0.6,
                    linewidth=1.5,
                    label="Constraints" if not self.constraint_lines else "",
                )
                self.constraint_lines.append(line)

    def _update_plot_elements(
        self, particles: List[np.ndarray], x: List[float], y: List[float]
    ):
        """Update existing plot elements."""
        # Update particle positions
        if self.particle_scatter is not None:
            self.particle_scatter.set_offsets(list(zip(x, y)))

        # Update trails
        for i, trail_line in enumerate(self.trail_lines):
            if len(self.trails[i]) > 1:
                trail_x, trail_y = zip(*self.trails[i])
                trail_line.set_data(trail_x, trail_y)

        # Update constraints
        constraints = self.simulation.get_constraints()
        for idx, (i, j) in enumerate(constraints):
            if i < len(particles) and j < len(particles):
                if idx < len(self.constraint_lines):
                    self.constraint_lines[idx].set_data([x[i], x[j]], [y[i], y[j]])

    def _adjust_axis_limits(self, x: List[float], y: List[float]):
        """Adjust axis limits to fit particles with margin."""
        if not x or not y:
            return

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        # Calculate margins
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_margin = max(x_range * self.config.axis_margin, self.config.min_axis_margin)
        y_margin = max(y_range * self.config.axis_margin, self.config.min_axis_margin)

        # Apply margins
        self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def toggle_pause(self, event):
        """Toggle pause state of the simulation."""
        self.paused = not self.paused
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw()

    def reset_simulation(self, event):
        """Reset the simulation to initial state."""
        # Reinitialize the softbody
        self.simulation._initialize_softbody()

        # Clear trails
        for trail in self.trails:
            trail.clear()

        # Force redraw without updating trails
        self.paused = False
        self.pause_button.label.set_text("Pause")

        # Update plot elements without updating trails
        particles = self.simulation.get_particles()
        if particles:
            x = [p[0] for p in particles]
            y = [p[1] for p in particles]

            if self.particle_scatter is None:
                self._initialize_plot_elements(particles)
            else:
                self._update_plot_elements(particles, x, y)
                self._adjust_axis_limits(x, y)

        self.fig.canvas.draw()

    def on_press(self, event):
        """Handle mouse press events for particle selection."""
        if event.inaxes != self.ax:
            return

        # Check if a particle was clicked
        particles = self.simulation.get_particles()
        x = [p[0] for p in particles]
        y = [p[1] for p in particles]

        # Find clicked particle
        for i, (xi, yi) in enumerate(zip(x, y)):
            if np.sqrt((event.xdata - xi) ** 2 + (event.ydata - yi) ** 2) < 0.1:
                self.selected_particle = i
                self.dragging = True
                break

    def on_motion(self, event):
        """Handle mouse motion events for particle dragging."""
        if (
            not self.dragging
            or self.selected_particle is None
            or event.inaxes != self.ax
        ):
            return

        # Update the position of the selected particle
        try:
            self.simulation.set_particle_position(
                self.selected_particle, np.array([event.xdata, event.ydata, 0.0])
            )
        except (IndexError, AttributeError):
            pass

    def on_release(self, event):
        """Handle mouse release events."""
        self.dragging = False
        self.selected_particle = None

    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == " ":
            # Space key toggles pause
            self.toggle_pause(event)
        elif event.key == "r":
            # R key resets simulation
            self.reset_simulation(event)

    def close(self):
        """Clean up visualization resources."""
        if hasattr(self, "animation_timer"):
            self.animation_timer.stop()
        plt.close(self.fig)
