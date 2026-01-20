"""
Visualization module for soft body simulation using Matplotlib.

This module provides interactive visualization of the soft body simulation with
particle trails, interactive controls, and configurable visualization parameters.
"""

import matplotlib

matplotlib.use("TkAgg")  # Set the backend explicitly

from collections import deque
from contextlib import contextmanager, suppress
from itertools import chain, islice, tee
from operator import attrgetter, itemgetter, methodcaller
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from config import VisualizationConfig, default_config


class VisualizationContext:
    """Context manager for managing visualization state."""

    def __init__(self, visualization):
        self.visualization = visualization
        self.original_state = {}

    def __enter__(self):
        """Enter the context, saving current state."""
        self.original_state = {
            "paused": self.visualization.paused,
            "selected_particle": self.visualization.selected_particle,
            "dragging": self.visualization.dragging,
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, restoring original state."""
        self.visualization.paused = self.original_state["paused"]
        self.visualization.selected_particle = self.original_state["selected_particle"]
        self.visualization.dragging = self.original_state["dragging"]
        return False


# This context manager is now a method of the Visualization class


@contextmanager
def suppress_matplotlib_warnings():
    """Context manager to suppress matplotlib warnings during operations."""
    with suppress(Exception):
        yield


class Visualization:
    """Interactive visualization for soft body simulation."""

    @contextmanager
    def visualization_context(self):
        """Context manager for visualization lifecycle management."""
        try:
            yield self
        finally:
            self.close()

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

        # Extract positions using functional patterns
        get_x = itemgetter(0)
        get_y = itemgetter(1)
        x = list(map(get_x, particles))
        y = list(map(get_y, particles))

        # Update trails using functional iteration
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
        # Use functional mapping for coordinate extraction
        x = list(map(itemgetter(0), particles))
        y = list(map(itemgetter(1), particles))

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

        # Plot constraints (springs) using functional filtering
        constraints = self.simulation.get_constraints()
        valid_constraints = [
            (i, j) for i, j in constraints if i < len(particles) and j < len(particles)
        ]

        for constraint_idx, (i, j) in enumerate(valid_constraints):
            (line,) = self.ax.plot(
                [x[i], x[j]],
                [y[i], y[j]],
                color="red",
                alpha=0.6,
                linewidth=1.5,
                label="Constraints" if constraint_idx == 0 else "",
            )
            self.constraint_lines.append(line)

    def _update_plot_elements(
        self, particles: List[np.ndarray], x: List[float], y: List[float]
    ):
        """Update existing plot elements using functional patterns."""
        # Update particle positions
        if self.particle_scatter is not None:
            self.particle_scatter.set_offsets(list(zip(x, y)))

        # Update trails using functional filtering and mapping
        trails_with_data = filter(lambda trail: len(trail) > 1, self.trails)
        for i, trail in enumerate(trails_with_data):
            if i < len(self.trail_lines):
                trail_x, trail_y = zip(*trail)
                self.trail_lines[i].set_data(trail_x, trail_y)

        # Update constraints using functional approach
        constraints = self.simulation.get_constraints()
        valid_constraints = [
            (idx, i, j)
            for idx, (i, j) in enumerate(constraints)
            if i < len(particles)
            and j < len(particles)
            and idx < len(self.constraint_lines)
        ]

        for constraint_idx, i, j in valid_constraints:
            self.constraint_lines[constraint_idx].set_data([x[i], x[j]], [y[i], y[j]])

    def _adjust_axis_limits(self, x: List[float], y: List[float]):
        """Adjust axis limits to fit particles with margin using functional approach."""
        if not x or not y:
            return

        # Use functional operations for min/max calculations
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        # Calculate margins using functional approach
        x_range = x_max - x_min
        y_range = y_max - y_min

        margin_calculator = lambda range_val: max(
            range_val * self.config.axis_margin, self.config.min_axis_margin
        )

        x_margin = margin_calculator(x_range)
        y_margin = margin_calculator(y_range)

        # Apply margins
        self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def toggle_pause(self, event):
        """Toggle pause state of the simulation."""
        self.paused = not self.paused
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw()

    @contextmanager
    def paused_simulation(self):
        """Context manager to temporarily pause simulation."""
        original_paused = self.paused
        self.paused = True
        try:
            yield
        finally:
            self.paused = original_paused

    def reset_simulation(self, event):
        """Reset the simulation to initial state."""
        # Use context manager to ensure clean reset
        with self.paused_simulation():
            # Reinitialize the softbody
            self.simulation._initialize_softbody()

            # Clear trails
            for trail in self.trails:
                trail.clear()

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
        """Handle mouse press events for particle selection using functional patterns."""
        if event.inaxes != self.ax:
            return

        # Use context manager for safe particle selection
        with suppress(Exception):
            # Check if a particle was clicked using functional approach
            particles = self.simulation.get_particles()
            positions = list(map(itemgetter(0, 1), particles))

            # Calculate distances using functional mapping
            event_pos = (event.xdata, event.ydata)
            distance_squared = (
                lambda pos: (pos[0] - event_pos[0]) ** 2 + (pos[1] - event_pos[1]) ** 2
            )

            # Find first particle within selection radius
            for i, pos in enumerate(positions):
                if distance_squared(pos) < 0.01:  # 0.1^2
                    self.selected_particle = i
                    self.dragging = True
                    break

    def on_motion(self, event):
        """Handle mouse motion events for particle dragging using functional patterns."""
        # Use functional approach for condition checking
        should_drag = all(
            [self.dragging, self.selected_particle is not None, event.inaxes == self.ax]
        )

        if not should_drag:
            return

        # Use context manager for safe particle dragging
        with suppress_matplotlib_warnings():
            # Create position array using functional approach
            position_creator = lambda x, y: np.array([x, y, 0.0])
            self.simulation.set_particle_position(
                self.selected_particle, position_creator(event.xdata, event.ydata)
            )

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
        with suppress(Exception):
            if hasattr(self, "animation_timer"):
                self.animation_timer.stop()
            plt.close(self.fig)
