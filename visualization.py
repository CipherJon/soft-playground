import matplotlib

matplotlib.use("TkAgg")  # Set the backend explicitly

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


class Visualization:
    def __init__(self, simulation):
        self.simulation = simulation
        self.fig, self.ax = plt.subplots()
        self.trails = [deque(maxlen=50) for _ in range(len(simulation.get_particles()))]
        self.ax.set_title("Soft Body Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        # Add buttons for interactivity
        self.paused = False
        self.selected_particle = None
        self.dragging = False

        # Pause/Resume button
        self.pause_ax = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.pause_button = Button(self.pause_ax, "Pause")
        self.pause_button.on_clicked(self.toggle_pause)

        # Connect event handlers
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

    def render(self):
        particles = self.simulation.get_particles()
        x = [p[0] for p in particles]
        y = [p[1] for p in particles]

        # Update trails
        for i, (xi, yi) in enumerate(zip(x, y)):
            self.trails[i].append((xi, yi))

        # Clear and redraw
        self.ax.clear()

        # Plot trails
        for trail in self.trails:
            if len(trail) > 1:
                trail_x, trail_y = zip(*trail)
                self.ax.plot(trail_x, trail_y, alpha=0.5, color="gray")

        # Plot particles
        self.ax.scatter(x, y, color="blue", label="Particles", picker=True)

        # Auto-adjust axis limits
        if x and y:
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            x_margin = max((x_max - x_min) * 0.2, 0.1)  # Ensure minimum margin
            y_margin = max((y_max - y_min) * 0.2, 0.1)  # Ensure minimum margin
            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

        self.ax.set_title("Soft Body Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.legend()
        self.ax.grid(True)

        plt.draw()  # Ensure the plot is updated

    def toggle_pause(self, event):
        self.paused = not self.paused
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")
        plt.draw()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        # Check if a particle was clicked
        particles = self.simulation.get_particles()
        x = [p[0] for p in particles]
        y = [p[1] for p in particles]

        for i, (xi, yi) in enumerate(zip(x, y)):
            if np.sqrt((event.xdata - xi) ** 2 + (event.ydata - yi) ** 2) < 0.1:
                self.selected_particle = i
                self.dragging = True
                break

    def on_motion(self, event):
        if (
            not self.dragging
            or self.selected_particle is None
            or event.inaxes != self.ax
        ):
            return

        # Update the position of the selected particle
        particles = self.simulation.get_particles()
        particles[self.selected_particle][0] = event.xdata
        particles[self.selected_particle][1] = event.ydata

    def on_release(self, event):
        self.dragging = False
        self.selected_particle = None
        plt.pause(0.01)
