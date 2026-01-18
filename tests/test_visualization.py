"""
Test cases for the visualization module.
"""

import os
import sys
import unittest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock, patch

import numpy as np

from config import VisualizationConfig
from softbody import SoftBodySimulation
from visualization import Visualization


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.sim = SoftBodySimulation()
        self.viz = Visualization(self.sim)

    def test_initialization(self):
        """Test if visualization initializes correctly."""
        self.assertIsNotNone(self.viz.fig)
        self.assertIsNotNone(self.viz.ax)
        self.assertEqual(len(self.viz.trails), self.sim.get_particle_count())
        self.assertFalse(self.viz.paused)
        self.assertIsNone(self.viz.selected_particle)
        self.assertFalse(self.viz.dragging)

    def test_render(self):
        """Test if visualization renders correctly."""
        # Mock the render method to avoid GUI issues in testing
        with patch.object(self.viz, "_initialize_plot_elements"):
            with patch.object(self.viz, "_update_plot_elements"):
                with patch.object(self.viz, "_adjust_axis_limits"):
                    self.viz.render()

        # Should not raise exceptions

    def test_toggle_pause(self):
        """Test pause toggle functionality."""
        # Create a mock event
        mock_event = MagicMock()

        # Initially not paused
        self.assertFalse(self.viz.paused)

        # Toggle to paused
        self.viz.toggle_pause(mock_event)
        self.assertTrue(self.viz.paused)
        self.assertEqual(self.viz.pause_button.label.get_text(), "Resume")

        # Toggle back to unpaused
        self.viz.toggle_pause(mock_event)
        self.assertFalse(self.viz.paused)
        self.assertEqual(self.viz.pause_button.label.get_text(), "Pause")

    def test_reset_simulation(self):
        """Test simulation reset functionality."""
        # Create a mock event
        mock_event = MagicMock()

        # Store initial positions
        initial_positions = self.sim.get_particles()

        # Update simulation to change positions
        self.sim.update()

        # Reset should restore initial state
        self.viz.reset_simulation(mock_event)

        # Check that trails are cleared
        for trail in self.viz.trails:
            self.assertEqual(len(trail), 0)

        # Check that simulation is unpaused
        self.assertFalse(self.viz.paused)

    def test_custom_configuration(self):
        """Test visualization with custom configuration."""
        config = VisualizationConfig()
        config.trail_length = 20
        config.particle_size = 100
        config.update_interval = 100

        viz = Visualization(self.sim, config)

        self.assertEqual(len(viz.trails), self.sim.get_particle_count())
        for trail in viz.trails:
            self.assertEqual(trail.maxlen, 20)

    def test_particle_selection(self):
        """Test particle selection logic."""
        # Create a mock event with particle coordinates
        mock_event = MagicMock()
        mock_event.inaxes = self.viz.ax

        # Get first particle position
        particles = self.sim.get_particles()
        mock_event.xdata = particles[0][0]
        mock_event.ydata = particles[0][1]

        # Test particle selection
        self.viz.on_press(mock_event)
        self.assertEqual(self.viz.selected_particle, 0)
        self.assertTrue(self.viz.dragging)

        # Test particle release
        self.viz.on_release(mock_event)
        self.assertIsNone(self.viz.selected_particle)
        self.assertFalse(self.viz.dragging)

    def test_particle_dragging(self):
        """Test particle dragging functionality."""
        # Create mock events
        press_event = MagicMock()
        press_event.inaxes = self.viz.ax
        motion_event = MagicMock()
        motion_event.inaxes = self.viz.ax

        # Get first particle position
        particles = self.sim.get_particles()
        press_event.xdata = particles[0][0]
        press_event.ydata = particles[0][1]
        motion_event.xdata = particles[0][0] + 1.0
        motion_event.ydata = particles[0][1] + 1.0

        # Select particle
        self.viz.on_press(press_event)

        # Drag particle
        self.viz.on_motion(motion_event)

        # Check that particle position was updated
        updated_particles = self.sim.get_particles()
        self.assertNotEqual(updated_particles[0][0], particles[0][0])
        self.assertNotEqual(updated_particles[0][1], particles[0][1])


if __name__ == "__main__":
    unittest.main()
