import os
import sys
import unittest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from softbody import SoftBodySimulation
from visualization import Visualization


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.sim = SoftBodySimulation()
        self.viz = Visualization(self.sim)

    def test_initialization(self):
        """Test if the visualization initializes correctly."""
        self.assertIsNotNone(self.viz.fig)
        self.assertIsNotNone(self.viz.ax)

    def test_render(self):
        """Test if the render method runs without errors."""
        try:
            self.viz.render()
        except Exception as e:
            self.fail(f"Render method raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
