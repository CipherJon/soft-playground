# Soft-playground
A Verlet ragdoll simulation project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/soft-playground.git
   cd soft-playground
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install system-level dependencies (for visualization):
   - For Debian/Ubuntu:
     ```bash
     sudo apt-get install python3-tk
     ```

## Running the Tests

To run the tests, use the following command:
```bash
python -m pytest tests/
```

## Running the Simulation

To run the simulation with visualization, use the following command:
```bash
python main.py
```

### Controls
- The visualization window will open automatically.
- The view will auto-adjust to follow the particles.
- Trails are shown to visualize the path of the particles.
- You can pan and zoom the graph using the Matplotlib toolbar.

## Features

- Physics engine with gravity, damping, and spring constraints.
- Soft body simulation for ragdoll-like behavior.
- Visualization using Matplotlib.

## Dependencies

- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `Pillow>=12.1.0`
