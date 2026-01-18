import matplotlib.pyplot as plt

import softbody
import visualization


def main():
    # Initialize the softbody simulation
    sim = softbody.SoftBodySimulation()

    # Initialize visualization
    viz = visualization.Visualization(sim)

    # Run the simulation loop
    try:
        while True:
            if not viz.paused:
                sim.update()
            viz.render()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        plt.close()


if __name__ == "__main__":
    main()
