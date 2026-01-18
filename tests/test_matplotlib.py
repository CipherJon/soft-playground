import matplotlib

matplotlib.use("GTK3Agg")  # Use GTK3Agg backend
print("Using backend:", matplotlib.get_backend())  # Debug statement

import matplotlib.pyplot as plt
import numpy as np

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Test Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# Show the plot
print("Displaying plot...")
plt.show()
print("Plot closed.")
