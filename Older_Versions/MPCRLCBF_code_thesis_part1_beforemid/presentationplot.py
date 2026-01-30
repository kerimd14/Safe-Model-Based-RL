import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots()

# Draw the circle with legend label h(x)
circle_inside = plt.Circle((-2, -2.25), 1.5, edgecolor="none", facecolor="blue",    # facecolor must be non-None for hatch to show
    hatch="//", fill=True, linewidth=2)
ax.add_patch(circle_inside)

circle_outside = plt.Circle((-2, -2.25), 1.5, color="blue", fill=False, linewidth=2, label=r"Obstacle")
ax.add_patch(circle_outside)

# Draw a dotted square (box) from (-5, -5) to (5, 5) with legend label x₀, x₁ constraint
square = patches.Rectangle(
    (-5, -5), 10, 10,
    linewidth=1.5, edgecolor='k', facecolor='none',
    linestyle='--', label=r"$x_0, x_1\ \mathrm{constraint}$"
)
ax.add_patch(square)

# Plot stylish markers for start and end
ax.scatter(-5, -5, s=100, c='red', marker='*')  # start marker
ax.scatter(0, 0, s=100, c='green', marker='X')  # end marker

# Annotate 'start' and 'end' with boxed labels and arrows
ax.annotate(
    "Start", xy=(-5, -5), xytext=(-3.5, -4.5),
    arrowprops=dict(facecolor='red', arrowstyle='->'),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5),
    fontsize=12, fontweight='bold', color='red'
)
ax.annotate(
    "End", xy=(0, 0), xytext=(1, 0.5),
    arrowprops=dict(facecolor='green', arrowstyle='->'),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=1.5),
    fontsize=12, fontweight='bold', color='green'
)

# Set limits to match the square
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

# Labels and title
ax.set_xlabel("$X$", fontsize =20)
ax.set_ylabel("$Y$", fontsize =20)
ax.set_title("Numerical Example Setup")

# Ensure equal aspect ratio and grid
ax.axis("equal")
ax.grid()

# Legend for square and circle
ax.legend(loc='upper right', fontsize = 11)

plt.show()
