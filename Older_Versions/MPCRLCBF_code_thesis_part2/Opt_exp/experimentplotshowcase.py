import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

# --- config ---
positions = [(-2.0, -1.5), (-3.0, -3.3), (-2.0, 0.0)]
radii     = [0.7, 0.7, 1.0]
modes     = ["step_bounce", "step_bounce", "static"]
mode_params = [
    {"bounds": (-4.0,  0.0), "speed": 2.3, "dir":  1},
    {"bounds": (-4.0,  1.0), "speed": 2.0, "dir": -1},
    {"bounds": (-2.0, -2.0), "speed": 0.0},
]

BOX_MIN, BOX_MAX = -5, 5
colors = plt.get_cmap("tab10").colors

# --- precompute for speed scaling ---
speeds = [mp.get("speed", 0.0) if m == "step_bounce" else 0.0
          for m, mp in zip(modes, mode_params)]
max_speed = max(1e-9, max(speeds))  # avoid /0

# fraction of band width used by the *fastest* arrow (keeps arrows inside band)
ARROW_FRAC_MAX = 0.35  # 35% of (xmax-xmin) for the fastest obstacle

fig, ax = plt.subplots(figsize=(7, 6))

# constraint box
ax.add_patch(patches.Rectangle((BOX_MIN, BOX_MIN),
                               BOX_MAX-BOX_MIN, BOX_MAX-BOX_MIN,
                               lw=1.5, ec='k', fc='none', ls='--'))

legend_handles = [patches.Patch(ec='k', fc='none', ls='--', label=r"$x_0,x_1$ constraint")]

for i, ((cx, cy), r, mode, mp) in enumerate(zip(positions, radii, modes, mode_params), start=1):
    color = colors[(i-1) % len(colors)]

    if mode == "step_bounce":
        xmin, xmax = mp["bounds"]
        speed = mp["speed"]
        direction = np.sign(mp.get("dir", 1))

        # swept band
        ax.add_patch(patches.Rectangle((xmin, cy-r), xmax-xmin, 2*r,
                                       lw=1.5, ec=color, fc=color, alpha=0.12))
        ax.vlines([xmin, xmax], cy-r, cy+r, colors=color, linestyles=":", lw=1)

        # initial disk
        ax.add_patch(plt.Circle((cx, cy), r, fc=color, ec='none', hatch='//', alpha=0.25))
        ax.add_patch(plt.Circle((cx, cy), r, fc='none', ec=color, lw=2))

        # --- speed-scaled arrow ---
        band_w = (xmax - xmin)
        # arrow length = (fraction of band) * (speed / max_speed)
        L = ARROW_FRAC_MAX * band_w * (speed / max_speed)
        # place arrow away from edges so the head stays within the band
        margin = 0.1 * band_w
        x0 = np.clip(cx, xmin + margin, xmax - margin)
        ax.arrow(x0, cy, direction * L, 0.0,
                 width=0.02,
                 head_width=0.25*r, head_length=0.25*r,
                 color=color, length_includes_head=True)

        # legend (per obstacle)
        legend_handles += [
            patches.Patch(facecolor=color, edgecolor=color, alpha=0.12,
                          label=f"Obstacle {i} Bound  [{xmin}, {xmax}]"),
            Line2D([0], [0], marker='o', color=color, lw=0,
                   markerfacecolor='white', markeredgecolor=color, markersize=8,
                   label=f"Obstacle {i}"),
        ]

    else:  # static
        ax.add_patch(plt.Circle((cx, cy), r, fc=color, ec='none', hatch='//', alpha=0.25))
        ax.add_patch(plt.Circle((cx, cy), r, fc='none', ec=color, lw=2))
        legend_handles += [
            Line2D([0], [0], marker='o', color=color, lw=0,
                   markerfacecolor='white', markeredgecolor=color, markersize=8,
                   label=f"Obstacle {i} Static"),
        ]

# Start / End
ax.scatter(BOX_MIN, BOX_MIN, s=120, c='red', marker='*')
ax.scatter(0, 0, s=120, c='green', marker='X')
ax.annotate("Start", xy=(BOX_MIN, BOX_MIN), xytext=(BOX_MIN+1.3, BOX_MIN+0.7),
            arrowprops=dict(facecolor='red', arrowstyle='->'),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5),
            fontsize=12, fontweight='bold', color='red')
ax.annotate("End", xy=(0, 0), xytext=(1.0, 0.5),
            arrowprops=dict(facecolor='green', arrowstyle='->'),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", lw=1.5),
            fontsize=12, fontweight='bold', color='green')

# cosmetics
ax.set_xlim([BOX_MIN, BOX_MAX]); ax.set_ylim([BOX_MIN, BOX_MAX])
ax.set_xlabel(r"$X$", fontsize=18); ax.set_ylabel(r"$Y$", fontsize=18)
# ax.set_title("Experiment Setup: Moving Obstacles (arrow length speed)", fontsize=13)
ax.axis("equal"); ax.grid(True, alpha=0.5)

# speed scale proxy (shows how arrow length maps to speed)
scale_band_w = 1.0  # normalized
L_fast = ARROW_FRAC_MAX * scale_band_w
legend_handles.append(Line2D([0], [0], color='k', lw=0,
                             marker=r'$\rightarrow$',
                             markersize=14,
                             label=f"Initial Direction"))

ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True)

plt.tight_layout(); plt.show()
