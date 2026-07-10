import matplotlib
matplotlib.use("Agg")  # Use this when running from terminal/SSH/no GUI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# Tomato Arm Monte Carlo Workspace Sampling
# Uses your DH-based FK model
# Units: centimeters
# Saves plots as PNG files
# ============================================================


# -----------------------------
# 1. Robot geometry in centimeters
# -----------------------------

# base_link origin to shoulder pitch joint height
h = 10.597

# shoulder -> elbow
L1 = 17.78

# elbow -> wrist
L2 = 15.24

# wrist -> end_effector_link attachment
L3 = 3.45

# If later you want the actual tomato contact point/tool tip,
# replace L3 with:
# L3 = 3.45 + 3.193


# -----------------------------
# 2. Joint limits from URDF
# -----------------------------

# joint_1 is continuous, but for tomato picking you probably only need the front half.
# Change to -np.pi, np.pi if you want full yaw.
q1_min = np.deg2rad(-90)
q1_max = np.deg2rad(90)

q2_min = -1.618
q2_max = 1.595

q3_min = -1.825
q3_max = 1.792

q4_min = -1.855
q4_max = 1.786


# -----------------------------
# 3. Standard DH transform
# -----------------------------

def dh_transform(theta, d, a, alpha):
    """
    Standard DH transform:
    A_i = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)

    Units:
        d and a are in centimeters.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [c, -s * ca,  s * sa, a * c],
        [s,  c * ca, -c * sa, a * s],
        [0,      sa,      ca,     d],
        [0,       0,       0,     1],
    ])


# -----------------------------
# 4. Forward kinematics from DH
# -----------------------------

def fk_dh(q1, q2, q3, q4):
    """
    DH table:

    i | theta        | d       | a    | alpha
    1 | q1           | h       | 0    | -pi/2
    2 | q2 - pi/2    | 0       | L1   | 0
    3 | q3           | 0       | L2   | 0
    4 | q4           | 0       | L3   | 0

    Returns:
        position: [x, y, z] in centimeters
        T: full 4x4 transform from base_link to end_effector_link
    """

    A1 = dh_transform(q1, h, 0.0, -np.pi / 2)
    A2 = dh_transform(q2 - np.pi / 2, 0.0, L1, 0.0)
    A3 = dh_transform(q3, 0.0, L2, 0.0)
    A4 = dh_transform(q4, 0.0, L3, 0.0)

    T = A1 @ A2 @ A3 @ A4
    position = T[:3, 3]

    return position, T


# -----------------------------
# 5. Faster closed-form FK for Monte Carlo
# -----------------------------

def fk_position_vectorized(q1, q2, q3, q4):
    """
    Vectorized FK position.

    This matches the zero-pose convention where all links point straight up:
        q1 = q2 = q3 = q4 = 0
        x = 0
        y = 0
        z = h + L1 + L2 + L3

    Returns:
        points in centimeters
    """

    r = (
        L1 * np.sin(q2)
        + L2 * np.sin(q2 + q3)
        + L3 * np.sin(q2 + q3 + q4)
    )

    z = (
        h
        + L1 * np.cos(q2)
        + L2 * np.cos(q2 + q3)
        + L3 * np.cos(q2 + q3 + q4)
    )

    x = r * np.cos(q1)
    y = r * np.sin(q1)

    return np.column_stack((x, y, z))


# -----------------------------
# 6. FK sanity checks
# -----------------------------

print("=== FK sanity checks ===")

test_poses = {
    "zero_pose": (0, 0, 0, 0),
    "shoulder_30_deg": (0, np.deg2rad(30), 0, 0),
    "shoulder_90_deg": (0, np.deg2rad(90), 0, 0),
    "elbow_30_deg": (0, 0, np.deg2rad(30), 0),
    "wrist_30_deg": (0, 0, 0, np.deg2rad(30)),
    "yaw_90_shoulder_30": (np.deg2rad(90), np.deg2rad(30), 0, 0),
}

for name, qs in test_poses.items():
    p_dh, _ = fk_dh(*qs)

    p_fast = fk_position_vectorized(
        np.array([qs[0]]),
        np.array([qs[1]]),
        np.array([qs[2]]),
        np.array([qs[3]]),
    )[0]

    print(f"{name:22s}")
    print(f"  DH matrix FK:   x={p_dh[0]: .2f}, y={p_dh[1]: .2f}, z={p_dh[2]: .2f} cm")
    print(f"  Fast FK:        x={p_fast[0]: .2f}, y={p_fast[1]: .2f}, z={p_fast[2]: .2f} cm")


# -----------------------------
# 7. Monte Carlo sampling
# -----------------------------

N = 100000

rng = np.random.default_rng(seed=42)

q1_samples = rng.uniform(q1_min, q1_max, N)
q2_samples = rng.uniform(q2_min, q2_max, N)
q3_samples = rng.uniform(q3_min, q3_max, N)
q4_samples = rng.uniform(q4_min, q4_max, N)

points = fk_position_vectorized(
    q1_samples,
    q2_samples,
    q3_samples,
    q4_samples,
)

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


# -----------------------------
# 8. Practical tomato-picking workspace
# -----------------------------

# This uses base_link frame, in centimeters.
# Target tomato center height is around 15-20 cm above table/base.
# If your base_link is not exactly table height, adjust z_min/z_max.

x_min, x_max = 24, 32
y_min, y_max = -10, 10
z_min, z_max = 15, 20

practical_mask = (
    (x >= x_min) & (x <= x_max)
    & (y >= y_min) & (y <= y_max)
    & (z >= z_min) & (z <= z_max)
)

practical_points = points[practical_mask]

print("\n=== Workspace results ===")
print(f"Total sampled points: {N}")
print(f"Points inside practical tomato workspace: {len(practical_points)}")
print(f"Percent inside practical workspace: {100 * len(practical_points) / N:.2f}%")

print("\n=== Reachable cloud bounds ===")
print(f"x range: {x.min(): .1f} to {x.max(): .1f} cm")
print(f"y range: {y.min(): .1f} to {y.max(): .1f} cm")
print(f"z range: {z.min(): .1f} to {z.max(): .1f} cm")


# -----------------------------
# 9. Draw 3D workspace box
# -----------------------------

def plot_box_3d(ax, xlim, ylim, zlim):
    xs = [xlim[0], xlim[1]]
    ys = [ylim[0], ylim[1]]
    zs = [zlim[0], zlim[1]]

    corners = np.array([
        [xs[0], ys[0], zs[0]],
        [xs[1], ys[0], zs[0]],
        [xs[1], ys[1], zs[0]],
        [xs[0], ys[1], zs[0]],
        [xs[0], ys[0], zs[1]],
        [xs[1], ys[0], zs[1]],
        [xs[1], ys[1], zs[1]],
        [xs[0], ys[1], zs[1]],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            linewidth=2,
        )


# -----------------------------
# 10. Save 3D workspace plot
# -----------------------------

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    x,
    y,
    z,
    s=1,
    alpha=0.06,
    label="Reachable workspace samples",
)

if len(practical_points) > 0:
    ax.scatter(
        practical_points[:, 0],
        practical_points[:, 1],
        practical_points[:, 2],
        s=5,
        alpha=0.8,
        label="Samples inside practical picking workspace",
    )

plot_box_3d(
    ax,
    (x_min, x_max),
    (y_min, y_max),
    (z_min, z_max),
)

ax.set_title("Monte Carlo Reachable Workspace")
ax.set_xlabel("x forward/back (cm)")
ax.set_ylabel("y left/right (cm)")
ax.set_zlabel("z up/down (cm)")
ax.legend()

# Make axes roughly equal scale
max_range = np.array([
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min(),
]).max() / 2.0

mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
plt.savefig("workspace_3d_cm.png", dpi=300)
plt.close()

print("Saved workspace_3d_cm.png")


# -----------------------------
# 11. Save top view: x-y
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    x,
    y,
    s=1,
    alpha=0.06,
    label="Reachable workspace samples",
)

if len(practical_points) > 0:
    plt.scatter(
        practical_points[:, 0],
        practical_points[:, 1],
        s=5,
        alpha=0.8,
        label="Practical workspace samples",
    )

rect_xy = Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    fill=False,
    linewidth=2,
)

plt.gca().add_patch(rect_xy)
plt.title("Top View of Workspace")
plt.xlabel("x forward/back (cm)")
plt.ylabel("y left/right (cm)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("workspace_top_xy_cm.png", dpi=300)
plt.close()

print("Saved workspace_top_xy_cm.png")


# -----------------------------
# 12. Save side view: x-z
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    x,
    z,
    s=1,
    alpha=0.06,
    label="Reachable workspace samples",
)

if len(practical_points) > 0:
    plt.scatter(
        practical_points[:, 0],
        practical_points[:, 2],
        s=5,
        alpha=0.8,
        label="Practical workspace samples",
    )

rect_xz = Rectangle(
    (x_min, z_min),
    x_max - x_min,
    z_max - z_min,
    fill=False,
    linewidth=2,
)

plt.gca().add_patch(rect_xz)
plt.title("Side View of Workspace")
plt.xlabel("x forward/back (cm)")
plt.ylabel("z up/down (cm)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("workspace_side_xz_cm.png", dpi=300)
plt.close()

print("Saved workspace_side_xz_cm.png")


# -----------------------------
# 13. Save front view: y-z
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y,
    z,
    s=1,
    alpha=0.06,
    label="Reachable workspace samples",
)

if len(practical_points) > 0:
    plt.scatter(
        practical_points[:, 1],
        practical_points[:, 2],
        s=5,
        alpha=0.8,
        label="Practical workspace samples",
    )

rect_yz = Rectangle(
    (y_min, z_min),
    y_max - y_min,
    z_max - z_min,
    fill=False,
    linewidth=2,
)

plt.gca().add_patch(rect_yz)
plt.title("Front View of Workspace")
plt.xlabel("y left/right (cm)")
plt.ylabel("z up/down (cm)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("workspace_front_yz_cm.png", dpi=300)
plt.close()

print("Saved workspace_front_yz_cm.png")


print("\nDone. Open the PNG files to view the plots.")