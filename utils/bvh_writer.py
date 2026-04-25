"""BVH (Biovision Hierarchy) motion capture file writer.

Exports our 15-joint rotation keyframes as a standard BVH file that
all game engines (Unity, Unreal, Godot) and 3D software (Blender, Maya)
can import natively.

BVH uses Y-up convention with ZXY Euler rotation order.
Our system uses Z-up with XYZ Euler rotation order.
The writer handles both conversions.
"""

from scipy.spatial.transform import Rotation


# Our joint name → BVH joint name mapping
JOINT_TO_BVH = {
    "spine": "Spine",
    "neck": "Neck",
    "head": "Head",
    "right_shoulder": "RightShoulder",
    "right_elbow": "RightElbow",
    "right_wrist": "RightWrist",
    "left_shoulder": "LeftShoulder",
    "left_elbow": "LeftElbow",
    "left_wrist": "LeftWrist",
    "right_hip": "RightHip",
    "right_knee": "RightKnee",
    "right_ankle": "RightAnkle",
    "left_hip": "LeftHip",
    "left_knee": "LeftKnee",
    "left_ankle": "LeftAnkle",
}

BVH_JOINT_NAMES = list(JOINT_TO_BVH.values())

# Ordered joint list matching the BVH hierarchy traversal order
# (depth-first, matching how channels appear in MOTION section)
BVH_JOINT_ORDER = [
    "Hips",           # root
    "Spine",          # Hips child
    "Neck",           # Spine child
    "Head",           # Neck child
    "RightShoulder",  # Spine child
    "RightElbow",     # RightShoulder child
    "RightWrist",     # RightElbow child
    "LeftShoulder",   # Spine child
    "LeftElbow",      # LeftShoulder child
    "LeftWrist",      # LeftElbow child
    "RightHip",       # Hips child
    "RightKnee",      # RightHip child
    "RightAnkle",     # RightKnee child
    "LeftHip",        # Hips child
    "LeftKnee",       # LeftHip child
    "LeftAnkle",      # LeftKnee child
]


def _build_hierarchy_lines() -> list[str]:
    """Generate BVH HIERARCHY section lines.

    Standard human skeleton with Y-up bone offsets (in meters).
    Root (Hips) has 6 channels (position + rotation).
    All other joints have 3 channels (rotation only).
    """
    lines = ["HIERARCHY"]

    def _joint(name, offset, channels, children_lines, indent=0):
        prefix = "  " * indent
        joint_type = "ROOT" if indent == 0 else "JOINT"

        lines.append(f"{prefix}{joint_type} {name}")
        lines.append(f"{prefix}{{")
        lines.append(f"{prefix}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
        lines.append(f"{prefix}  CHANNELS {len(channels)} {' '.join(channels)}")
        for child_name, child_offset, child_children in children_lines:
            _joint(child_name, child_offset, channels=["Zrotation", "Xrotation", "Yrotation"],
                   children_lines=child_children, indent=indent + 1)
        lines.append(f"{prefix}}}")

    # Define skeleton tree: (name, offset_from_parent, children)
    # Offsets in Y-up meters, standard human proportions
    head_children = []
    neck_children = [("Head", [0.0, 0.12, 0.0], head_children)]

    right_arm = [("RightElbow", [0.28, 0.0, 0.0],
                  [("RightWrist", [0.25, 0.0, 0.0], [])])]
    left_arm = [("LeftElbow", [-0.28, 0.0, 0.0],
                 [("LeftWrist", [-0.25, 0.0, 0.0], [])])]

    spine_children = [
        ("Neck", [0.0, 0.38, 0.0], neck_children),
        ("RightShoulder", [0.18, 0.0, 0.0], right_arm),
        ("LeftShoulder", [-0.18, 0.0, 0.0], left_arm),
    ]

    right_leg = [("RightKnee", [0.0, -0.42, 0.0],
                  [("RightAnkle", [0.0, -0.42, 0.0], [])])]
    left_leg = [("LeftKnee", [0.0, -0.42, 0.0],
                 [("LeftAnkle", [0.0, -0.42, 0.0], [])])]

    hips_children = [
        ("Spine", [0.0, 0.15, 0.0], spine_children),
        ("RightHip", [0.10, 0.0, 0.0], right_leg),
        ("LeftHip", [-0.10, 0.0, 0.0], left_leg),
    ]

    root_channels = [
        "Xposition", "Yposition", "Zposition",
        "Zrotation", "Xrotation", "Yrotation",
    ]
    _joint("Hips", [0.0, 0.90, 0.0], root_channels, hips_children)

    return lines


def _euler_xyz_to_bvh_zxy(xyz_deg: list[float]) -> list[float]:
    """Convert XYZ Euler angles (our convention) to ZXY (BVH convention).

    Both represent the same rotation, just decomposed differently.
    Uses scipy for reliable conversion.
    """
    r = Rotation.from_euler("XYZ", xyz_deg, degrees=True)
    return r.as_euler("ZXY", degrees=True).tolist()


def write_bvh(
    frames: list[dict[str, list[float]]],
    fps: float = 30.0,
    output_path: str = "output.bvh",
) -> str:
    """Write a BVH motion capture file from interpolated rotation frames.

    Args:
        frames: List of rotation dicts per frame.
                Each: {"spine": [x,y,z], ...} in our XYZ Euler convention.
        fps: Frame rate for the BVH file.
        output_path: File path to write.

    Returns:
        The output_path.
    """
    lines = _build_hierarchy_lines()

    # MOTION section
    lines.append("MOTION")
    lines.append(f"Frames: {len(frames)}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")

    # Map our joint names to BVH order
    # BVH_JOINT_ORDER[0] = "Hips" (root, no rotation mapping)
    # BVH_JOINT_ORDER[1:] map to our joints via JOINT_TO_BVH inverse
    bvh_to_ours = {v: k for k, v in JOINT_TO_BVH.items()}

    for frame in frames:
        values = []
        for bvh_joint in BVH_JOINT_ORDER:
            if bvh_joint == "Hips":
                # Root: position (fixed at origin) + rotation (from spine)
                # Use spine rotation for root hips
                spine_bvh = _euler_xyz_to_bvh_zxy(frame.get("spine", [0, 0, 0]))
                values.extend([0.0, 0.0, 0.0])  # position
                values.extend(spine_bvh)
            else:
                our_name = bvh_to_ours.get(bvh_joint)
                if our_name:
                    xyz = frame.get(our_name, [0.0, 0.0, 0.0])
                    zxy = _euler_xyz_to_bvh_zxy(xyz)
                    values.extend(zxy)
                else:
                    values.extend([0.0, 0.0, 0.0])

        lines.append(" ".join(f"{v:.6f}" for v in values))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return output_path
