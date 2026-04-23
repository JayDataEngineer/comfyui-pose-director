"""OpenPose data structures and constants.

COCO-18 format, matching the reference rendering from controlnet_aux.
"""

# COCO-18 keypoint names and indices
KEYPOINT_NAMES = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar",
}

# Total number of body keypoints
NUM_BODY_KEYPOINTS = 18

# Skeleton bone connections (0-indexed) — matches reference OpenPose limbSeq
SKELETON_BONES = [
    (1, 2),   # Neck → RShoulder
    (1, 5),   # Neck → LShoulder
    (2, 3),   # RShoulder → RElbow
    (3, 4),   # RElbow → RWrist
    (5, 6),   # LShoulder → LElbow
    (6, 7),   # LElbow → LWrist
    (1, 8),   # Neck → RHip
    (8, 9),   # RHip → RKnee
    (9, 10),  # RKnee → RAnkle
    (1, 11),  # Neck → LHip
    (11, 12), # LHip → LKnee
    (12, 13), # LKnee → LAnkle
    (0, 1),   # Nose → Neck
    (0, 14),  # Nose → REye
    (14, 16), # REye → REar
    (0, 15),  # Nose → LEye
    (15, 17), # LEye → LEar
]

# Per-bone colors (RGB) — matches reference OpenPose rendering gradient
# 17 bones = 17 colors from red through green to blue to purple
BONE_COLORS = [
    (255, 0, 0),       # Neck → RShoulder (red)
    (255, 85, 0),      # Neck → LShoulder
    (255, 170, 0),     # RShoulder → RElbow
    (255, 255, 0),     # RElbow → RWrist (yellow)
    (170, 255, 0),     # LShoulder → LElbow
    (85, 255, 0),      # LElbow → LWrist
    (0, 255, 0),       # Neck → RHip (green)
    (0, 255, 85),      # RHip → RKnee
    (0, 255, 170),     # RKnee → RAnkle
    (0, 255, 255),     # Neck → LHip (cyan)
    (0, 170, 255),     # LHip → LKnee
    (0, 85, 255),      # LKnee → LAnkle
    (0, 0, 255),       # Nose → Neck (blue)
    (85, 0, 255),      # Nose → REye
    (170, 0, 255),     # REye → REar
    (255, 0, 255),     # Nose → LEye (magenta)
    (255, 0, 170),     # LEye → LEar
]

# Per-keypoint colors (RGB) — same gradient, one per keypoint
KEYPOINT_COLORS = [
    (255, 0, 0),       # 0 Nose
    (255, 85, 0),      # 1 Neck
    (255, 170, 0),     # 2 RShoulder
    (255, 255, 0),     # 3 RElbow
    (170, 255, 0),     # 4 RWrist
    (85, 255, 0),      # 5 LShoulder
    (0, 255, 0),       # 6 LElbow
    (0, 255, 85),      # 7 LWrist
    (0, 255, 170),     # 8 RHip
    (0, 255, 255),     # 9 RKnee
    (0, 170, 255),     # 10 RAnkle
    (0, 85, 255),      # 11 LHip
    (0, 0, 255),       # 12 LKnee
    (85, 0, 255),      # 13 LAnkle
    (170, 0, 255),     # 14 REye
    (255, 0, 255),     # 15 LEye
    (255, 0, 170),     # 16 REar
    (255, 0, 85),      # 17 LEar
]

# Body part groups for LLM prompting and high-level color coding
BODY_PARTS = {
    "head": [0, 14, 15, 16, 17],
    "torso": [1, 2, 5, 8, 11],
    "right_arm": [2, 3, 4],
    "left_arm": [5, 6, 7],
    "right_leg": [8, 9, 10],
    "left_leg": [11, 12, 13],
}

# Keypoint labels for LLM prompts — includes index for clarity
KEYPOINT_LABELS = [f"{i}:{name}" for i, name in KEYPOINT_NAMES.items()]
