"""OpenPose data structures and constants."""

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

# Skeleton bones as pairs of keypoint indices
SKELETON_BONES = [
    (1, 2),   # Neck → RShoulder
    (1, 5),   # Neck → LShoulder
    (2, 3),   # RShoulder → RElbow
    (3, 4),   # RElbow → RWrist
    (5, 6),   # LShoulder → LElbow
    (6, 7),   # LElbow → LWrist
    (1, 8),   # Neck → RHip (mid-spine)
    (1, 11),  # Neck → LHip (mid-spine)
    (8, 9),   # RHip → RKnee
    (9, 10),  # RKnee → RAnkle
    (11, 12), # LHip → LKnee
    (12, 13), # LKnee → LAnkle
    (0, 1),   # Nose → Neck
    (0, 14),  # Nose → REye
    (0, 15),  # Nose → LEye
    (14, 16), # REye → REar
    (15, 17), # LEye → LEar
]

# Body part groups for LLM prompting
BODY_PARTS = {
    "head": [0, 14, 15, 16, 17],
    "torso": [1, 2, 5, 8, 11],
    "right_arm": [2, 3, 4],
    "left_arm": [5, 6, 7],
    "right_leg": [8, 9, 10],
    "left_leg": [11, 12, 13],
}

# Colors for rendering (BGR format for OpenCV compatibility)
BODY_PART_COLORS = {
    "head": (255, 255, 255),
    "torso": (255, 200, 100),
    "right_arm": (100, 255, 100),
    "left_arm": (100, 100, 255),
    "right_leg": (255, 100, 100),
    "left_leg": (100, 255, 255),
}

KEYPOINT_COLOR = (255, 255, 255)
