"""PoseExtractor — Bridge node to convert DWPose POSE_KEYPOINT output into our keypoints STRING format."""

import json

from ..utils.json_parser import extract_keypoints_from_dwpose


class PoseExtractorNode:
    """Extract body keypoints from DWPose/OpenPose POSE_KEYPOINT output.

    DWPose returns nested JSON with body, face, and hand keypoints.
    This node extracts just the body keypoints into the flat array format
    that PoseManipulator and MotionDirector expect.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
            },
            "optional": {
                "person_index": ("INT", {"default": 0, "min": 0, "max": 20}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    OUTPUT_NAMES = ("keypoints_json", "canvas_height", "canvas_width")
    FUNCTION = "extract"
    CATEGORY = "pose_director"
    DESCRIPTION = "Extract body keypoints from DWPose POSE_KEYPOINT into flat JSON array."

    def extract(self, pose_keypoint, person_index=0):
        kpts, h, w = extract_keypoints_from_dwpose(pose_keypoint, person_index)
        return (json.dumps(kpts), h or 0, w or 0)
