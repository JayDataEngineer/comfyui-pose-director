from .nodes.pose_manipulator import PoseManipulatorNode
from .nodes.motion_director import MotionDirectorNode
from .nodes.pose_renderer import PoseRendererNode

NODE_CLASS_MAPPINGS = {
    "PoseManipulator": PoseManipulatorNode,
    "MotionDirector": MotionDirectorNode,
    "PoseRenderer": PoseRendererNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseManipulator": "🦴 Pose Manipulator (LLM)",
    "MotionDirector": "🎬 Motion Director (LLM)",
    "PoseRenderer": "🦴 Pose Renderer",
}
