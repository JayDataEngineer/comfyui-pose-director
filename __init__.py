try:
    from .nodes.pose_manipulator import PoseManipulatorNode
    from .nodes.motion_director import MotionDirectorNode
    from .nodes.pose_renderer import PoseRendererNode
    from .nodes.pose_extractor import PoseExtractorNode
    from .nodes.keyframe_reviewer import KeyframeReviewerNode
    from .nodes.body_mesh_renderer import BodyMeshRendererNode
    from .nodes.prompt_enhancer import PromptEnhancerNode
    from .nodes.animation_exporter import AnimationExporterNode

    NODE_CLASS_MAPPINGS = {
        "PoseManipulator": PoseManipulatorNode,
        "MotionDirector": MotionDirectorNode,
        "PoseRenderer": PoseRendererNode,
        "PoseExtractor": PoseExtractorNode,
        "KeyframeReviewer": KeyframeReviewerNode,
        "BodyMeshRenderer": BodyMeshRendererNode,
        "PromptEnhancer": PromptEnhancerNode,
        "AnimationExporter": AnimationExporterNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "PoseManipulator": "Pose Manipulator (LLM)",
        "MotionDirector": "Motion Director (LLM)",
        "PoseRenderer": "Pose Renderer",
        "PoseExtractor": "Pose Extractor (DWPose Bridge)",
        "KeyframeReviewer": "Keyframe Reviewer (Vision LLM)",
        "BodyMeshRenderer": "Body Mesh Renderer (Anny)",
        "PromptEnhancer": "Prompt Enhancer (Z-Image)",
        "AnimationExporter": "Animation Exporter (BVH/Video)",
    }
except ImportError:
    # Not running inside ComfyUI's custom_nodes package context
    # (e.g., during pytest when the root __init__.py is imported standalone)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
