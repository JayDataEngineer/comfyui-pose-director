"""KeyframeReviewer — Sends keyframe images to a vision LLM for consistency/quality review."""

import torch

from ..utils.api_client import call_llm_vision

SYSTEM_PROMPT = """You are an expert animation reviewer. You will be shown a sequence of keyframe images from an animation pipeline. Each image shows a character in a specific pose.

Review the keyframes for:
1. **Character consistency**: Does the character look the same across all keyframes? (hair color, clothing, proportions, style)
2. **Pose quality**: Are the poses anatomically plausible? Does the motion progression look natural?
3. **Pose progression**: Do the poses flow smoothly from one to the next? Are there any sudden jumps or discontinuities?
4. **Artifacts**: Any visual artifacts, deformations, or issues?

You MUST respond with ONLY a JSON object:
{
  "passed": true/false,
  "score": <1-10>,
  "issues": ["issue1", "issue2", ...],
  "summary": "Brief summary of the review"
}

Be strict but fair. A score of 7+ means the keyframes are good enough for video generation. Below 5 means they should be regenerated."""


class KeyframeReviewerNode:
    """Review keyframe images for consistency and quality using a vision LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "review_prompt": ("STRING", {
                    "default": "Review these keyframe images for character consistency and pose quality.",
                    "multiline": True,
                }),
                "pass_threshold": ("INT", {"default": 7, "min": 1, "max": 10}),
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com/v1beta/openai/"}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "models/gemini-2.5-flash"}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    OUTPUT_NAMES = ("review", "passed", "score")
    FUNCTION = "review"
    CATEGORY = "pose_director"
    DESCRIPTION = "Send keyframe images to a vision LLM for consistency/quality review before video generation."

    def review(self, images, review_prompt, pass_threshold, base_url, api_key, model):
        # Split batch into individual image tensors
        image_list = [images[i] for i in range(images.shape[0])]

        response = call_llm_vision(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            text_prompt=f"{review_prompt}\n\nHere are {len(image_list)} keyframe images in sequence:",
            images=image_list,
        )

        # Parse the response
        from ..utils.json_parser import parse_llm_json
        try:
            result = parse_llm_json(response)
            score = int(result.get("score", 0))
            passed = score >= pass_threshold
            issues = result.get("issues", [])
            summary = result.get("summary", "")

            review_text = f"Score: {score}/10 {'PASSED' if passed else 'FAILED'}\n"
            review_text += f"Summary: {summary}\n"
            if issues:
                review_text += "Issues:\n" + "\n".join(f"  - {i}" for i in issues)

            return (review_text, passed, score)
        except Exception:
            # If JSON parsing fails, return the raw response
            return (response, False, 0)
