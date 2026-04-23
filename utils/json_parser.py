"""Robust JSON parsing for LLM responses."""

import json
import re


def parse_llm_json(response: str):
    """Parse JSON from an LLM response with multiple fallback strategies.

    Tries in order:
    1. Direct JSON parse
    2. Strip markdown code blocks (```json ... ```)
    3. Regex extraction of first JSON object or array
    4. Clean common LLM artifacts (trailing commas, comments)

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON (dict or list).

    Raises:
        ValueError: If no valid JSON could be extracted.
    """
    text = response.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code block
    if "```" in text:
        blocks = text.split("```")
        for block in blocks[1:]:
            if block.startswith("json"):
                block = block[4:]
            elif block.startswith("JSON"):
                block = block[4:]
            block = block.strip()
            # Remove trailing ``` if present
            if block.endswith("```"):
                block = block[:-3].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Strategy 3: Regex extraction — find first JSON object or array
    # Match balanced braces/brackets
    for pattern in (r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try fixing common issues
                fixed = _fix_common_json_issues(candidate)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue

    # Strategy 4: Broader regex — grab everything between first {/[
    for opener, closer in [('{', '}'), ('[', ']')]:
        start = text.find(opener)
        if start >= 0:
            # Find matching closer
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        fixed = _fix_common_json_issues(candidate)
                        try:
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            break

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON issues from LLM output."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Remove JS-style comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    # Remove trailing commas in general
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def extract_keypoints_from_dwpose(pose_data, person_index=0):
    """Extract body keypoints from DWPose/OpenPose nested format.

    DWPose returns: {"people": [{"pose_keypoints_2d": [...], ...}], "canvas_height": ..., "canvas_width": ...}
    We want just the flat [x, y, conf, ...] body keypoints array.

    Handles COCO-17 (17 keypoints) by converting to COCO-18 (18 keypoints)
    with a zero-confidence placeholder for the Neck keypoint.

    Also handles normalized (0-1) coordinates from OpenposePreprocessor by
    converting them to pixel coordinates using canvas dimensions.

    Args:
        pose_data: Dict from DWPose or a flat list of keypoints.
        person_index: Which person to extract (default 0).

    Returns:
        Tuple of (flat_keypoints_list, canvas_height, canvas_width)
    """
    # Already a flat list of keypoints
    if isinstance(pose_data, list) and len(pose_data) > 0 and isinstance(pose_data[0], (int, float)):
        return _normalize_keypoints(pose_data), None, None

    # List of people dicts
    if isinstance(pose_data, list) and len(pose_data) > 0 and isinstance(pose_data[0], dict):
        if person_index >= len(pose_data):
            person_index = 0
        person = pose_data[person_index]
        kpts = person.get("pose_keypoints_2d", person.get("keypoints", []))
        return _normalize_keypoints(kpts), None, None

    # DWPose/OpenPose nested format
    if isinstance(pose_data, dict):
        # Could be {"people": [...], "canvas_height": ..., "canvas_width": ...}
        if "people" in pose_data:
            people = pose_data["people"]
            if person_index >= len(people):
                person_index = 0
            person = people[person_index]
            kpts = person.get("pose_keypoints_2d", person.get("keypoints", []))
            h = pose_data.get("canvas_height")
            w = pose_data.get("canvas_width")
            kpts = _normalize_keypoints(kpts)
            kpts = _denormalize_coords(kpts, h, w)
            return kpts, h, w

        # Single person without wrapper
        if "pose_keypoints_2d" in pose_data or "keypoints" in pose_data:
            kpts = pose_data.get("pose_keypoints_2d", pose_data.get("keypoints", []))
            h = pose_data.get("canvas_height")
            w = pose_data.get("canvas_width")
            kpts = _normalize_keypoints(kpts)
            kpts = _denormalize_coords(kpts, h, w)
            return kpts, h, w

        # Could be a list of keypoints as values
        return list(pose_data.values()), None, None

    return [], None, None


def _denormalize_coords(kpts, canvas_h, canvas_w):
    """Convert normalized (0-1) coordinates to pixel coordinates if needed.

    OpenposePreprocessor returns normalized coords while DWPose returns pixel coords.
    Detection heuristic: if all x <= 1.0 and all y <= 1.0, coords are normalized.
    """
    if not kpts or not canvas_h or not canvas_w:
        return kpts

    # Check if coordinates are normalized (all x and y values <= 1.0)
    is_normalized = True
    for i in range(0, len(kpts), 3):
        if kpts[i] > 1.0 or kpts[i + 1] > 1.0:
            is_normalized = False
            break

    if not is_normalized:
        return kpts

    # Convert to pixel coordinates
    result = []
    for i in range(0, len(kpts), 3):
        result.extend([kpts[i] * canvas_w, kpts[i + 1] * canvas_h, kpts[i + 2]])
    return result


def _normalize_keypoints(kpts):
    """Ensure keypoints are in COCO-18 format (54 floats).

    If COCO-17 (51 floats), insert Neck keypoint at index 1
    by averaging LShoulder and RShoulder positions.
    """
    if not kpts:
        return kpts

    num_points = len(kpts) // 3

    # Already COCO-18 or other 18+ point format
    if num_points >= 18:
        return kpts[:54]  # Trim to first 18 keypoints

    # COCO-17 format: same as COCO-18 but missing Neck (index 1)
    # COCO-17 indices: 0=Nose, 1=RShoulder, 2=RElbow, ..., 16=LEar
    # COCO-18 indices: 0=Nose, 1=Neck, 2=RShoulder, ..., 17=LEar
    if num_points == 17:
        # Insert Neck at index 1 (average of RShoulder and LShoulder)
        nose_x, nose_y, nose_c = kpts[0], kpts[1], kpts[2]
        rsho_x, rsho_y = kpts[3], kpts[4]  # COCO-17 idx 1 = RShoulder
        lsho_x, lsho_y = kpts[15], kpts[16]  # COCO-17 idx 5 = LShoulder
        neck_x = (rsho_x + lsho_x) / 2
        neck_y = (rsho_y + lsho_y) / 2
        neck_c = min(nose_c, 0.9)

        # Build COCO-18: Nose, Neck, then shift everything
        result = [nose_x, nose_y, nose_c, neck_x, neck_y, neck_c]
        result.extend(kpts[3:])  # RShoulder through LEar
        return result

    # Unknown format — return as-is
    return kpts
