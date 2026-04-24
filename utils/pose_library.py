"""Pose seed library for base pose lookup and loading."""

import json
from pathlib import Path

POSES_DIR = Path(__file__).parent.parent / "poses"


def list_poses(tags: list[str] | None = None) -> list[dict]:
    """List available poses, optionally filtered by tags.

    Args:
        tags: If provided, only return poses that have at least one matching tag.
              Case-insensitive matching.
    Returns:
        List of pose dicts with 'name', 'description', 'tags' keys (no keypoints).
    """
    results = []
    for path in sorted(POSES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entry = {"name": data["name"], "description": data["description"], "tags": data["tags"]}
        if tags:
            tags_lower = {t.lower() for t in tags}
            pose_tags_lower = {t.lower() for t in data.get("tags", [])}
            if not tags_lower & pose_tags_lower:
                continue
        results.append(entry)
    return results


def load_pose(name: str) -> dict:
    """Load a specific pose by name.

    Args:
        name: Pose name without extension (e.g., 'standing_neutral').
    Returns:
        Full pose dict including 'keypoints' array.
    Raises:
        FileNotFoundError: If pose file doesn't exist.
    """
    path = POSES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Pose '{name}' not found at {path}")
    return json.loads(path.read_text())


def find_best_pose(description: str) -> dict | None:
    """Find the best matching pose for a text description.

    Uses simple keyword matching: tokenizes the description, checks
    against pose names, tags, and description words. Returns the pose
    with the highest match count, or None if no matches.
    """
    words = set(description.lower().split())
    best_pose = None
    best_score = 0

    for path in POSES_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        score = 0
        name_words = set(data["name"].lower().replace("_", " ").split())
        tag_words = {t.lower() for t in data.get("tags", [])}
        desc_words = set(data["description"].lower().split())

        # +3 for name match, +2 for tag match, +1 for description word match
        score += 3 * len(words & name_words)
        score += 2 * len(words & tag_words)
        score += len(words & desc_words)

        if score > best_score:
            best_score = score
            best_pose = data

    return best_pose
