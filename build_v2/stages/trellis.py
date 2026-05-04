"""Trellis stage — 3D model generation via Ray TRELLIS service.

Delegates to the Ray-hosted TRELLIS service for image-to-3D conversion.
The TRELLIS service is managed by the Ray cluster and exposed via HTTP.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from build_v2.stages.base import Stage, StageResult


class TrellisStage(Stage):
    """Generate 3D model from character image via Ray TRELLIS service.

    Calls the Ray-hosted TRELLIS HTTP service to convert the approved
    character image into a 3D model (GLB format). The TRELLIS service
    must be available in the Ray cluster.

    Environment:
        TRELLIS_URL: URL of the TRELLIS service (default: http://localhost:18080/3d/trellis)
    """

    name = "trellis"
    tier_required = "t2"
    depends_on = ["generate"]

    _DEFAULT_URL = "http://localhost:18080/3d/trellis"

    def _get_trellis_url(self) -> str:
        return os.environ.get("TRELLIS_URL", self._DEFAULT_URL)

    def is_complete(self, state: dict) -> bool:
        trellis_dir = self.ctx.output_dir / "trellis"
        return trellis_dir.exists() and bool(list(trellis_dir.glob("*.glb")))

    def run(self, state: dict) -> StageResult:
        ctx = self.ctx

        if not ctx.approved_image.exists():
            return StageResult(status="failed", error="Approved image not found")

        trellis_dir = ctx.output_dir / "trellis"
        trellis_dir.mkdir(parents=True, exist_ok=True)

        trellis_url = self._get_trellis_url()

        try:
            import requests

            with open(ctx.approved_image, "rb") as f:
                resp = requests.post(
                    trellis_url,
                    files={"image": (ctx.approved_image.name, f, "image/png")},
                    data={"name": ctx.character_name},
                    timeout=600,  # 10 min — TRELLIS can be slow
                )
            resp.raise_for_status()

        except ImportError:
            return StageResult(status="failed", error="requests library not available")
        except Exception as e:
            return StageResult(status="failed", error=f"TRELLIS service error: {e}")

        # Parse response — TRELLIS returns the GLB file directly or a JSON with paths
        content_type = resp.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                result = resp.json()
                model_url = result.get("model_url") or result.get("url", "")
                if not model_url:
                    return StageResult(
                        status="failed",
                        error=f"TRELLIS returned no model URL: {result}",
                    )
                # Download the model
                model_resp = requests.get(model_url, timeout=120)
                model_resp.raise_for_status()
                model_bytes = model_resp.content
            except (json.JSONDecodeError, KeyError) as e:
                return StageResult(
                    status="failed",
                    error=f"TRELLIS response parse error: {e}",
                )
        else:
            # Response is the model file directly
            model_bytes = resp.content

        if not model_bytes:
            return StageResult(status="failed", error="TRELLIS produced empty output")

        out_path = trellis_dir / f"{ctx.character_name}.glb"
        out_path.write_bytes(model_bytes)

        artifacts = [str(out_path.relative_to(ctx.output_dir))]
        return StageResult(
            status="complete",
            artifacts=artifacts,
            meta={"format": "glb", "source": "trellis"},
        )
