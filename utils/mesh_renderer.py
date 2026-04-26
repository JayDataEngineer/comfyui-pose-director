"""Deterministic 3D mesh → 2D image rendering using pyrender.

Renders Anny body meshes as gray silhouettes with studio lighting on white
backgrounds. No diffusion — pure geometry and OpenGL.

Provides a PIL fallback renderer for environments without GPU/OpenGL support.
"""

import numpy as np
from typing import Optional


def render_mesh_pyrender(
    vertices: np.ndarray,
    faces: np.ndarray,
    width: int = 1024,
    height: int = 1024,
) -> np.ndarray:
    """Render a 3D mesh to a 2D image using pyrender (headless OpenGL).

    Args:
        vertices: (V, 3) float32 vertex positions.
        faces: (F, 3) int32 triangle face indices.
        width: Output image width.
        height: Output image height.

    Returns:
        (H, W, 3) uint8 numpy array — gray body on white background.
    """
    import trimesh
    import pyrender

    # Center and scale mesh to fit viewport with headroom above
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    verts_centered = vertices - bbox_center

    # Scale based on vertical (Z) extent to prevent head clipping
    z_max = np.abs(verts_centered[:, 2]).max()
    x_max = np.abs(verts_centered[:, 0]).max()
    # Use the larger of half-height (with headroom) or half-width
    scale_extent = max(z_max * 1.4, x_max * 1.1)
    if scale_extent > 0:
        verts_centered = verts_centered * (1.0 / scale_extent)
    # Shift mesh down slightly so head has extra room
    verts_centered[:, 2] -= 0.05

    mesh = trimesh.Trimesh(vertices=verts_centered, faces=faces)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh,
        vertex_colors=np.full((len(verts_centered), 4), [180, 180, 180, 255], dtype=np.uint8),
    )

    # Scene with white background
    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[0.3, 0.3, 0.3],
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    # Camera — orthographic, front-facing along -Y axis (Anny is Z-up)
    camera = pyrender.OrthographicCamera(xmag=1.2, ymag=1.2)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 3.0  # distance from origin along Z
    # Rotate to look along -Y: rotate 90° around X
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    camera_pose = rot_x @ camera_pose
    scene.add(camera, pose=camera_pose)

    # Key light (upper right)
    key_pose = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=key_pose)

    # Fill light (lower left, dimmer)
    fill_pose = np.array([
        [1, 0, 0, -2],
        [0, -1, 0, -1],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add(pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=1.5), pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, _depth = renderer.render(scene)
    renderer.delete()

    return color


def render_mesh_pil(
    vertices: np.ndarray,
    faces: np.ndarray,
    width: int = 1024,
    height: int = 1024,
) -> np.ndarray:
    """Fallback: render mesh to 2D using PIL (no GPU/OpenGL required).

    Projects vertices orthographically, draws filled triangles with flat shading.
    Lower quality than pyrender but works everywhere.

    Returns:
        (H, W, 3) uint8 numpy array.
    """
    from PIL import Image, ImageDraw

    # Center and scale — bounding box center, extra headroom
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    verts = vertices - bbox_center
    z_max = np.abs(verts[:, 2]).max()
    x_max = np.abs(verts[:, 0]).max()
    scale_extent = max(z_max * 1.4, x_max * 1.1)
    if scale_extent > 0:
        verts = verts * (1.0 / scale_extent)
    verts[:, 2] -= 0.05

    # Orthographic projection: use X and -Z (screen coords)
    # Anny's up axis is Z, not Y — so project X→screen_x, Z→screen_y
    screen_x = (verts[:, 0] * (width / 2) + width / 2).astype(int)
    screen_y = (-verts[:, 2] * (height / 2) + height / 2).astype(int)

    # Depth for painter's algorithm (back-to-front)
    # Viewing along -Y axis, so depth = Y coordinate
    depth_vals = verts[:, 1]
    face_depths = depth_vals[faces].mean(axis=1)
    sorted_indices = np.argsort(-face_depths)  # back to front

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for fi in sorted_indices:
        tri = faces[fi]
        pts = [
            (int(screen_x[tri[0]]), int(screen_y[tri[0]])),
            (int(screen_x[tri[1]]), int(screen_y[tri[1]])),
            (int(screen_x[tri[2]]), int(screen_y[tri[2]])),
        ]
        # Depth-based shading
        depth = face_depths[fi]
        shade = max(120, min(220, int(170 + depth * 30)))
        draw.polygon(pts, fill=(shade, shade, shade))

    return np.array(img)


def _rotate_vertices_y(vertices: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate vertices around the Y-axis (vertical in Z-up convention).

    Args:
        vertices: (V, 3) vertex positions.
        angle_deg: Rotation angle in degrees. Positive = clockwise from above.
            0° = front, 90° = right-facing, 180° = back, 270° = left-facing.

    Returns:
        Rotated vertices (same shape).
    """
    if abs(angle_deg) < 0.01:
        return vertices
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([
        [cos_t,  0, sin_t],
        [0,      1, 0    ],
        [-sin_t, 0, cos_t],
    ], dtype=np.float64)
    return vertices @ rot.T


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    width: int = 1024,
    height: int = 1024,
    backend: str = "auto",
    model_rotation_y: float = 0.0,
) -> np.ndarray:
    """Render a 3D mesh to 2D image.

    Args:
        vertices: (V, 3) float32/float64 vertex positions.
        faces: (F, 3) int32 triangle indices.
        width: Output image width.
        height: Output image height.
        backend: "pyrender", "pil", or "auto" (try pyrender, fall back to PIL).
        model_rotation_y: Rotate entire body around Y-axis in degrees.
            0=front, 90=right, 180=back, 270=left.

    Returns:
        (H, W, 3) uint8 numpy array.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    # Apply whole-body rotation
    vertices = _rotate_vertices_y(vertices, model_rotation_y)

    if backend == "auto":
        try:
            return render_mesh_pyrender(vertices, faces, width, height)
        except Exception as e:
            print(f"[mesh_renderer] pyrender failed ({e}), falling back to PIL")
            return render_mesh_pil(vertices, faces, width, height)
    elif backend == "pyrender":
        return render_mesh_pyrender(vertices, faces, width, height)
    else:
        return render_mesh_pil(vertices, faces, width, height)
