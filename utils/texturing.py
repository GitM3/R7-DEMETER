import numpy as np
import open3d as o3d
from typing import Tuple


def _grid_faces(m: int, n: int) -> np.ndarray:
    """Generate triangle faces for an m x n grid, matching utils.pcd.grid_to_mesh."""
    faces = []
    for i in range(m - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    return np.asarray(faces, dtype=np.int32)


def _grid_vertex_uvs(m: int, n: int, flip_u: bool = False, flip_v: bool = True, rotate_deg: int = 0) -> np.ndarray:
    """Return per-vertex UVs for an m x n grid: u=j/(n-1), v=i/(m-1).

    By default v is flipped (1 - v) to match common image origin conventions.
    flip_u optionally mirrors horizontally; useful to adjust texture orientation without rebuilding geometry.
    rotate_deg rotates UVs in 90-degree increments clockwise (0/90/180/270).
    """
    us = np.linspace(0.0, 1.0, num=n)
    vs = np.linspace(0.0, 1.0, num=m)
    if flip_u:
        us = 1.0 - us
    if flip_v:
        vs = 1.0 - vs
    U, V = np.meshgrid(us, vs)
    uvs = np.stack([U, V], axis=-1).reshape(-1, 2)

    # Rotate in the UV plane
    rotate_deg = rotate_deg % 360
    if rotate_deg not in (0, 90, 180, 270):
        raise ValueError("rotate_deg must be one of {0, 90, 180, 270}")
    if rotate_deg == 90:
        uvs = np.stack([uvs[:, 1], 1.0 - uvs[:, 0]], axis=-1)
    elif rotate_deg == 180:
        uvs = 1.0 - uvs
    elif rotate_deg == 270:
        uvs = np.stack([1.0 - uvs[:, 1], uvs[:, 0]], axis=-1)
    return uvs.astype(np.float64)


def _expand_triangle_uvs(faces: np.ndarray, vertex_uvs: np.ndarray) -> np.ndarray:
    """Expand per-vertex UVs to per-triangle-corner UVs of shape [num_tris*3, 2]."""
    tri_uvs = []
    for f in faces:
        tri_uvs.append(vertex_uvs[f[0]])
        tri_uvs.append(vertex_uvs[f[1]])
        tri_uvs.append(vertex_uvs[f[2]])
    return np.asarray(tri_uvs, dtype=np.float64)


def textured_mesh_from_grid(grid_xyz: np.ndarray, texture_path: str, flip_u: bool = False, flip_v: bool = True, rotate_deg: int = 0) -> o3d.geometry.TriangleMesh:
    """Build a TriangleMesh from a [m,n,3] grid and attach UV + texture.

    - UVs: canonical param mapping (u=j/(n-1), v=i/(m-1)), with optional vertical flip.
    - Texture: loaded via Open3D `read_image`.
    """
    assert grid_xyz.ndim == 3 and grid_xyz.shape[-1] == 3, "grid must be [m,n,3]"
    m, n, _ = grid_xyz.shape

    vertices = grid_xyz.reshape(-1, 3)
    faces = _grid_faces(m, n)
    v_uvs = _grid_vertex_uvs(m, n, flip_u=flip_u, flip_v=flip_v, rotate_deg=rotate_deg)
    tri_uvs = _expand_triangle_uvs(faces, v_uvs)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)

    # Load texture
    img = o3d.io.read_image(texture_path)
    mesh.textures = [img]

    mesh.compute_vertex_normals()
    return mesh


def attach_uv_to_grid_mesh(mesh: o3d.geometry.TriangleMesh, grid_shape: Tuple[int, int], flip_u: bool = False, flip_v: bool = True, rotate_deg: int = 0) -> None:
    """Attach canonical UVs to an existing grid mesh with known [m,n] shape."""
    m, n = grid_shape
    faces = np.asarray(mesh.triangles)
    if faces.size == 0:
        raise ValueError("Mesh has no triangles; cannot attach UVs")
    v_uvs = _grid_vertex_uvs(m, n, flip_u=flip_u, flip_v=flip_v, rotate_deg=rotate_deg)
    tri_uvs = _expand_triangle_uvs(faces, v_uvs)
    expected = len(faces) * 3
    if len(tri_uvs) != expected:
        raise ValueError(f"UV count {len(tri_uvs)} does not match triangle corners {expected} for grid {m}x{n}")
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
    # Open3D expects one material id per triangle when textures are present.
    if len(mesh.triangle_material_ids) != len(faces):
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(faces), dtype=np.int32))
