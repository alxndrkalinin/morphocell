"""Extract features from a label image using trimesh."""

from __future__ import annotations

import warnings
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from skimage.measure import marching_cubes

try:  # pragma: no cover - optional dependency
    import trimesh  # type: ignore
    from trimesh.curvature import (
        discrete_gaussian_curvature_measure,
        discrete_mean_curvature_measure,
    )

    _TRIMESH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    trimesh = None  # type: ignore[assignment]
    discrete_gaussian_curvature_measure = None  # type: ignore[assignment]
    discrete_mean_curvature_measure = None  # type: ignore[assignment]
    _TRIMESH_AVAILABLE = False
    warnings.warn(
        "trimesh is not installed. Mesh feature extraction is unavailable.",
        RuntimeWarning,
    )

if TYPE_CHECKING:  # pragma: no cover - typing guard
    import trimesh as _trimesh

from ..cuda import asnumpy


def mesh_feature_list() -> list[str]:
    """Return a list of features extracted from a trimesh object."""
    return [
        "Area",
        "Volume",
        "Min Axis Length",
        "Med Axis Length",
        "Max Axis Length",
        "Scale",
        "Inertia PC1",
        "Inertia PC2",
        "Inertia PC3",
        "Bounding Box Volume",
        "Oriented Bounding Box Volume",
        "Bounding Cylinder Volume",
        "Bounding Sphere Volume",
        "Convex Hull Volume",
        "Convex Hull Area",
        "Sphericity",
        "Extent",
        "Solidity",
        "Avg Gaussian Curvature",
        "Avg Mean Curvature",
    ]


def extract_features(
    label_image: np.ndarray, features: Optional[list[str]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from a label image using trimesh."""
    features = features or mesh_feature_list()
    if not _TRIMESH_AVAILABLE:
        warnings.warn(
            "trimesh is required for mesh feature extraction. Install with 'pip install morphocell[mesh]'.",
            RuntimeWarning,
        )
        return np.array([], dtype=int), np.zeros((0, len(features)))

    label_image = asnumpy(label_image)
    labels = np.unique(label_image)

    feature_values = []
    for label in labels:
        mask = label_image == label
        mesh_features = extract_surface_features(mask)
        feature_values.append([mesh_features[feat] for feat in features])

    return labels, np.asarray(feature_values)


def extract_surface_features(mask: np.ndarray) -> dict[str, float]:
    """Generate mesh from a single label mask and extract its surface features."""
    if not _TRIMESH_AVAILABLE:
        warnings.warn(
            "trimesh is required for mesh feature extraction. Install with 'pip install morphocell[mesh]'.",
            RuntimeWarning,
        )
        return {}

    mesh = mask2mesh(mask)
    if mesh is None:
        return {}

    return extract_mesh_features(mesh)


def mask2mesh(
    mask_3d: np.ndarray, marching_cubes_kwargs: Optional[dict[str, Any]] = None
) -> _trimesh.Trimesh | None:
    """Convert 3D mask to a mesh using marching cubes."""
    if not _TRIMESH_AVAILABLE:
        warnings.warn(
            "trimesh is required for mesh feature extraction. Install with 'pip install morphocell[mesh]'.",
            RuntimeWarning,
        )
        return None
    marching_cubes_kwargs = marching_cubes_kwargs or {}
    verts, faces, _, _ = marching_cubes(mask_3d, **marching_cubes_kwargs)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


def ellipsoid_sphericity(v: float, sa: float) -> float:
    """Calculate the sphericity of an ellipsoid given its volume and surface area."""
    return np.power(np.pi, 1 / 3) * np.power(6 * v, 2 / 3) / sa


def extract_mesh_features(mesh: _trimesh.Trimesh) -> dict[str, float]:
    """Extract surface features from a trimesh object."""
    if not _TRIMESH_AVAILABLE:
        warnings.warn(
            "trimesh is required for mesh feature extraction. Install with 'pip install morphocell[mesh]'.",
            RuntimeWarning,
        )
        return {}

    axis_len = sorted(mesh.extents)
    intertia_pcs = mesh.principal_inertia_components
    volume = mesh.volume
    surface_area = mesh.area

    try:
        bounding_volume = mesh.bounding_cylinder.volume
    except Exception as e:
        bounding_volume = 0
        warnings.warn(
            f"Unable extract bounding cylinder volume. Returning 0. Exception: {e}"
        )

    return {
        "Area": surface_area,
        "Volume": volume,
        "Min Axis Length": axis_len[0],
        "Med Axis Length": axis_len[1],
        "Max Axis Length": axis_len[2],
        "Scale": mesh.scale,
        "Inertia PC1": intertia_pcs[0],
        "Inertia PC2": intertia_pcs[1],
        "Inertia PC3": intertia_pcs[2],
        "Bounding Box Volume": mesh.bounding_box.volume,
        "Oriented Bounding Box Volume": mesh.bounding_box_oriented.volume,
        "Bounding Cylinder Volume": bounding_volume,
        "Bounding Sphere Volume": mesh.bounding_sphere.volume,
        "Convex Hull Volume": mesh.convex_hull.volume,
        "Convex Hull Area": mesh.convex_hull.area,
        "Sphericity": ellipsoid_sphericity(volume, surface_area),
        "Extent": mesh.volume / mesh.bounding_box.volume,
        "Solidity": mesh.volume / mesh.convex_hull.volume,
        "Avg Gaussian Curvature": discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, 1
        ).mean(),
        "Avg Mean Curvature": discrete_mean_curvature_measure(
            mesh, mesh.vertices, 1
        ).mean(),
    }
