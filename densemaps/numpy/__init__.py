"""NumPy/SciPy backend for densemaps (CPU only, no PyTorch required)."""

from .maps import (
    PointWiseMap,
    SparseMap,
    P2PMap,
    PreciseMap,
    EmbP2PMap,
    EmbPreciseMap,
    KernelDenseDistMap,
    EmbKernelDenseDistMap,
)
from .nn_utils import knn_query, compute_sqdistmat
from .point_to_triangle import (
    nn_query_precise_np,
    project_pc_to_triangles,
    barycentric_to_precise,
)

__all__ = [
    "PointWiseMap",
    "SparseMap",
    "P2PMap",
    "PreciseMap",
    "EmbP2PMap",
    "EmbPreciseMap",
    "KernelDenseDistMap",
    "EmbKernelDenseDistMap",
    "knn_query",
    "compute_sqdistmat",
    "nn_query_precise_np",
    "project_pc_to_triangles",
    "barycentric_to_precise",
]
