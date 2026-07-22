"""PyTorch backend for densemaps (CUDA / keops capable).

Importing this subpackage requires ``torch``. The memory-scalable :class:`KernelDistMap`
additionally requires ``pykeops`` (install the ``[keops]`` extra); it raises a clear
``ImportError`` at construction time if keops is missing.
"""

from .maps import (
    PointWiseMap,
    SparseMap,
    P2PMap,
    PreciseMap,
    EmbP2PMap,
    EmbPreciseMap,
    KernelDenseDistMap,
    EmbKernelDenseDistMap,
    KernelDistMap,
)
from .nn_utils import nn_query, nn_query_dist, compute_sqdistmat
from .point_to_triangle import (
    nn_query_precise_torch,
    project_pc_to_triangles,
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
    "KernelDistMap",
    "nn_query",
    "nn_query_dist",
    "compute_sqdistmat",
    "nn_query_precise_torch",
    "project_pc_to_triangles",
]
