densemaps — abstract correspondence maps for 3D geometry
========================================================

``densemaps`` is a lightweight library for representing **correspondence maps** between 3D shapes
(surfaces, point clouds) under a single interface, with interchangeable **NumPy** and **PyTorch**
backends.

A map $T : S_2 \to S_1$ is treated as an $n_2 \times n_1$ matrix, but is never materialised unless
needed. The central operation is *pull-back* (function transfer) $f_{pb} = T f$, which — together
with map composition and nearest-neighbour extraction — works identically across every
representation:

- **Vertex-to-vertex** (:class:`~densemaps.numpy.maps.P2PMap`) — an index per point.
- **Vertex-to-point / barycentric** (:class:`~densemaps.numpy.maps.PreciseMap`) — at most 3
  non-zeros per row.
- **Soft / kernel** (:class:`~densemaps.numpy.maps.KernelDenseDistMap`, and the memory-scalable
  keops-backed :class:`densemaps.torch.maps.KernelDistMap`) — dense row-stochastic maps.

The memory-scalable machinery is described in *Memory-Scalable and Simplified Functional Map
Learning* (Magnet & Ovsjanikov, CVPR 2024, https://arxiv.org/abs/2404.00330).

Installation
------------

The NumPy backend has no PyTorch dependency; install the ``torch`` / ``keops`` extras only if you
need the PyTorch backend or the memory-scalable ``KernelDistMap``:

.. code-block:: bash

   git clone https://github.com/RobinMagnet/ScalableDenseMaps.git
   cd ScalableDenseMaps
   pip install .                 # NumPy backend only
   pip install ".[torch]"        # + PyTorch backend
   pip install ".[torch,keops]"  # + memory-scalable KernelDistMap (pykeops)

Quickstart
----------

.. code-block:: python

   from densemaps.torch import maps

   # Per-vertex embeddings for the two shapes.
   emb1 = ...  # (N1, p)
   emb2 = ...  # (N2, p)

   # A memory-scalable kernel map S2 -> S1; the (N2, N1) matrix is never stored.
   P21 = maps.KernelDistMap(emb1, emb2, blur=1e-1)

   P21.cuda()            # move to GPU (and .cpu() back)

   uv1 = ...             # some function on S1, e.g. uv-coordinates (N1, 2)
   uv2 = P21 @ uv1       # transferred to S2 (N2, 2)  ==  P21.pull_back(uv1)

   p2p_21 = P21.get_nn() # collapse to a vertex-to-vertex map (N2,)
   P21_dense = P21.to_dense()  # materialise the (N2, N1) matrix (small problems only)

The same API is available on the CPU-only NumPy backend as ``from densemaps.numpy import maps``.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   examples
   api
