# Examples

Interactive demos for `densemaps`. They need PyVista:

```bash
pip install ".[examples]"   # or: pip install pyvista
```

## `interactive_projection.py`

A movable point projected onto a single triangle through three maps at once:

- **vertex-to-vertex** (`EmbP2PMap`) — snaps to the nearest triangle vertex (orange).
- **vertex-to-face** (`EmbPreciseMap`) — closest point on the triangle face (green).
- **kernel, row-normalized** (`EmbKernelDenseDistMap`) — blur-weighted average of the vertices, i.e.
  the expected position; generally not the surface footpoint (purple).

All projections are computed purely via the library's `P.pull_back(V)` (function transfer);
positions are used directly as embeddings. Move the point with the x/y/z sliders and tune the kernel
sharpness with the blur slider — everything updates continuously as you drag.

```bash
python examples/interactive_projection.py            # interactive window
python examples/interactive_projection.py --no-show  # headless numeric check
```
