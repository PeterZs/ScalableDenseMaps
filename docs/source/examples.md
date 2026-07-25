# Examples

The [`examples/`](https://github.com/RobinMagnet/ScalableDenseMaps/tree/master/examples) folder
contains runnable demos. They need PyVista:

```bash
pip install ".[examples]"   # or: pip install pyvista
```

## Interactive projection

`examples/interactive_projection.py` projects a movable point onto a single triangle through three
maps at once, and shows each projection live as you drag x/y/z sliders:

- **vertex-to-vertex** (`EmbP2PMap`) — snaps to the nearest triangle vertex.
- **vertex-to-face** (`EmbPreciseMap`) — the closest point on the triangle face.
- **kernel, row-normalized** (`EmbKernelDenseDistMap`) — a blur-weighted average of the vertices
  (the *expected* position), tuned live with a blur slider.

The key idea is that positions are used directly as **embeddings**, so the map's embedding space is
plain geometry and each projected location is obtained purely through the library's function
transfer `P.pull_back(V)` — no bespoke projection code. The map builders live in a single `MAPS`
dict, so adding another representation makes it appear in the demo automatically.

```{note}
The window is interactive, so run it locally rather than in a headless environment:

    python examples/interactive_projection.py            # interactive window
    python examples/interactive_projection.py --no-show  # headless numeric check
```

### Source

```{literalinclude} ../../examples/interactive_projection.py
:language: python
:linenos:
```
