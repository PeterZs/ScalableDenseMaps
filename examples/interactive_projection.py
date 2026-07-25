"""Interactive projection of a moving point onto a mesh, via densemaps.

A single triangle plays the role of the target shape ``S1``; a movable point plays the role of the
(single-vertex) source shape ``S2``. We build a correspondence map ``S2 -> S1`` from *positions as
embeddings*, so the map's embedding space is plain 3D geometry and its projection is the geometric
closest point. The projected location is obtained purely through the library's function transfer
``P.pull_back(V)`` — no bespoke projection code:

- ``EmbP2PMap``            (vertex-to-vertex): pull_back picks the nearest triangle **vertex**.
- ``EmbPreciseMap``        (vertex-to-face):   pull_back returns the closest point on the **face**
  (barycentric combination of the 3 vertices).
- ``EmbKernelDenseDistMap`` (kernel, row-normalized): pull_back returns a soft, blur-weighted
  **average** of the vertices (the expected position) -- generally not the surface footpoint.

All projections are shown at once. Move the point with the x/y/z sliders and tune the kernel
sharpness with the blur slider. Sliders update continuously as you drag.

Run:
    python examples/interactive_projection.py            # interactive window
    python examples/interactive_projection.py --no-show  # headless numeric check
"""

import os
import sys

import numpy as np

# Allow running straight from a checkout (`python examples/interactive_projection.py`) without
# installing the package first.
try:
    from densemaps.numpy.maps import EmbP2PMap, EmbPreciseMap, EmbKernelDenseDistMap
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from densemaps.numpy.maps import EmbP2PMap, EmbPreciseMap, EmbKernelDenseDistMap

# --- Geometry: target shape S1 is a single triangle -------------------------------------------
V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (3, 3) vertices
F = np.array([[0, 1, 2]])  # (1, 3) faces
INIT_P = np.array([0.3, 0.3, 0.6])  # initial position of the movable point
INIT_BLUR = 0.3  # initial kernel blur (Gaussian std over embedding distances)

# Each map: display color + a builder that returns a densemaps map S2 -> S1 from the point `p`
# and the current `blur` (ignored by the maps that don't use it). Add another map here and it
# shows up in the demo automatically.
MAPS = {
    "vertex-to-vertex": ("orange", lambda p, blur: EmbP2PMap(V, p)),
    "vertex-to-face": ("green", lambda p, blur: EmbPreciseMap(V, p, F)),
    "kernel (row-norm)": ("purple", lambda p, blur: EmbKernelDenseDistMap(V, p, blur=blur)),
}


def project(point, blur=INIT_BLUR):
    """Project ``point`` through every map. Pure/testable — no PyVista involved.

    Parameters
    ----------
    point : array-like, shape (3,)
        Position of the movable point (the single vertex of S2).
    blur : float
        Gaussian blur used by the kernel map (ignored by the others).

    Returns
    -------
    dict[str, tuple[np.ndarray, str]]
        ``{map_name: (projected_xyz (3,), color)}``.
    """
    p = np.asarray(point, dtype=float).reshape(1, 3)  # (n2=1, 3) embedding == position
    out = {}
    for name, (color, build) in MAPS.items():
        proj = build(p, blur).pull_back(V)[0]  # transfer the vertex-coordinate function -> (3,)
        out[name] = (np.asarray(proj, dtype=float), color)
    return out


def smoke_check():
    """Headless numeric sanity check (no display)."""
    projections = project(INIT_P, INIT_BLUR)
    print(f"point = {INIT_P.tolist()}  blur = {INIT_BLUR}")
    for name, (proj, _) in projections.items():
        print(f"  {name:>18s} -> {np.round(proj, 6).tolist()}")

    vv = projections["vertex-to-vertex"][0]
    vf = projections["vertex-to-face"][0]

    # vertex-to-vertex must land exactly on one of the triangle vertices.
    assert np.isclose(np.linalg.norm(V - vv, axis=1), 0.0).any(), "v2v is not a triangle vertex"
    # vertex-to-face must lie in the triangle plane (z = 0 here).
    assert abs(vf[2]) < 1e-9, "v2f is not in the triangle plane"

    # Kernel point sharpens toward the nearest vertex as blur -> 0 and toward the centroid as
    # blur -> inf (a convex combination of the vertices either way).
    ker_sharp = project(INIT_P, blur=0.02)["kernel (row-norm)"][0]
    ker_soft = project(INIT_P, blur=50.0)["kernel (row-norm)"][0]
    assert np.linalg.norm(ker_sharp - vv) < 1e-3, "kernel should snap to nearest vertex at low blur"
    assert (
        np.linalg.norm(ker_soft - V.mean(0)) < 1e-2
    ), "kernel should tend to centroid at high blur"
    print("smoke check OK")


def run_interactive():
    import pyvista as pv

    p0 = INIT_P.copy()
    state = {"p": p0.copy(), "blur": INIT_BLUR}

    plotter = pv.Plotter()
    legend = "red = point\n" + "\n".join(f"{color} = {name}" for name, (color, _) in MAPS.items())
    plotter.add_text(legend, font_size=10, position="upper_right")

    # Triangle surface + its vertices.
    tri = pv.PolyData(V, faces=np.hstack([[3], [0, 1, 2]]))
    plotter.add_mesh(tri, color="lightsteelblue", opacity=0.4, show_edges=True, line_width=2)
    plotter.add_point_labels(
        V, ["v0", "v1", "v2"], font_size=14, point_size=10, render_points_as_spheres=True
    )

    # Movable point.
    pt_mesh = pv.PolyData(p0.reshape(1, 3))
    plotter.add_mesh(pt_mesh, color="red", point_size=18, render_points_as_spheres=True)

    # One projection point + connector line per map.
    proj_meshes, line_meshes = {}, {}
    for name, (proj, color) in project(p0, state["blur"]).items():
        proj_meshes[name] = pv.PolyData(proj.reshape(1, 3))
        plotter.add_mesh(
            proj_meshes[name], color=color, point_size=16, render_points_as_spheres=True
        )
        line_meshes[name] = pv.Line(p0, proj)
        plotter.add_mesh(line_meshes[name], color=color, line_width=2)

    def refresh():
        p = state["p"]
        pt_mesh.points = p.reshape(1, 3)
        for name, (proj, _) in project(p, state["blur"]).items():
            proj_meshes[name].points = proj.reshape(1, 3)
            line_meshes[name].points = np.vstack([p, proj])
        plotter.render()

    def make_axis_cb(axis):
        def cb(value):
            state["p"][axis] = value
            refresh()

        return cb

    def blur_cb(value):
        state["blur"] = value
        refresh()

    # Stacked sliders: x, y, z (position) then blur (kernel sharpness). `interaction_event="always"`
    # makes them update continuously while dragging, not only on release.
    for axis, label in enumerate("xyz"):
        y = 0.90 - 0.12 * axis
        plotter.add_slider_widget(
            make_axis_cb(axis),
            rng=(-1.5, 1.5),
            value=float(p0[axis]),
            title=label,
            pointa=(0.025, y),
            pointb=(0.31, y),
            style="modern",
            interaction_event="always",
        )

    plotter.add_slider_widget(
        blur_cb,
        rng=(0.05, 1.5),
        value=float(state["blur"]),
        title="blur",
        pointa=(0.025, 0.90 - 0.12 * 3),
        pointb=(0.31, 0.90 - 0.12 * 3),
        style="modern",
        interaction_event="always",
    )

    plotter.add_axes()
    plotter.show()


def main():
    if "--no-show" in sys.argv:
        smoke_check()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
