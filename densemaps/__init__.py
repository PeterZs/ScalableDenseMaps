"""densemaps: abstract, memory-scalable correspondence maps for 3D geometry.

Two interchangeable backends are provided as subpackages:

- :mod:`densemaps.numpy` -- NumPy/SciPy, CPU only, no PyTorch required.
- :mod:`densemaps.torch` -- PyTorch (CUDA/keops) backend; adds the memory-scalable
  ``KernelDistMap``. Importing it requires ``torch`` (install the ``[torch]`` extra).

The top-level package intentionally does **not** import the torch backend, so that
``import densemaps`` works in a torch-free environment.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
