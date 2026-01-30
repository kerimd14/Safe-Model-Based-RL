# npz_builder.py
import os
import numpy as np
from typing import Any, Dict

class NPZBuilder:
    """
    Collect arrays/metadata and write one compressed NPZ at the end.
    Designed for adding already-shaped NumPy arrays once at the end.
    """
    def __init__(self, out_dir: str, stem: str, float_dtype: str | None = None):
        self.out_dir = out_dir
        self.stem = stem
        self.float_dtype = float_dtype  # e.g. "float32" to shrink files
        os.makedirs(out_dir, exist_ok=True)
        self._store: Dict[str, Any] = {}

    def add(self, **kwargs):
        """Add arrays (expects well-formed shapes)."""
        for k, v in kwargs.items():
            if v is None:
                continue
            arr = np.asarray(v)
            if self.float_dtype and arr.dtype.kind == "f":
                arr = arr.astype(self.float_dtype, copy=False)
            self._store[k] = arr

    def meta(self, **kwargs):
        """Add scalars/strings/small constants; saved as meta__* keys."""
        for k, v in kwargs.items():
            if isinstance(v, (str, bytes)):
                self._store[f"meta__{k}"] = np.array([v]).astype("U")
            elif np.isscalar(v):
                self._store[f"meta__{k}"] = np.asarray([v])
            else:
                self._store[f"meta__{k}"] = np.asarray(v)

    def finalize(self, suffix: str = "") -> str:
        fname = f"{self.stem}{('_' + suffix) if suffix else ''}.npz"
        out_path = os.path.join(self.out_dir, fname)
        np.savez_compressed(out_path, **self._store)
        return out_path
