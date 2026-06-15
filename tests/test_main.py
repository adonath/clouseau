import json

import numpy as np
from safetensors.numpy import save_file

from clouseau.__main__ import Diff, diff


def _save(path, arrays, order):
    """Write a safetensors file with the key ordering metadata clouseau expects."""
    metadata = {"order": json.dumps({str(idx): key for idx, key in enumerate(order)})}
    save_file(arrays, path, metadata=metadata)


def test_diff(tmp_path, capsys, monkeypatch):
    # keep rich from wrapping the tree so substring assertions stay stable
    monkeypatch.setenv("COLUMNS", "200")

    ref = {
        "a": np.ones(3, np.float32),
        "b": np.full(3, 2.0, np.float32),
        "c": np.full(3, 3.0, np.float32),
    }
    _save(tmp_path / "ref.safetensors", ref, order=["a", "b", "c"])

    # different file order, "b" differs, "c" missing, extra "x"
    fut = {
        "b": np.full(3, 2.5, np.float32),
        "a": np.ones(3, np.float32),
        "x": np.zeros(3, np.float32),
    }
    _save(tmp_path / "fut.safetensors", fut, order=["b", "a", "x"])

    diff(
        Diff(
            filename=str(tmp_path / "fut.safetensors"),
            filename_ref=str(tmp_path / "ref.safetensors"),
        )
    )

    out = capsys.readouterr().out

    # aligned by name despite the differing on-disk order
    assert "a: Equal to tolerance" in out
    # data_ref is the reference: value is ACTUAL, reference is DESIRED
    assert "Not equal to tolerance" in out
    assert "2.5 (ACTUAL), 2.0 (DESIRED)" in out
    # reference-only and file-only leaves are surfaced, not mis-aligned
    assert "c: missing in fut.safetensors" in out
    assert "x: only in fut.safetensors, not in reference" in out
