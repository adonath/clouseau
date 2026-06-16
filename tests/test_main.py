import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from clouseau.__main__ import Diff, diff, remap_keys
from clouseau.visualize import ArrayDiffFormatter


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
    assert "a: close" in out
    # quantitative divergence is reported, not just pass/fail
    assert "differs" in out
    assert "max_abs=5.00e-01" in out
    assert "3/3 exceed" in out
    # max_rel is taken against the reference (0.5 / 2.0 = 0.25, not 0.5 / 2.5)
    assert "max_rel=2.50e-01" in out
    # reference-only and file-only leaves are surfaced, not mis-aligned
    assert "c: missing in fut.safetensors" in out
    assert "x: only in fut.safetensors, not in reference" in out


def test_remap_keys():
    data = {"layers.0.weight": 1, "layers.1.weight": 2}

    remapped = remap_keys(data, {"layers": "blocks", r"\.weight": ".__call__"})

    assert list(remapped) == ["blocks.0.__call__", "blocks.1.__call__"]
    assert remap_keys(data, {}) is data  # no rules -> untouched


def test_remap_keys_non_injective():
    with pytest.raises(ValueError, match="not injective"):
        remap_keys({"a": 1, "b": 2}, {".": "x"})


def test_diff_key_map(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")

    # reference uses JAX-style naming, file under test uses PyTorch-style naming
    ref = {"blocks.0.__call__": np.ones(3, np.float32)}
    _save(tmp_path / "ref.safetensors", ref, order=["blocks.0.__call__"])

    fut = {"layers.0.weight": np.ones(3, np.float32)}
    _save(tmp_path / "fut.safetensors", fut, order=["layers.0.weight"])

    diff(
        Diff(
            filename=str(tmp_path / "fut.safetensors"),
            filename_ref=str(tmp_path / "ref.safetensors"),
            key_map={"layers": "blocks", r"\.weight": ".__call__"},
        )
    )

    out = capsys.readouterr().out

    # remapped key now aligns with the reference instead of being flagged missing
    assert "blocks.0.__call__: close" in out
    assert "missing" not in out


def test_array_diff_formatter_reports_magnitude():
    fmt = ArrayDiffFormatter()
    ref = np.ones(1000, np.float32)

    # a few-ulp float32 difference is reported as close, not a hard failure
    near = ref + np.float32(5e-6)
    assert "close" in fmt((near, ref))

    far = ref.copy()
    far[0] = 2.0
    out = fmt((far, ref))
    assert "differs" in out
    assert "max_abs=1.00e+00" in out
    assert "1/1000 exceed" in out


def test_array_diff_formatter_shape_mismatch():
    out = ArrayDiffFormatter()((np.ones(3, np.float32), np.ones(4, np.float32)))
    assert "shape mismatch" in out
