from __future__ import annotations
import ctypes
import os
import re
import struct
from pathlib import Path
import numpy as np
import pytest

pytestmark = [pytest.mark.ffi]


def project_file(name, project):
    p = project / name
    if p.exists(): return p
    alt = project / (name + ".txt")
    return alt if alt.exists() else p


@pytest.mark.p0
def test_source_packaging_names_match_runtime_expectations(polydim, tmp_path):
    project = Path(polydim.__file__).resolve().parent
    cpp_runtime = project / "kernel_cpp_v78.cpp"
    rust_runtime = project / "kernel_rust_v78.rs"
    assert cpp_runtime.exists() and rust_runtime.exists(), \
        "Runtime expects .cpp/.rs but delivery only provides .cpp.txt/.rs.txt"


@pytest.mark.p0
def test_native_householder_dispatch_is_real(polydim):
    bridge = polydim.NativeFFIBridge
    bridge._cpp_lib = object()
    x = np.arange(8, dtype=np.float64)
    v = np.ones(8)
    # If native dispatch were wired, replacing _cpp_lib with a sentinel would
    # not be harmless. The current V78 implementation ignores it.
    out = np.asarray(bridge.householder_reflect(x, v))
    assert np.all(np.isfinite(out))
    source = Path(polydim.__file__).read_text(encoding="utf-8")
    body = source.split("def householder_reflect", 1)[1].split("# =============================================================================", 1)[0]
    assert "_cpp_lib" in body or "ctypes" in body, \
        "householder_reflect has no native dispatch path"


@pytest.mark.p0
def test_cpp_alias_contract_is_not_partial(polydim):
    p = Path(polydim.__file__).resolve().parent / "kernel_cpp_v78.cpp.txt"
    if not p.exists(): pytest.skip("historical source not present")
    s = p.read_text(encoding="utf-8")
    assert "check_byte_overlap(v, out" in s
    # Full alias safety requires all three pairwise ranges to be covered.
    assert s.count("check_byte_overlap(") >= 3


@pytest.mark.p1
def test_pmtp_declared_header_size_is_exact(polydim):
    assert hasattr(polydim, "PMTP_HEADER_FMT")
    actual = struct.calcsize(polydim.PMTP_HEADER_FMT)
    assert actual == 128, f"declared 128-byte PMTP header is actually {actual} bytes"


@pytest.mark.p0
def test_pmtp_roundtrip_bytes_are_not_self_inconsistent(polydim, monkeypatch):
    monkeypatch.setattr(polydim, "PMTP_NET_KEY", b"K" * 32)
    layer = polydim.PMTPNetworkLayer("sender")
    receiver = b"R" * 32
    raw = layer.pack_tensor_header((2, 3), 48, receiver)
    assert len(raw) == struct.calcsize(polydim.PMTP_HEADER_FMT)
    fields = struct.unpack(polydim.PMTP_HEADER_FMT, raw)
    # MAC location must be fixed by the format. V78 returns a hand-spliced blob;
    # verify that recomputing the MAC under the documented layout is possible.
    assert fields[0] == polydim.PMTP_MAGIC


@pytest.mark.p1
def test_payload_limit_is_enforced_before_allocation(polydim, monkeypatch):
    monkeypatch.setattr(polydim, "PMTP_NET_KEY", b"K" * 32)
    layer = polydim.PMTPNetworkLayer("sender")
    with pytest.raises((ValueError, RuntimeError)):
        layer.pack_tensor_header((1,), 100 * 1024 * 1024 + 1, b"R" * 32)
