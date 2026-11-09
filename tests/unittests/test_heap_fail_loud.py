# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the symmetric heap's "fail loud" safety checks:

- A collective heap op attempted while a CUDA/HIP graph is captured raises a clear error
  instead of the driver's cryptic "operation not permitted when stream is capturing" / a hang.
- A divergent allocation sequence across ranks is detected and reported precisely, instead of
  surfacing later as a store timeout or a silently wrong reuse.
"""

import gc

import pytest
import torch

import iris

ALLOC_TYPE = "vmem_chunked"


def test_collective_ops_reject_graph_capture(monkeypatch):
    """allocate/collect/refresh/as_symmetric must raise while a graph is capturing.

    Deterministic: force is_current_stream_capturing() True so the guard fires regardless of
    hardware. Removing any _reject_during_graph_capture() call turns the matching assertion red.
    """
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    try:
        # Not capturing yet: a normal allocation works.
        ctx.zeros(128, dtype=torch.float32)
        ctx.barrier()

        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

        with pytest.raises(RuntimeError, match="graph is capturing"):
            ctx.zeros(128, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="graph is capturing"):
            ctx.heap.collect()
        with pytest.raises(RuntimeError, match="graph is capturing"):
            ctx.heap.refresh_peer_access()
        with pytest.raises(RuntimeError, match="graph is capturing"):
            ctx.as_symmetric(torch.randn(16, dtype=torch.float32, device=ctx.device))
    finally:
        monkeypatch.undo()
        ctx.barrier()
        del ctx
        gc.collect()


def test_real_graph_capture_rejects_allocate():
    """Integration: during a real CUDA/HIP graph capture, allocate raises the clear error.

    Confirms is_current_stream_capturing() actually reports True under capture (not just the
    monkeypatched guard). Skips cleanly if graph capture cannot be set up on this stack.
    """
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    try:
        ctx.zeros(64, dtype=torch.float32)  # warm up before capture
        ctx.barrier()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        raised = False
        with torch.cuda.stream(capture_stream):
            try:
                graph.capture_begin()
            except Exception as exc:
                pytest.skip(f"CUDA/HIP graph capture unavailable: {exc}")
            try:
                with pytest.raises(RuntimeError, match="capturing"):
                    ctx.zeros(128, dtype=torch.float32)
                raised = True
            finally:
                try:
                    graph.capture_end()
                except Exception:
                    # Capture was aborted by our guard; ending an incomplete capture may error.
                    pass
        assert raised, "allocate() did not raise during graph capture"
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_layout_divergence_detected():
    """A rank that allocates a different sequence is caught by check_symmetric_layout().

    Rank 0 slips in one extra allocation (small, so it neither grows a chunk nor triggers a
    collective -- no deadlock), shifting its bump pointer. check_symmetric_layout() then raises
    on every rank, naming the diverging fingerprints. Removing the fingerprint comparison in
    _layout_agreed_across_ranks() makes this pass silently -> test goes red.
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    if ctx.get_num_ranks() < 2:
        pytest.skip("requires at least 2 ranks")
    try:
        # Symmetric baseline allocation on every rank.
        keep = ctx.zeros(1024, dtype=torch.float32)  # noqa: F841
        # Divergence: only rank 0 allocates extra. Small enough to avoid chunk growth / refresh.
        if ctx.cur_rank == 0:
            extra = ctx.zeros(2048, dtype=torch.float32)  # noqa: F841

        with pytest.raises(RuntimeError, match="layout diverged across ranks"):
            ctx.heap.check_symmetric_layout()
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_layout_divergence_detected_via_collect():
    """collect() auto-invokes the layout check, so divergence raises there too (not only the public API).

    Covers the in-collect() seam -- the exact point #600's silent divergence otherwise surfaces as a
    store timeout. All ranks call collect() (collective, same order), so there is no deadlock; the
    check raises before the released-offset exchange.
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    if ctx.get_num_ranks() < 2:
        pytest.skip("requires at least 2 ranks")
    try:
        keep = ctx.zeros(1024, dtype=torch.float32)  # noqa: F841
        if ctx.cur_rank == 0:
            extra = ctx.zeros(2048, dtype=torch.float32)  # noqa: F841

        with pytest.raises(RuntimeError, match="layout diverged across ranks"):
            ctx.heap.collect()
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_layout_order_divergence_detected():
    """Same sizes in a different order is caught, even though it leaves bump/mapped_extent identical.

    Two distinct block sizes (1 and 2 granules) allocated in opposite order per rank put the objects
    at different offsets while the cumulative bump pointer matches. Only the order-sensitive
    allocation-sequence hash in the fingerprint distinguishes them; a bump-only fingerprint would miss
    this and silently desync offsets.
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    if ctx.get_num_ranks() < 2:
        pytest.skip("requires at least 2 ranks")
    try:
        g = ctx.heap.allocator.granularity
        small = g // 4  # float32 elements -> one-granule block
        large = (2 * g) // 4  # -> two-granule block (distinct aligned size)
        if ctx.cur_rank == 0:
            a = ctx.zeros(small, dtype=torch.float32)  # noqa: F841
            b = ctx.zeros(large, dtype=torch.float32)  # noqa: F841
        else:
            b = ctx.zeros(large, dtype=torch.float32)  # noqa: F841
            a = ctx.zeros(small, dtype=torch.float32)  # noqa: F841

        # Cumulative bump is identical across ranks; only the sequence hash differs.
        with pytest.raises(RuntimeError, match="layout diverged across ranks"):
            ctx.heap.check_symmetric_layout()
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_layout_agrees_under_asymmetric_gc():
    """Identical allocation sequence must NOT be flagged, even if frees happen at different times.

    The fingerprint (bump, mapped_extent, num_chunks) excludes the live set, which differs by GC
    timing, so dropping a tensor on one rank but not the other must not trip the check. A fingerprint
    that counted live allocations would raise here (false positive).
    """
    ctx = iris.iris(64 << 20, allocator_type=ALLOC_TYPE)
    if ctx.get_num_ranks() < 2:
        pytest.skip("requires at least 2 ranks")
    try:
        a = ctx.zeros(2048, dtype=torch.float32)  # noqa: F841
        b = ctx.zeros(2048, dtype=torch.float32)
        c = ctx.zeros(2048, dtype=torch.float32)  # noqa: F841

        # Asymmetric GC: rank 0 drops the middle tensor now; other ranks keep it. Frees do not
        # move bump/mapped_extent/num_chunks, so layouts still agree.
        if ctx.cur_rank == 0:
            del b
            gc.collect()

        ctx.heap.check_symmetric_layout()  # must not raise
        ctx.barrier()
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


def test_check_symmetric_layout_single_rank_is_noop():
    """check_symmetric_layout() is a no-op at a single rank (nothing to compare)."""
    ctx = iris.iris(4 << 20, allocator_type=ALLOC_TYPE)
    try:
        ctx.zeros(1024, dtype=torch.float32)
        if ctx.get_num_ranks() == 1:
            ctx.heap.check_symmetric_layout()  # returns without a store exchange
    finally:
        ctx.barrier()
        del ctx
        gc.collect()


if __name__ == "__main__":
    # test_collective_ops_reject_graph_capture needs the pytest monkeypatch fixture; run via pytest.
    test_real_graph_capture_rejects_allocate()
    test_layout_divergence_detected()
    test_layout_divergence_detected_via_collect()
    test_layout_order_divergence_detected()
    test_layout_agrees_under_asymmetric_gc()
    test_check_symmetric_layout_single_rank_is_noop()
