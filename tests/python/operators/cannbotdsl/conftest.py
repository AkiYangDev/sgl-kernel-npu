"""Keep the migrated independent references on CPU and select NPU explicitly."""

import os

import pytest


@pytest.fixture(autouse=True)
def configure_public_interface_test(request, monkeypatch):
    # Override the parent two-implementation/performance fixture: these suites
    # compare against independent CPU golden functions, not torch_native.
    torch = pytest.importorskip("torch")
    previous = None
    if request.node.get_closest_marker("npu"):
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("An A5 NPU is required")
        from tests.python.utils.device import validate_npu_options

        index = request.config.getoption("npu_device", default=None)
        if index is None:
            index = int(os.environ.get("NPU_DEVICE_ID", torch.npu.current_device()))
        validate_npu_options(index, request.config.getoption("numprocesses", default=0))
        if index >= torch.npu.device_count():
            raise pytest.UsageError("NPU device index is outside the visible range")
        previous = torch.npu.current_device()
        torch.npu.set_device(index)
        monkeypatch.setenv("NPU_DEVICE_ID", str(index))
    try:
        # Upstream golden functions create implicit-device tensors and use FP64.
        # Device inputs are explicitly moved to NPU by each test.
        with torch.device("cpu"):
            yield
    finally:
        if previous is not None:
            torch.npu.set_device(previous)
