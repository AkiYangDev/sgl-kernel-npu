"""Attention and indexer CANNBot-DSL sources."""


def _metadata_output(out, size, device, inputs):
    """Allocate or validate a caller-owned scheduling table without device reads."""
    import torch

    if out is None:
        return torch.empty(size, dtype=torch.int32, device=device)
    if (
        not isinstance(out, torch.Tensor)
        or out.dtype != torch.int32
        or tuple(out.shape) != (size,)
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous int32 [{size}] on {device}")
    start = out.data_ptr()
    end = start + out.numel() * out.element_size()
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        if tensor.device != device:
            continue
        # Include holes in strided views: rejecting overlap is conservative.
        span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        input_start = tensor.data_ptr()
        input_end = input_start + span * tensor.element_size()
        if start < input_end and input_start < end:
            raise ValueError("out must not overlap metadata inputs")
    return out
