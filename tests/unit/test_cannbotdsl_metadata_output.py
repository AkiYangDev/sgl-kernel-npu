"""CPU checks for caller-owned metadata buffers; no DSL or NPU import."""

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/cannbot-dsl/attention/__init__.py"
)
spec = importlib.util.spec_from_file_location("metadata_allocation", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
allocate = module._metadata_output


class TestMetadataOutput(unittest.TestCase):
    def test_allocate_and_reuse(self):
        out = allocate(None, 1024, torch.device("cpu"), ())
        out.fill_(23)
        self.assertEqual(out.shape, (1024,))
        self.assertEqual(out.dtype, torch.int32)
        self.assertIs(allocate(out, 1024, out.device, (torch.ones(2), None)), out)
        self.assertTrue(torch.all(out == 23))  # validation never clears outputs

    def test_invalid_outputs(self):
        for out in (
            object(),
            torch.empty(1023, dtype=torch.int32),
            torch.empty(1024, dtype=torch.int64),
            torch.empty(1024, 1, dtype=torch.int32),
            torch.empty(2048, dtype=torch.int32)[::2],
        ):
            with self.subTest(out=type(out)), self.assertRaises(ValueError):
                allocate(out, 1024, torch.device("cpu"), ())
        with self.assertRaises(ValueError):
            allocate(
                torch.empty(1024, dtype=torch.int32), 1024, torch.device("meta"), ()
            )

    def test_aliases_and_disjoint_views(self):
        storage = torch.zeros(2048, dtype=torch.int32)
        out = storage[:1024]
        for inp in (out, storage[1023:1025], storage[:2048:2]):
            with self.subTest(input_shape=inp.shape), self.assertRaisesRegex(
                ValueError, "overlap"
            ):
                allocate(out, 1024, out.device, (inp,))
        self.assertIs(
            allocate(out, 1024, out.device, (storage[1024:], storage[:0])), out
        )


if __name__ == "__main__":
    unittest.main()
