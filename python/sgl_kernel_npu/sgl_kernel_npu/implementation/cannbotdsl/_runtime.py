"""Load the pinned DSL backend only when an A5 operator is selected."""

import importlib
import importlib.metadata
import os
from functools import lru_cache
from pathlib import Path

DSL_VERSION = "0.8.0"
AICPU_LIBRARIES = (
    "mqsmla_metadata_kernel.so",
    "metadata_kernel.so",
    "qsli_metadata_kernel.so",
)


@lru_cache(maxsize=1)
def initialize():
    try:
        version = importlib.metadata.version("cannbot-dsl")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "This A5 operator requires cannbot-dsl==0.8.0. "
            "Install the matching Linux/Python wheel."
        ) from error
    if version != DSL_VERSION:
        raise RuntimeError(
            f"This backend requires cannbot-dsl=={DSL_VERSION}; found {version}"
        )

    source = Path(__file__).resolve().parent / "attention"
    native = source / "_native"
    if native.is_dir():
        missing = [
            name for name in AICPU_LIBRARIES if not (source / "_aicpu" / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"Incomplete CANNBot-DSL AICPU bundle: {missing}")
        from cannbotdsl.aot import register_directory

        register_directory(native)
    elif os.environ.get("CANNBOTDSL_NATIVE_BINARY_MODE") == "require":
        raise RuntimeError(
            "This wheel has no CANNBot-DSL Native bundle. "
            "Build with --with-cannbot-dsl, or explicitly allow JIT execution."
        )


@lru_cache(maxsize=None)
def load_operator(name):
    initialize()
    module = importlib.import_module(f".attention.{name}", __package__)
    return getattr(module, name)
