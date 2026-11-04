"""CPU-only tests of dispatch, provenance and Native resource packaging.

Compiler/device execution is deliberately not simulated as an accuracy result.
"""

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu"
DSL = PACKAGE / "implementation/cannbotdsl"
UPSTREAM = json.loads((DSL / "UPSTREAM.json").read_text())
NAMES = [Path(item["file"]).stem for item in UPSTREAM["files"]]


def load_file(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "python/sgl_kernel_npu"))
    return importlib.import_module("build_tools.cannbotdsl")


@pytest.fixture
def runtime(monkeypatch):
    module = load_file("_test_dsl_runtime", DSL / "_runtime.py", monkeypatch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.8.0")
    monkeypatch.delenv("CANNBOTDSL_NATIVE_BINARY_MODE", raising=False)
    return module


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("arch", [2201, 3510])
def test_public_dispatch_and_argument_contract(name, arch, monkeypatch):
    # Load real registry/generation code in an isolated package. Only the device
    # query and final kernel call are substituted.
    root = "_dsl_contract"
    for suffix, directory in [
        ("", PACKAGE),
        (".operators", PACKAGE / "operators"),
        (".operators.utils", PACKAGE / "operators/utils"),
        (".operators.attention", PACKAGE / "operators/attention"),
        (".operators.indexer", PACKAGE / "operators/indexer"),
        (".implementation", PACKAGE / "implementation"),
        (".implementation.cannbotdsl", DSL),
    ]:
        module = ModuleType(root + suffix)
        module.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, module.__name__, module)
    generation = load_file(
        root + ".operators.utils.generation",
        PACKAGE / "operators/utils/generation.py",
        monkeypatch,
    )
    monkeypatch.setattr(
        generation,
        "get_hardware",
        lambda: generation.Hardware(arch, generation.generation_from_arch(arch)),
    )
    registry = load_file(
        root + ".operators.utils.registry",
        PACKAGE / "operators/utils/registry.py",
        monkeypatch,
    )
    calls, returned = [], object()
    loader = ModuleType(root + ".implementation.cannbotdsl._runtime")

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return returned

    def resolve(actual):
        assert actual == name
        return record

    loader.load_operator = resolve
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    domain = "attention" if name.startswith("mixed_") else "indexer"
    module = load_file(
        f"{root}.operators.{domain}.{name}",
        PACKAGE / f"operators/{domain}/{name}.py",
        monkeypatch,
    )
    public = getattr(module, name)
    signature = inspect.signature(public)
    original = next(
        n
        for n in ast.parse(
            (DSL / "attention" / f"{name}.py").read_text(encoding="utf-8")
        ).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    original_names = [a.arg for a in original.args.args + original.args.kwonlyargs]
    assert list(signature.parameters) == original_names + ["impl"]
    assert signature.parameters["impl"].kind is inspect.Parameter.KEYWORD_ONLY
    kwargs = {key: object() for key in original_names}
    for impl in ("auto", "cannbotdsl"):
        if arch == 2201:
            with pytest.raises(registry.UnsupportedGenerationError):
                public(**kwargs, impl=impl)
        else:
            assert public(**kwargs, impl=impl) is returned
            assert calls[-1] == ((), kwargs)
    if arch == 3510:
        required = {
            key: value
            for key, value in kwargs.items()
            if signature.parameters[key].default is inspect.Parameter.empty
        }
        public(**required)
        expected = {
            key: required[key] if key in required else signature.parameters[key].default
            for key in original_names
        }
        assert calls[-1] == ((), expected)
    if arch == 2201:
        assert not calls
    with pytest.raises(registry.ImplementationNotFoundError):
        public(**kwargs, impl="triton")


def test_pinned_source_bytes():
    assert len(UPSTREAM["files"]) == 7
    for source in UPSTREAM["files"]:
        data = (DSL / "attention" / source["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        ast.parse(data)


def test_wrong_dsl_version_fails_before_import(runtime, monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "0.7.1")
    with pytest.raises(RuntimeError, match="found 0.7.1"):
        runtime.initialize()


def test_missing_dsl_has_actionable_error(runtime, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="cannbot-dsl==0.8.0"):
        runtime.initialize()


def test_jit_import_does_not_require_native_resources(runtime):
    runtime.initialize()


def test_native_require_rejects_source_only_install(runtime, monkeypatch):
    monkeypatch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "require")
    with pytest.raises(RuntimeError, match="no CANNBot-DSL Native bundle"):
        runtime.initialize()


def test_native_bundle_requires_all_metadata_libraries(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "_runtime.py"))
    native = tmp_path / "attention/_native"
    native.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Incomplete.*AICPU"):
        runtime.initialize()
    aicpu = tmp_path / "attention/_aicpu"
    aicpu.mkdir()
    for name in runtime.AICPU_LIBRARIES:
        (aicpu / name).write_bytes(b"test fixture")
    registrations = []
    module = ModuleType("cannbotdsl.aot")
    module.register_directory = registrations.append
    monkeypatch.setitem(sys.modules, "cannbotdsl.aot", module)
    runtime.initialize()
    runtime.initialize()
    assert registrations == [native]


@pytest.mark.parametrize(
    "version,arch", [("9.2.1", "3510"), ("9.2.0", "2201"), ("9.2", "3510")]
)
def test_build_rejects_other_target_combinations(backend, version, arch):
    with pytest.raises(ValueError, match="CANN 9.2.0"):
        backend.resolve_plan(version, arch)


def make_bundle(backend, tmp_path):
    output = tmp_path / "bundle"
    files = ["_native/manifest.json", "_native/index.json"]
    files += [f"_native/operators/{name}/operator.json" for name in backend.OPERATORS]
    files += [f"_aicpu/{name}" for _, _, name in backend.METADATA]
    for name in files:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture")
    source = tmp_path / "package" / backend.RELATIVE / "attention"
    source.mkdir(parents=True)
    (source / "entry.py").write_text("# same compiled source\n")
    manifest = dict(
        backend.resolve_plan("9.2.0", "3510"),
        sources=backend._hashes(source),
        resources=backend._hashes(output),
    )
    (output / "build-manifest.json").write_text(json.dumps(manifest))
    return output, source


def test_resources_and_sources_are_verified_before_staging(backend, tmp_path):
    output, source = make_bundle(backend, tmp_path)
    result = backend.stage_backend(output, tmp_path / "package")
    assert result["version"] == "0.8.0"
    for _, _, filename in backend.METADATA:
        assert (source / "_aicpu" / filename).read_bytes() == b"test fixture"


@pytest.mark.parametrize(
    "field,value",
    [("target", "dav-2201"), ("operators", ["qli"]), ("backend", "other")],
)
def test_bundle_target_contract_is_checked(backend, tmp_path, field, value):
    output, source = make_bundle(backend, tmp_path)
    path = output / "build-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="bundle contract"):
        backend.stage_backend(output, tmp_path / "package")
    assert not (source / "_native").exists()


@pytest.mark.parametrize("missing", ["torch", "torch_npu"])
def test_missing_torch_dependency_fails_before_build_work(
    backend, tmp_path, monkeypatch, missing
):
    third_party = importlib.import_module("build_tools.third_party")
    monkeypatch.setattr(third_party, "detect_cann_version", lambda: "9.2.0")
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(backend.sys, "version_info", (3, 12, 0))

    def version(name):
        if name == missing:
            raise importlib.metadata.PackageNotFoundError(name)
        return {"cannbot-dsl": "0.8.0", "torch": "2.10.0", "torch_npu": "2.10.0"}[name]

    monkeypatch.setattr(backend.importlib.metadata, "version", version)
    monkeypatch.setattr(backend, "run", lambda *a, **k: pytest.fail("compiler started"))
    with pytest.raises(RuntimeError, match=missing):
        backend.build_backend(source_root=tmp_path)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("change", ["resource", "source", "missing"])
def test_incomplete_or_stale_bundle_is_rejected(backend, tmp_path, change):
    output, source = make_bundle(backend, tmp_path)
    if change == "resource":
        (output / "_aicpu/metadata_kernel.so").write_bytes(b"corrupt")
    elif change == "source":
        (source / "entry.py").write_text("# edited after compilation\n")
    else:
        (output / "_aicpu/metadata_kernel.so").unlink()
    with pytest.raises(RuntimeError, match="checksum|sources changed"):
        backend.stage_backend(output, tmp_path / "package")
    assert not (source / "_native").exists()


def test_compile_worker_uses_separate_aicpu_workdirs(backend, tmp_path, monkeypatch):
    source, output = tmp_path / "sources", tmp_path / "output"
    source.mkdir()
    (source / "__init__.py").write_text("")
    for name, kernel, _ in backend.METADATA:
        (source / f"{name}.py").write_text(f"def {kernel}(): pass\n")
    compiled, exports = [], []
    aot = ModuleType("cannbotdsl.aot")
    aot.build = lambda **kwargs: exports.append(kwargs)
    toolchain = ModuleType("cannbotdsl.aicpu.toolchain")

    def compile_aicpu(kernel, **kwargs):
        path = Path(kwargs["workdir"])
        path.mkdir(parents=True)
        library = path / "kernel.so"
        library.write_bytes(kernel.__module__.encode())
        compiled.append(kwargs)
        return SimpleNamespace(so_path=library)

    toolchain.compile_aicpu_kernel = compile_aicpu
    monkeypatch.setitem(sys.modules, "cannbotdsl.aot", aot)
    monkeypatch.setitem(sys.modules, "cannbotdsl.aicpu.toolchain", toolchain)
    try:
        backend.compile_sources(source, output)
    finally:
        for key in list(sys.modules):
            if key.startswith("_sgl_ds41_metadata"):
                sys.modules.pop(key)
    assert exports == [
        dict(
            collectors=source.resolve(),
            operators=list(backend.OPERATORS),
            target="dav-3510",
            output=output.resolve() / "_native",
        )
    ]
    assert len({item["workdir"] for item in compiled}) == 3
    assert all(
        item["launch_mode"] == "interface" and item["npu_arch"] == "dav-3510"
        for item in compiled
    )
    assert len(list((output / "_aicpu").glob("*.so"))) == 3


def test_cli_plan_and_wheel_tag_include_dsl(backend, monkeypatch, capsys):
    setup = load_file("_dsl_test_setup", ROOT / "setup.py", monkeypatch)
    setup.main(
        [
            "--arch-version",
            "3510",
            "--without-third-party",
            "--with-cannbot-dsl",
            "--plan",
            "--cann-version",
            "9.2.0",
        ]
    )
    assert json.loads(capsys.readouterr().out) == [
        backend.resolve_plan("9.2.0", "3510")
    ]
    wheel = importlib.import_module("build_tools.torch_wheel")
    base = dict(selection_kind="all", arch_version="3510", operators=[], third_party=[])
    assert wheel.wheel_build_tag(base) == "1_refactor_3510_all"
    native = dict(base, cannbotdsl={"version": "0.8.0", "resources": {"x": "a"}})
    first = wheel.wheel_build_tag(native)
    assert "_cb_" in first
    native["cannbotdsl"]["resources"]["x"] = "b"
    assert wheel.wheel_build_tag(native) != first


@pytest.mark.parametrize(
    "arguments",
    [
        ["--arch-version", "2201", "--with-cannbot-dsl"],
        ["--arch-version", "3510", "--artifact", "run", "--with-cannbot-dsl"],
        ["--arch-version", "3510", "--with-cannbot-dsl", "--cann-version", "9.2.0"],
    ],
)
def test_cli_rejects_invalid_dsl_build_modes(backend, monkeypatch, arguments):
    setup = load_file("_dsl_test_setup", ROOT / "setup.py", monkeypatch)
    with pytest.raises(SystemExit):
        setup.arguments(arguments)
