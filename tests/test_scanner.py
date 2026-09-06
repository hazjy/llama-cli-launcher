"""Tests for MTP / DSpark drafter module detection and quick/deep scan."""

from llama_launcher.scanner import ModelScanner


def test_no_drafter(workdir):
    model = workdir / "Model-Q4_K_M.gguf"
    model.write_bytes(b"x")
    assert ModelScanner().find_drafter_for_model(model) is None


def test_mtp_drafter(workdir):
    model = workdir / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
    mtp = workdir / "mtp-gemma-4-26B-A4B-it.gguf"
    model.write_bytes(b"x")
    mtp.write_bytes(b"x")
    found = ModelScanner().find_drafter_for_model(model)
    assert found is not None
    path, spec_type = found
    assert path == mtp
    assert spec_type == "draft-mtp"


def test_dspark_drafter(workdir):
    model = workdir / "LFM2.5-8B-A1B-Q5_K_M.gguf"
    dsp = workdir / "LFM2.5-8B-A1B-DSpark-F16.gguf"
    model.write_bytes(b"x")
    dsp.write_bytes(b"x")
    found = ModelScanner().find_drafter_for_model(model)
    assert found is not None
    path, spec_type = found
    assert path == dsp
    assert spec_type == "draft-dspark"


def test_mmproj_generic_name_fallback(workdir):
    # "mmproj-BF16.gguf" (no model-name prefix) must still be found
    # when it is the only mmproj in the directory.
    model = workdir / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
    mmproj = workdir / "mmproj-BF16.gguf"
    model.write_bytes(b"x")
    mmproj.write_bytes(b"x")
    found = ModelScanner().find_mmproj_for_model(model)
    assert found == mmproj


def test_mmproj_generic_ambiguous_not_found(workdir):
    # Two generic mmprojs in the same dir -> do not guess
    model = workdir / "Model-Q4_K_M.gguf"
    (workdir / "mmproj-A.gguf").write_bytes(b"x")
    (workdir / "mmproj-B.gguf").write_bytes(b"x")
    model.write_bytes(b"x")
    assert ModelScanner().find_mmproj_for_model(model) is None


def test_scan_quick_lists_files_without_metadata(workdir):
    (workdir / "ModelA-Q4_K_M.gguf").write_bytes(b"x")
    (workdir / "ModelB-Q6_K.gguf").write_bytes(b"x")
    (workdir / "mmproj-ModelA-BF16.gguf").write_bytes(b"x")
    (workdir / "ModelA-DSpark-F16.gguf").write_bytes(b"x")
    scanner = ModelScanner(model_dirs=[workdir], recursive=False)
    result = scanner.scan_quick()
    assert result.count == 2
    for md in result.models:
        assert md.architecture is None  # no GGUF metadata read
        assert md.file_size_bytes == 1
    a = next(m for m in result.models if "ModelA" in m.display_name)
    assert a.mmproj_path is not None
    assert a.drafter_type == "draft-dspark"


def test_scan_deep_reads_metadata(workdir):
    model = workdir / "ModelA-Q4_K_M.gguf"
    model.write_bytes(b"not a real gguf 0123456789")
    scanner = ModelScanner(model_dirs=[workdir], recursive=False)
    quick = scanner.scan_quick().models[0]
    deep = scanner.scan_deep(quick)
    assert deep is not None
    assert deep.architecture is None  # garbage file -> basic metadata fallback
    assert deep.file_size_bytes == quick.file_size_bytes