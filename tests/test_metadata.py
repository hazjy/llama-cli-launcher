"""Tests for multimodal detection in model metadata."""

from pathlib import Path

from llama_launcher.metadata import ModelMetadata


def make_model() -> ModelMetadata:
    return ModelMetadata(file_path=Path(r"C:\models\test-Q4_K_M.gguf"), file_size_bytes=1024**3)


def test_text_model_not_multimodal():
    m = make_model()
    assert not m.is_multimodal


def test_mmproj_path_marks_multimodal(workdir):
    mmproj = workdir / "mmproj-test-BF16.gguf"
    mmproj.write_bytes(b"x")
    m = make_model()
    m.mmproj_path = mmproj
    assert m.is_multimodal


def test_missing_mmproj_path_not_multimodal():
    m = make_model()
    m.mmproj_path = Path(r"C:\nope\missing.gguf")
    assert not m.is_multimodal


def test_has_vision_encoder_marks_multimodal():
    m = make_model()
    m.has_vision_encoder = True
    assert m.is_multimodal


def test_mmproj_filename_marks_multimodal():
    m = ModelMetadata(file_path=Path(r"C:\models\mmproj-test-BF16.gguf"), file_size_bytes=123)
    assert m.is_mmproj
    assert m.is_multimodal


def test_model_summary_detail_line():
    m = make_model()
    m.architecture = "qwen35"
    m.quantization_type = "Q6_K"
    m.block_count = 32
    m.context_length = 1048576
    s = m.model_summary
    assert "test-Q4_K_M" in s
    assert "Qwen3.5" in s
    assert "Q6_K" in s
    assert "32 层" in s
    assert "多模态" not in s  # not multimodal yet
    m.has_vision_encoder = True
    assert "多模态" in m.model_summary