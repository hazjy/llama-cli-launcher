"""GGUF model metadata reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelMetadata:
    """Metadata extracted from a GGUF model file."""
    
    # File info
    file_path: Path
    file_size_bytes: int
    
    # General info
    name: Optional[str] = None
    architecture: Optional[str] = None
    file_type: Optional[str] = None
    description: Optional[str] = None
    
    # Model hyperparameters
    context_length: Optional[int] = None
    embedding_length: Optional[int] = None
    block_count: Optional[int] = None
    feed_forward_length: Optional[int] = None
    attention_head_count: Optional[int] = None
    attention_head_count_kv: Optional[int] = None
    
    # Quantization info
    quantization_type: Optional[str] = None
    
    # Vision/Multimodal info
    has_vision_encoder: bool = False
    
    # Speculative decoding module (MTP head or DSpark drafter)
    drafter_path: Optional[Path] = None
    drafter_type: Optional[str] = None
    
    # Raw metadata for advanced use
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    
    # Associated mmproj file for text-only models
    mmproj_path: Optional[Path] = None
    
    @property
    def is_multimodal(self) -> bool:
        """Check if this model supports multimodal input."""
        # GGUF declared a vision encoder (clip.* fields)
        if self.has_vision_encoder:
            return True
        # Has an associated mmproj file
        if self.mmproj_path and self.mmproj_path.exists():
            return True
        # File itself is a multimodal projector
        if self.is_mmproj:
            return True
        return False

    @property
    def is_mmproj(self) -> bool:
        """Check if this file is a mmproj (multimodal projection) file."""
        return self.file_path.name.startswith("mmproj-")
    
    @property
    def file_size_gb(self) -> float:
        """Get file size in GB."""
        return self.file_size_bytes / (1024 ** 3)
    
    @property
    def file_size_display(self) -> str:
        """Get human-readable file size."""
        gb = self.file_size_bytes / (1024 ** 3)
        if gb >= 1.0:
            return f"{gb:.1f}GB"
        mb = self.file_size_bytes / (1024 ** 2)
        return f"{mb:.0f}MB"
    
    @property
    def display_name(self) -> str:
        """Get display name for the model (always use filename)."""
        return self.file_path.stem
    
    @property
    def architecture_display(self) -> str:
        """Get human-readable architecture name."""
        arch_map = {
            "llama": "LLaMA",
            "mistral": "Mistral",
            "mixtral": "Mixtral",
            "qwen2": "Qwen2",
            "qwen3": "Qwen3",
            "qwen35": "Qwen3.5",
            "lfm2moe": "LFM2.5",
            "bailingmoe3": "BailingMoE3",
            "phi2": "Phi-2",
            "phi3": "Phi-3",
            "phi4": "Phi-4",
            "gemma": "Gemma",
            "gemma2": "Gemma 2",
            "gemma3": "Gemma 3",
            "command-r": "Command-R",
            "internlm2": "InternLM2",
            "chatglm": "ChatGLM",
            "mpt": "MPT",
            "falcon": "Falcon",
            "starcoder": "StarCoder",
        }
        if self.architecture:
            return arch_map.get(self.architecture.lower(), self.architecture.upper())
        return "Unknown"
    
    @property
    def model_summary(self) -> str:
        """One-line detail summary of the model (after deep scan)."""
        parts = [self.display_name]
        
        if self.architecture:
            parts.append(self.architecture_display)
        if self.quantization_type:
            parts.append(self.quantization_type)
        
        parts.append(self.file_size_display)
        
        if self.block_count:
            parts.append(f"{self.block_count} 层")
        if self.context_length:
            parts.append(f"ctx {self.context_length}")
        if self.is_multimodal:
            parts.append("多模态")
        if self.drafter_type:
            parts.append(self.drafter_type)
        
        return " | ".join(parts)


def _decode_field(reader: Any, key: str) -> Any:
    """Decode a field value from GGUF reader.
    
    The GGUF reader stores fields with a specific parts layout:
    - parts[0]: key length
    - parts[1]: key bytes
    - parts[2]: type indicator
    - parts[3]: value length (or value for numeric types)
    - parts[4]: value bytes (for string types)
    
    Args:
        reader: GGUFReader instance
        key: Field key to read
        
    Returns:
        Decoded value or None
    """
    try:
        from gguf import GGUFValueType
        import numpy as np
    except ImportError:
        return None
    
    field = reader.get_field(key)
    if field is None:
        return None
    
    for i, t in enumerate(field.types):
        if t == GGUFValueType.STRING:
            if len(field.parts) >= 5:
                val_bytes = field.parts[4]
                return bytes(val_bytes).decode('utf-8', errors='replace')
        elif t in (GGUFValueType.UINT32, GGUFValueType.INT32):
            if len(field.parts) >= 4:
                return int(field.parts[3][0])
        elif t in (GGUFValueType.UINT64, GGUFValueType.INT64):
            if len(field.parts) >= 4:
                return int(field.parts[3][0])
        elif t == GGUFValueType.FLOAT32:
            if len(field.parts) >= 4:
                return float(np.frombuffer(field.parts[3], dtype=np.float32)[0])
    
    return None


def _read_field_str(reader: Any, key: str) -> Optional[str]:
    """Read a string field."""
    val = _decode_field(reader, key)
    if val is None:
        return None
    try:
        return str(val)
    except (ValueError, TypeError):
        return None


def _read_field_int(reader: Any, key: str) -> Optional[int]:
    """Read an integer field."""
    val = _decode_field(reader, key)
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


class MetadataReader:
    """Read metadata from GGUF model files."""
    
    def __init__(self):
        """Initialize the metadata reader."""
        self._reader_class = None
        self._init_reader()
    
    def _init_reader(self):
        """Initialize the GGUF reader library."""
        try:
            from gguf.gguf_reader import GGUFReader
            self._reader_class = GGUFReader
        except ImportError:
            self._reader_class = None
    
    @property
    def is_available(self) -> bool:
        """Check if GGUF reader is available."""
        return self._reader_class is not None
    
    def read_metadata(self, model_path: Path) -> Optional[ModelMetadata]:
        """Read metadata from a GGUF model file.
        
        Args:
            model_path: Path to the .gguf model file
            
        Returns:
            ModelMetadata if successful, None otherwise
        """
        if not model_path.exists():
            return None
        
        if not self.is_available:
            return self._read_basic_metadata(model_path)
        
        try:
            reader = self._reader_class(str(model_path))
            return self._parse_gguf_metadata(model_path, reader)
        except Exception as e:
            print(f"Warning: Failed to read GGUF metadata from {model_path.name}: {e}")
            return self._read_basic_metadata(model_path)
    
    def _read_basic_metadata(self, model_path: Path) -> ModelMetadata:
        """Read basic metadata without GGUF parsing (fallback)."""
        file_size = model_path.stat().st_size
        
        return ModelMetadata(
            file_path=model_path,
            file_size_bytes=file_size,
            name=model_path.stem,
        )
    
    def _parse_gguf_metadata(
        self, model_path: Path, reader: Any
    ) -> ModelMetadata:
        """Parse metadata from GGUF reader."""
        file_size = model_path.stat().st_size
        
        # Extract known fields using new API
        metadata = ModelMetadata(
            file_path=model_path,
            file_size_bytes=file_size,
        )
        
        # General info
        metadata.name = _read_field_str(reader, "general.name")
        metadata.architecture = _read_field_str(reader, "general.architecture")
        metadata.description = _read_field_str(reader, "general.description")
        
        # File type (quantization)
        file_type_id = _read_field_int(reader, "general.file_type")
        if file_type_id is not None:
            metadata.file_type = self._quantization_type_name(file_type_id)
            metadata.quantization_type = metadata.file_type
        
        # Context length - try architecture-specific key first
        arch = metadata.architecture or "llama"
        metadata.context_length = _read_field_int(reader, f"{arch}.context_length")
        
        # Embedding and architecture params
        metadata.embedding_length = _read_field_int(reader, f"{arch}.embedding_length")
        metadata.block_count = _read_field_int(reader, f"{arch}.block_count")
        metadata.feed_forward_length = _read_field_int(reader, f"{arch}.feed_forward_length")
        metadata.attention_head_count = _read_field_int(reader, f"{arch}.attention.head_count")
        metadata.attention_head_count_kv = _read_field_int(reader, f"{arch}.attention.head_count_kv")
        
        # Vision encoder check: presence only, no type identification
        has_vision = _read_field_int(reader, "clip.has_vision_encoder")
        if has_vision is None:
            has_vision = _read_field_int(reader, "clip.vision.embedding_length")
        metadata.has_vision_encoder = bool(has_vision)
        
        return metadata
    
    @staticmethod
    def _quantization_type_name(file_type: int) -> str:
        """Convert GGUF file type integer to name."""
        quant_names = {
            0: "F32",
            1: "F16",
            2: "Q4_0",
            3: "Q4_1",
            7: "Q8_0",
            8: "Q8_1",
            10: "Q2_K",
            11: "Q3_K_S",
            12: "Q3_K_M",
            13: "Q3_K_L",
            14: "Q4_K_S",
            15: "Q4_K_M",
            16: "Q5_K_S",
            17: "Q5_K_M",
            18: "Q6_K",
            26: "IQ2_XXS",
            27: "IQ2_XS",
            28: "IQ2_S",
            29: "IQ2_M",
            30: "IQ3_XXS",
            31: "IQ3_XS",
            32: "IQ1_S",
            33: "IQ4_NL",
            34: "IQ3_S",
            35: "IQ2_S",
            36: "IQ4_XS",
        }
        return quant_names.get(file_type, f"Type_{file_type}")


# Global singleton
_reader: Optional[MetadataReader] = None


def get_reader() -> MetadataReader:
    """Get the global MetadataReader instance."""
    global _reader
    if _reader is None:
        _reader = MetadataReader()
    return _reader


def read_model_metadata(model_path: Path) -> Optional[ModelMetadata]:
    """Read metadata from a GGUF model file.
    
    Args:
        model_path: Path to the .gguf model file
        
    Returns:
        ModelMetadata if successful, None otherwise
    """
    return get_reader().read_metadata(model_path)
