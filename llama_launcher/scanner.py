"""Model directory scanner for GGUF files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .metadata import ModelMetadata, read_model_metadata


@dataclass
class ScanResult:
    """Result of scanning a model directory."""
    
    models: list[ModelMetadata] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        """Number of models found."""
        return len(self.models)
    
    @property
    def has_multimodal(self) -> bool:
        """Check if any multimodal models were found."""
        return any(m.is_multimodal for m in self.models)
    
    def filter_multimodal(self) -> list[ModelMetadata]:
        """Get only multimodal models."""
        return [m for m in self.models if m.is_multimodal]
    
    def filter_text_only(self) -> list[ModelMetadata]:
        """Get only text-only models."""
        return [m for m in self.models if not m.is_multimodal]
    
    def sort_by_size(self, reverse: bool = True) -> None:
        """Sort models by file size."""
        self.models.sort(key=lambda m: m.file_size_bytes, reverse=reverse)
    
    def sort_by_name(self) -> None:
        """Sort models by display name."""
        self.models.sort(key=lambda m: m.display_name.lower())


class ModelScanner:
    """Scan directories for GGUF model files."""
    
    # Default directories to search (platform-specific)
    DEFAULT_DIRS = [
        Path.home() / "models",
        Path.home() / ".cache" / "llama.cpp" / "models",
        Path.home() / ".local" / "share" / "llama.cpp" / "models",
        Path("C:/Users") / os.getenv("USERNAME", "") / "models",
        # Common Windows paths
        Path("D:/llama-b10472-bin-win-vulkan-x64/models"),
        Path("C:/llama.cpp/models"),
        Path("D:/llama.cpp/models"),
    ]
    
    def __init__(
        self,
        model_dirs: Optional[list[Path | str]] = None,
        recursive: bool = True,
        metadata_reader: Optional[Callable[[Path], Optional[ModelMetadata]]] = None,
    ):
        """Initialize the model scanner.
        
        Args:
            model_dirs: Directories to scan. If None, uses defaults.
            recursive: Whether to scan subdirectories recursively.
            metadata_reader: Custom metadata reader function.
        """
        self.model_dirs = []
        if model_dirs:
            for d in model_dirs:
                self.model_dirs.append(Path(d))
        else:
            self.model_dirs = [d for d in self.DEFAULT_DIRS if d.exists()]
        
        self.recursive = recursive
        self.metadata_reader = metadata_reader or read_model_metadata
    
    def scan(
        self,
        progress_callback: Optional[Callable[[Path, int, int], None]] = None,
    ) -> ScanResult:
        """Scan configured directories for GGUF models.
        
        Args:
            progress_callback: Optional callback(current_file, current, total)
            
        Returns:
            ScanResult with found models and any errors
        """
        result = ScanResult()
        
        # First, collect all GGUF files
        all_files: list[Path] = []
        for scan_dir in self.model_dirs:
            if not scan_dir.exists():
                continue
            all_files.extend(self._find_gguf_files(scan_dir))
        
        # Remove duplicates (by resolved path)
        seen: set[Path] = set()
        unique_files: list[Path] = []
        for f in all_files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(f)
        
        total = len(unique_files)
        
        # Read metadata for each file
        for i, gguf_path in enumerate(unique_files):
            if progress_callback:
                progress_callback(gguf_path, i + 1, total)
            
            try:
                metadata = self.metadata_reader(gguf_path)
                if metadata:
                    self._attach_companions(metadata)
                    result.models.append(metadata)
            except Exception as e:
                result.errors.append((gguf_path, str(e)))
        
        return result
    
    def scan_quick(
        self,
        progress_callback: Optional[Callable[[Path, int, int], None]] = None,
    ) -> ScanResult:
        """Fast scan: list GGUF files without reading their metadata.

        Only file names/sizes plus file-name-level companion detection
        (mmproj / MTP / DSpark) are gathered, so this is fast even for
        large model directories. Architecture/quantization/layers stay
        unknown until ``scan_deep`` is called on the chosen model.
        """
        result = ScanResult()
        
        all_files: list[Path] = []
        for scan_dir in self.model_dirs:
            if not scan_dir.exists():
                continue
            all_files.extend(self._find_gguf_files(scan_dir))
        
        seen: set[Path] = set()
        unique_files: list[Path] = []
        for f in all_files:
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(f)
        
        total = len(unique_files)
        for i, gguf_path in enumerate(unique_files):
            if progress_callback:
                progress_callback(gguf_path, i + 1, total)
            metadata = ModelMetadata(
                file_path=gguf_path,
                file_size_bytes=gguf_path.stat().st_size,
            )
            self._attach_companions(metadata)
            result.models.append(metadata)
        
        return result
    
    def scan_deep(self, model: ModelMetadata) -> Optional[ModelMetadata]:
        """Full scan of a single model: read GGUF metadata + attach companions.

        Args:
            model: Lightweight metadata (e.g. from ``scan_quick``)

        Returns:
            A new fully-read ModelMetadata, or None on failure
        """
        try:
            metadata = self.metadata_reader(model.file_path)
        except Exception as e:
            result = ScanResult()
            result.errors.append((model.file_path, str(e)))
            return None
        if metadata is None:
            return None
        self._attach_companions(metadata)
        return metadata
    
    def _attach_companions(self, metadata: ModelMetadata) -> None:
        """Attach mmproj / drafter companions via file-name matching.

        Pure file-name matching — no GGUF content is read here.
        """
        mmproj = self.find_mmproj_for_model(metadata.file_path)
        if mmproj:
            metadata.mmproj_path = mmproj
        drafter = self.find_drafter_for_model(metadata.file_path)
        if drafter:
            metadata.drafter_path, metadata.drafter_type = drafter
    
    def _find_gguf_files(self, directory: Path) -> list[Path]:
        """Find all .gguf files in a directory, excluding mmproj files."""
        files: list[Path] = []
        
        try:
            if self.recursive:
                for item in directory.rglob("*.gguf"):
                    if item.is_file():
                        # Skip mmproj / mtp / *dspark* files: these are companion modules,
                        # not standalone models
                        if self._is_companion_file(item.name):
                            continue
                        files.append(item)
            else:
                for item in directory.glob("*.gguf"):
                    if item.is_file():
                        # Skip companion module files
                        if self._is_companion_file(item.name):
                            continue
                        files.append(item)
        except PermissionError:
            pass
        
        return files
    
    @staticmethod
    def _is_companion_file(name: str) -> bool:
        """Check if a GGUF filename is a companion module (mmproj/mtp/dspark)."""
        return (
            name.startswith("mmproj-")
            or name.startswith("mtp-")
            or "dspark" in name.lower()
        )
    
    def find_mmproj_for_model(self, model_path: Path) -> Optional[Path]:
        """Find the corresponding mmproj file for a model.
        
        Args:
            model_path: Path to the main model file
            
        Returns:
            Path to mmproj file if found, None otherwise
        """
        # Look for mmproj file in the same directory
        model_dir = model_path.parent
        base_name = self._strip_quant_suffix(model_path.stem)
        
        # Try to find matching mmproj file
        # Pattern: mmproj-{base_name}*.gguf
        pattern = f"mmproj-{base_name}*.gguf"
        matches = list(model_dir.glob(pattern))
        if matches:
            return matches[0]
        
        # Fallback: a single generic mmproj-*.gguf in the same dir
        # (e.g. "mmproj-BF16.gguf" in a one-model-per-folder layout)
        generic = list(model_dir.glob("mmproj-*.gguf"))
        if len(generic) == 1:
            return generic[0]
        
        return None
    
    @staticmethod
    def _strip_quant_suffix(model_name: str) -> str:
        """Remove quantization suffix like -Q4_K_M from a model name.

        Examples: "Qwythos-9B-v2-Q4_K_M" -> "Qwythos-9B-v2"
                  "LFM2.5-8B-A1B-Q5_K_M" -> "LFM2.5-8B-A1B"
        """
        for suffix in ["-Q2_K", "-Q3_K_S", "-Q3_K_M", "-Q3_K_L", "-Q4_0", "-Q4_1",
                       "-Q4_K_S", "-Q4_K_M", "-Q5_0", "-Q5_1", "-Q5_K_S", "-Q5_K_M",
                       "-Q6_K", "-Q8_0", "-Q8_1", "-F16", "-F32"]:
            if model_name.endswith(suffix):
                return model_name[:-len(suffix)]
        return model_name
    
    def find_drafter_for_model(
        self, model_path: Path
    ) -> Optional[tuple[Path, str]]:
        """Find an MTP / DSpark drafter module file next to a model.

        Companion files in the same directory:
          - ``mtp-<model>.gguf``          -> "draft-mtp"
          - ``<model>...DSpark....gguf``  -> "draft-dspark"

        Args:
            model_path: Path to the main model file

        Returns:
            ``(path, spec_type)`` tuple or None
        """
        model_dir = model_path.parent
        base_name = self._strip_quant_suffix(model_path.stem)
        
        # MTP head module: mtp-<base>*.gguf
        for f in sorted(model_dir.glob("mtp-*.gguf")):
            module = f.stem[len("mtp-"):]
            if module and base_name.startswith(module):
                return f, "draft-mtp"
        
        # DSpark drafter: <base>...DSpark....gguf (case-insensitive match)
        dspark_files = list(model_dir.glob("*dspark*.gguf"))
        if not dspark_files:
            dspark_files = list(model_dir.glob("*DSpark*.gguf"))
        for f in sorted(dspark_files):
            idx = f.stem.lower().find("dspark")
            if idx <= 0:
                continue
            module = f.stem[:idx].strip("-")
            if base_name.startswith(module) or module.startswith(base_name):
                return f, "draft-dspark"
        
        return None
    
    def scan_single(self, model_path: Path) -> Optional[ModelMetadata]:
        """Scan a single model file.
        
        Args:
            model_path: Path to the GGUF file
            
        Returns:
            ModelMetadata if successful, None otherwise
        """
        if not model_path.exists():
            return None
        
        return self.metadata_reader(model_path)


def scan_default_dirs() -> ScanResult:
    """Quick scan of default model directories."""
    scanner = ModelScanner()
    return scanner.scan()


def scan_directory(path: Path | str) -> ScanResult:
    """Scan a specific directory for models."""
    scanner = ModelScanner(model_dirs=[path])
    return scanner.scan()
