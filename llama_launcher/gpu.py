"""GPU detection and information for llama.cpp."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    
    name: str
    index: int
    total_memory_mb: int
    free_memory_mb: int
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    
    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory_mb / 1024
    
    @property
    def free_memory_gb(self) -> float:
        """Free memory in GB."""
        return self.free_memory_mb / 1024
    
    @property
    def used_memory_mb(self) -> int:
        """Used memory in MB."""
        return self.total_memory_mb - self.free_memory_mb
    
    @property
    def memory_utilization(self) -> float:
        """Memory utilization percentage."""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100
    
    @property
    def display_name(self) -> str:
        """Display name with memory info."""
        return f"{self.name} ({self.total_memory_gb:.1f}GB)"


class GPUDetector:
    """Detect and query GPU information."""
    
    def __init__(self):
        """Initialize GPU detector."""
        self._gpu_cache: Optional[list[GPUInfo]] = None
    
    def detect_gpus(self, force_refresh: bool = False) -> list[GPUInfo]:
        """Detect available GPUs.
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            List of detected GPUs
        """
        if self._gpu_cache is not None and not force_refresh:
            return self._gpu_cache
        
        gpus = self._detect_nvidia_gpus()
        if not gpus:
            gpus = self._detect_amd_gpus()
        
        self._gpu_cache = gpus
        return gpus
    
    def _detect_nvidia_gpus(self) -> list[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode != 0:
                return []
            
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        total_memory_mb=int(float(parts[2])),
                        free_memory_mb=int(float(parts[3])),
                        driver_version=parts[4],
                    ))
            
            return gpus
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return []
    
    def _detect_amd_gpus(self) -> list[GPUInfo]:
        """Detect AMD GPUs using Windows Registry (most reliable for VRAM)."""
        import json
        import winreg
        
        gpus = []
        
        # Method 1: Read from Windows Registry (most accurate for AMD)
        try:
            # Enumerate GPU devices in registry
            base_key = winreg.HKEY_LOCAL_MACHINE
            sub_key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            
            with winreg.OpenKey(base_key, sub_key) as key:
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            try:
                                driver_desc = winreg.QueryValueEx(subkey, "DriverDesc")[0]
                                if "AMD" in driver_desc.upper():
                                    # Get memory size from registry
                                    try:
                                        mem_size = winreg.QueryValueEx(subkey, r"HardwareInformation.qwMemorySize")[0]
                                        total_mb = mem_size // (1024 * 1024)
                                    except FileNotFoundError:
                                        total_mb = 0
                                    
                                    # Get driver version
                                    try:
                                        driver_version = winreg.QueryValueEx(subkey, "DriverVersion")[0]
                                    except FileNotFoundError:
                                        driver_version = None
                                    
                                    gpus.append(GPUInfo(
                                        index=len(gpus),
                                        name=driver_desc,
                                        total_memory_mb=total_mb,
                                        free_memory_mb=total_mb,  # Will be estimated later
                                        driver_version=driver_version,
                                    ))
                            except FileNotFoundError:
                                pass
                        i += 1
                    except OSError:
                        break
            
            if gpus:
                return gpus
                
        except Exception:
            pass
        
        # Method 2: Fallback to PowerShell
        return self._detect_amd_gpus_powershell()
    
    def _detect_amd_gpus_powershell(self) -> list[GPUInfo]:
        """Detect AMD GPUs using PowerShell."""
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-CimInstance -ClassName Win32_VideoController | Where-Object {$_.Name -like '*AMD*'} | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode != 0:
                return []
            
            import json
            data = json.loads(result.stdout)
            
            # Handle single or multiple GPUs
            if not isinstance(data, list):
                data = [data] if data else []
            
            gpus = []
            for i, gpu in enumerate(data):
                vram_bytes = gpu.get("AdapterRAM", 0) or 0
                total_mb = vram_bytes // (1024 * 1024) if vram_bytes > 0 else 8192
                
                gpus.append(GPUInfo(
                    index=i,
                    name=gpu.get("Name", f"AMD GPU {i}"),
                    total_memory_mb=total_mb,
                    free_memory_mb=total_mb,
                    driver_version=gpu.get("DriverVersion"),
                ))
            
            return gpus
            
        except Exception:
            return []
    
    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return len(self.detect_gpus()) > 0
    
    @property
    def gpu_count(self) -> int:
        """Number of detected GPUs."""
        return len(self.detect_gpus())
    
    @property
    def total_free_memory_mb(self) -> int:
        """Total free GPU memory across all GPUs."""
        return sum(g.free_memory_mb for g in self.detect_gpus())
    
    def get_max_gpu_layers(
        self,
        model_size_gb: float,
        context_length: int = 4096,
    ) -> int:
        """Estimate maximum GPU layers based on available memory.
        
        Args:
            model_size_gb: Model file size in GB
            context_length: Context length to allocate for
            
        Returns:
            Recommended max GPU layers (-1 for all)
        """
        gpus = self.detect_gpus()
        if not gpus:
            return 0  # CPU only
        
        # Rough estimation: need ~1.2x model size for inference
        # Plus additional memory for KV cache
        kv_cache_mb = context_length * 0.5  # Very rough estimate
        
        total_free_mb = sum(g.free_memory_mb for g in gpus)
        model_mb = model_size_gb * 1024
        required_mb = model_mb * 1.2 + kv_cache_mb
        
        if total_free_mb >= required_mb:
            return -1  # Can fit everything on GPU
        
        # Estimate layers based on memory ratio
        if model_mb > 0:
            ratio = total_free_mb / required_mb
            # Round down to nearest layer (rough estimate)
            return max(1, int(ratio * 100))
        
        return 0  # CPU only


# Global singleton
_detector: Optional[GPUDetector] = None


def get_detector() -> GPUDetector:
    """Get the global GPUDetector instance."""
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector


def detect_gpus() -> list[GPUInfo]:
    """Detect available GPUs."""
    return get_detector().detect_gpus()


def has_gpu() -> bool:
    """Check if any GPU is available."""
    return get_detector().has_gpu
