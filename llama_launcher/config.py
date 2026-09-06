"""Configuration file management for llama-launcher."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


# Default config directory name (relative to package)
CONFIG_DIR_NAME = "config"


def get_config_dir() -> Path:
    """Get or create the configuration directory."""
    # Use package directory for config
    package_dir = Path(__file__).parent.parent
    config_dir = package_dir / CONFIG_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_profiles_dir() -> Path:
    """Get or create the profiles directory."""
    profiles_dir = get_config_dir() / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


@dataclass
class ConfigProfile:
    """A saved configuration profile."""
    
    name: str
    file_path: Path
    
    # Core settings
    gpu_layers: int = -1
    fit: str = "on"  # "on", "off" - auto-allocate layers
    context_length: int = 4096
    
    # KV Cache
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: str = "auto"  # "on", "off", "auto"
    defrag_thold: float = 0.0
    
    # Performance
    threads: Optional[int] = None
    batch_size: int = 512
    ubatch_size: int = 512
    
    # Advanced settings (UI stage: parsed/shown, not yet wired into args).
    # null/None means "not set" — llama.cpp applies its own defaults then.
    n_cpu_moe: Optional[int] = None
    cpu_moe: Optional[bool] = None
    kv_offload: Optional[str] = None
    override_tensor: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        d.pop("name", None)
        d.pop("file_path", None)
        return {k: v for k, v in d.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @property
    def display_name(self) -> str:
        """Get display name."""
        return self.file_path.stem
    
    def get_summary_lines(self) -> list[tuple[str, str]]:
        """Get summary as list of (key, value) pairs (base section)."""
        lines = []
        
        # GPU layers
        if self.fit == "on":
            gpu_str = "自动 (fit)"
        elif self.gpu_layers == -1:
            gpu_str = "全部"
        elif self.gpu_layers == 0:
            gpu_str = "纯CPU"
        else:
            gpu_str = str(self.gpu_layers)
        lines.append(("GPU 层数", gpu_str))
        
        # Context length
        lines.append(("上下文长度", str(self.context_length)))
        
        # KV Cache
        lines.append(("K 缓存量化", self.cache_type_k))
        lines.append(("V 缓存量化", self.cache_type_v))
        
        # Flash Attention
        lines.append(("Flash Attention", self.flash_attn))
        
        # Batch sizes
        lines.append(("批处理大小", str(self.batch_size)))
        lines.append(("微批大小", str(self.ubatch_size)))
        
        return lines
    
    def get_advanced_lines(self) -> list[tuple[str, str]]:
        """Get advanced settings as set by the user.

        Only values the user actually set are shown; "not set" means
        llama.cpp will use its own defaults, so no default is displayed.
        """
        lines = []
        lines.append(("CPU 线程", str(self.threads) if self.threads is not None else "(未设置)"))
        lines.append(("n-cpu-moe", str(self.n_cpu_moe) if self.n_cpu_moe is not None else "(未设置)"))
        if self.cpu_moe is None:
            cpu_moe_str = "(未设置)"
        else:
            cpu_moe_str = "on" if self.cpu_moe else "off"
        lines.append(("cpu-moe", cpu_moe_str))
        lines.append(("KV offload", self.kv_offload or "(未设置)"))
        lines.append(("override-tensor", self.override_tensor or "(未设置)"))
        return lines
    
    def get_summary_sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Get summary split into base and advanced sections."""
        return [
            ("基础配置", self.get_summary_lines()),
            ("高级配置", self.get_advanced_lines()),
        ]


def load_profile(file_path: Path) -> Optional[ConfigProfile]:
    """Load a configuration profile from file."""
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        
        # Remove comment fields
        data.pop("_说明", None)
        data.pop("_comment", None)
        data.pop("_doc", None)
        
        # Handle flash_attn: convert bool to str for backward compatibility
        flash_attn_raw = data.get("flash_attn", "auto")
        if isinstance(flash_attn_raw, bool):
            flash_attn = "on" if flash_attn_raw else "off"
        else:
            flash_attn = str(flash_attn_raw)
        
        # cpu_moe: keep null/absent as "not set" (llama.cpp default applies)
        cpu_moe = data.get("cpu_moe")
        
        return ConfigProfile(
            name=file_path.stem,
            file_path=file_path,
            gpu_layers=data.get("gpu_layers", -1),
            fit=data.get("fit", "on"),
            context_length=data.get("context_length", 4096),
            cache_type_k=data.get("cache_type_k", "f16"),
            cache_type_v=data.get("cache_type_v", "f16"),
            flash_attn=flash_attn,
            defrag_thold=data.get("defrag_thold", 0.0),
            threads=data.get("threads"),
            batch_size=data.get("batch_size", 512),
            ubatch_size=data.get("ubatch_size", 512),
            n_cpu_moe=data.get("n_cpu_moe"),
            cpu_moe=cpu_moe,
            kv_offload=data.get("kv_offload"),
            override_tensor=data.get("override_tensor"),
        )
    except Exception as e:
        print(f"Warning: Failed to load profile {file_path}: {e}")
        return None


def list_profiles() -> list[ConfigProfile]:
    """List all available configuration profiles."""
    profiles_dir = get_profiles_dir()
    profiles = []
    
    for file_path in sorted(profiles_dir.glob("*.json")):
        if file_path.name == "template.json":
            continue
        profile = load_profile(file_path)
        if profile:
            profiles.append(profile)
    
    return profiles


def save_profile(profile: ConfigProfile, file_path: Optional[Path] = None) -> bool:
    """Save a configuration profile."""
    try:
        save_path = file_path or profile.file_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Warning: Failed to save profile: {e}")
        return False


def create_default_profiles() -> None:
    """Create default profile files if they don't exist."""
    profiles_dir = get_profiles_dir()
    
    defaults = {
        "gpu-high.json": {
            "gpu_layers": -1,
            "fit": "off",
            "context_length": 8192,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "flash_attn": "on",
            "defrag_thold": 0.0,
            "threads": None,
            "batch_size": 1024,
            "ubatch_size": 1024,
            "n_cpu_moe": None,
            "cpu_moe": None,
            "kv_offload": None,
            "override_tensor": None,
        },
        "balance.json": {
            "gpu_layers": 20,
            "fit": "off",
            "context_length": 4096,
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "flash_attn": "on",
            "defrag_thold": 0.0,
            "threads": None,
            "batch_size": 512,
            "ubatch_size": 512,
            "n_cpu_moe": None,
            "cpu_moe": None,
            "kv_offload": None,
            "override_tensor": None,
        },
        "cpu-only.json": {
            "gpu_layers": 0,
            "fit": "off",
            "context_length": 2048,
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "flash_attn": "off",
            "defrag_thold": 0.0,
            "threads": 8,
            "batch_size": 512,
            "ubatch_size": 512,
            "n_cpu_moe": None,
            "cpu_moe": None,
            "kv_offload": None,
            "override_tensor": None,
        },
    }
    
    for filename, settings in defaults.items():
        file_path = profiles_dir / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)


def create_template() -> None:
    """Create the template configuration file."""
    profiles_dir = get_profiles_dir()
    template_path = profiles_dir / "template.json"
    
    template = {
        "_说明": "llama-launcher 配置文件模板",
        "_用法": "复制此文件并重命名，删除带下划线的说明字段",
        
        "gpu_layers": -1,
        "_gpu_layers_说明": "GPU层数: -1=全部, 0=纯CPU, 其他=指定层数",
        
        "context_length": 4096,
        "_context_length_说明": "上下文长度，越大占用显存越多",
        
        "cache_type_k": "f16",
        "_cache_type_k_说明": "K缓存量化: f16(默认), q8_0(省50%), q4_0(省75%)",
        
        "cache_type_v": "f16",
        "_cache_type_v_说明": "V缓存量化: f16(默认), q8_0(省50%), q4_0(省75%)",
        
        "flash_attn": False,
        "_flash_attn_说明": "Flash Attention: 减少显存占用，推荐开启",
        
        "defrag_thold": 0.0,
        
        "threads": None,
        "_threads_说明": "CPU线程数: null=自动",
        
        "batch_size": 512,
        "_batch_size_说明": "提示处理批大小",
        
        "ubatch_size": 512,
        "_ubatch_size_说明": "推理微批大小",
        
        # --- 高级配置（UI 阶段，暂不传参） ---
        
        "n_cpu_moe": None,
        "_n_cpu_moe_说明": "MoE 前 N 层专家留 CPU (null=不设置，仅 MoE 模型)",
        
        "cpu_moe": None,
        "_cpu_moe_说明": "所有 MoE 专家留 CPU (null=不设置，仅 MoE 模型)",
        
        "kv_offload": None,
        "_kv_offload_说明": "KV 缓存 offload: on/off (null=不设置)",
        
        "override_tensor": None,
        "_override_tensor_说明": '张量级放置, 如 "ffn_down_exps=CPU" (null=不设置)',
    }
    
    if not template_path.exists():
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)


def init_config() -> None:
    """Initialize configuration directory with defaults."""
    get_config_dir()
    get_profiles_dir()
    create_template()
    create_default_profiles()


# Auto-initialize on import
init_config()
