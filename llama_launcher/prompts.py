"""Interactive prompts for configuration."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .metadata import ModelMetadata
from .gpu import GPUInfo, detect_gpus, has_gpu


@dataclass
class LaunchConfig:
    """Configuration for launching llama.cpp."""
    
    # Model
    model: ModelMetadata
    mmproj_path: Optional[Path] = None  # Multimodal projector path
    
    # Launch mode
    mode: str = "server"  # "server" or "cli"
    
    # GPU settings
    gpu_layers: int = -1  # -1 = auto/all, 0 = CPU only, >0 = specific layers
    fit: str = "on"  # "on", "off", "auto" - auto-allocate layers
    
    # Context settings
    context_length: int = 4096
    
    # Server settings
    port: int = 8080
    host: str = "127.0.0.1"
    
    # Advanced settings (null/None = not set; llama.cpp defaults apply)
    threads: Optional[int] = None
    n_cpu_moe: Optional[int] = None
    cpu_moe: Optional[bool] = None
    kv_offload: Optional[str] = None  # "on" / "off"
    override_tensor: Optional[str] = None
    # Speculative decoding module (MTP / DSpark), explicit override
    drafter_path: Optional[Path] = None
    drafter_type: Optional[str] = None  # "draft-mtp" / "draft-dspark"
    batch_size: int = 512
    ubatch_size: int = 512
    
    # KV Cache settings
    cache_type_k: str = "f16"  # f16, q8_0, q4_0
    cache_type_v: str = "f16"  # f16, q8_0, q4_0
    flash_attn: str = "auto"  # "on", "off", "auto"
    defrag_thold: float = 0.0  # 0.0 = disabled
    
    # Sampling (None = not set; llama.cpp defaults apply)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    
    @property
    def server_url(self) -> str:
        """Get server URL."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_multimodal(self) -> bool:
        """Check if model supports multimodal."""
        return self.model.is_multimodal
    
    def to_server_args(self) -> list[str]:
        """Convert to llama-server command line arguments."""
        args = [
            "-m", str(self.model.file_path),
            "--host", self.host,
            "--port", str(self.port),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
        ]
        
        # Context: 0 = load from model (omit -c, llama-server default)
        if self.context_length > 0:
            args.extend(["-c", str(self.context_length)])
        
        # GPU layers: only add -ngl if not using fit auto mode
        if self.fit != "on":
            args.extend(["-ngl", str(self.gpu_layers)])
        
        # Add fit parameter
        if self.fit != "on":
            args.extend(["--fit", self.fit])
        
        # Add mmproj file if available (for multimodal support)
        # Prefer explicit config, fall back to scanner-detected model mmproj
        mmproj = self.mmproj_path or self.model.mmproj_path
        if mmproj and mmproj.exists():
            args.extend(["--mmproj", str(mmproj)])
        
        if self.threads:
            args.extend(["-t", str(self.threads)])
        
        # KV Cache settings
        if self.cache_type_k != "f16":
            args.extend(["--cache-type-k", self.cache_type_k])
        if self.cache_type_v != "f16":
            args.extend(["--cache-type-v", self.cache_type_v])
        if self.flash_attn != "auto":
            args.extend(["--flash-attn", self.flash_attn])
        if self.defrag_thold > 0:
            args.extend(["--defrag-thold", str(self.defrag_thold)])
        
        # Speculative decoding module (MTP / DSpark drafter);
        # explicit config wins, else scanner-detected model drafter
        drafter = self.drafter_path or self.model.drafter_path
        if drafter and drafter.exists() and (
            self.drafter_type or self.model.drafter_type
        ):
            args.extend([
                "--spec-type", self.drafter_type or self.model.drafter_type,
                "--spec-draft-model", str(drafter),
            ])
        
        # MoE / placement advanced settings (verified present in llama.cpp help)
        if self.n_cpu_moe is not None:
            args.extend(["--n-cpu-moe", str(self.n_cpu_moe)])
        if self.cpu_moe is True:
            args.append("--cpu-moe")
        if self.kv_offload == "off":
            args.append("--no-kv-offload")
        elif self.kv_offload == "on":
            args.append("--kv-offload")
        if self.override_tensor:
            args.extend(["--override-tensor", self.override_tensor])
        
        # Sampling parameters (None = not set, llama.cpp defaults apply)
        if self.temperature is not None:
            args.extend(["--temp", str(self.temperature)])
        if self.top_k is not None:
            args.extend(["--top-k", str(self.top_k)])
        if self.top_p is not None:
            args.extend(["--top-p", str(self.top_p)])
        if self.repeat_penalty is not None:
            args.extend(["--repeat-penalty", str(self.repeat_penalty)])
        
        return args
    
    def to_cli_args(self) -> list[str]:
        """Convert to llama-cli command line arguments."""
        args = [
            "-m", str(self.model.file_path),
        ]
        
        # Context: 0 = load from model (omit -c, llama-cli default)
        if self.context_length > 0:
            args.extend(["-c", str(self.context_length)])
        
        # GPU layers: only add -ngl if not using fit auto mode
        if self.fit != "on":
            args.extend(["-ngl", str(self.gpu_layers)])
        
        # Add fit parameter
        if self.fit != "on":
            args.extend(["--fit", self.fit])
        
        # Add mmproj file if available (for multimodal support)
        # Prefer explicit config, fall back to scanner-detected model mmproj
        mmproj = self.mmproj_path or self.model.mmproj_path
        if mmproj and mmproj.exists():
            args.extend(["--mmproj", str(mmproj)])
        
        if self.threads:
            args.extend(["-t", str(self.threads)])
        
        # KV Cache settings
        if self.cache_type_k != "f16":
            args.extend(["--cache-type-k", self.cache_type_k])
        if self.cache_type_v != "f16":
            args.extend(["--cache-type-v", self.cache_type_v])
        if self.flash_attn != "auto":
            args.extend(["--flash-attn", self.flash_attn])
        
        # Speculative decoding module (MTP / DSpark drafter);
        # explicit config wins, else scanner-detected model drafter
        drafter = self.drafter_path or self.model.drafter_path
        if drafter and drafter.exists() and (
            self.drafter_type or self.model.drafter_type
        ):
            args.extend([
                "--spec-type", self.drafter_type or self.model.drafter_type,
                "--spec-draft-model", str(drafter),
            ])
        
        # MoE / placement advanced settings (verified present in llama.cpp help)
        if self.n_cpu_moe is not None:
            args.extend(["--n-cpu-moe", str(self.n_cpu_moe)])
        if self.cpu_moe is True:
            args.append("--cpu-moe")
        if self.kv_offload == "off":
            args.append("--no-kv-offload")
        elif self.kv_offload == "on":
            args.append("--kv-offload")
        if self.override_tensor:
            args.extend(["--override-tensor", self.override_tensor])
        
        # Sampling parameters (None = not set, llama.cpp defaults apply)
        if self.temperature is not None:
            args.extend(["--temp", str(self.temperature)])
        if self.top_k is not None:
            args.extend(["--top-k", str(self.top_k)])
        if self.top_p is not None:
            args.extend(["--top-p", str(self.top_p)])
        if self.repeat_penalty is not None:
            args.extend(["--repeat-penalty", str(self.repeat_penalty)])
        
        return args


def select_model_interactive(
    models: list[ModelMetadata],
) -> Optional[ModelMetadata]:
    """Interactively select a model from a list.
    
    Args:
        models: List of available models
        
    Returns:
        Selected model or None if cancelled
    """
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import IntPrompt, Confirm
    
    console = Console()
    
    if not models:
        console.print("[red]没有找到可用的模型[/red]")
        return None
    
    # Separate multimodal and text-only
    multimodal = [m for m in models if m.is_multimodal]
    text_only = [m for m in models if not m.is_multimodal]
    
    # Display models (quick scan: only file-level info, no GGUF reads)
    table = Table(title="可用模型", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("名称", style="green")
    table.add_column("类型", style="magenta")
    table.add_column("大小", style="magenta")
    
    all_models = []
    idx = 1
    
    for model in multimodal:
        table.add_row(
            str(idx),
            model.display_name,
            "[cyan]🖼️ 多模态[/cyan]" if model.is_multimodal else "[dim]📝 文本[/dim]",
            model.file_size_display,
        )
        all_models.append(model)
        idx += 1
    
    for model in text_only:
        table.add_row(
            str(idx),
            model.display_name,
            "[cyan]🖼️ 多模态[/cyan]" if model.is_multimodal else "[dim]📝 文本[/dim]",
            model.file_size_display,
        )
        all_models.append(model)
        idx += 1
    
    console.print(table)
    console.print()
    
    # Prompt for selection
    while True:
        try:
            choice = IntPrompt.ask(
                "选择模型编号",
                default=1,
                choices=[str(i) for i in range(1, len(all_models) + 1)],
                show_choices=False,
            )
            if 1 <= choice <= len(all_models):
                return all_models[choice - 1]
            console.print("[red]无效的选择，请重试[/red]")
        except (KeyboardInterrupt, EOFError):
            return None


def configure_multimodal_interactive(model: ModelMetadata) -> Optional[Path]:
    """Ask about multimodal settings and select projector.
    
    Args:
        model: The selected model
        
    Returns:
        Selected projector path or None if skipped
    """
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Confirm, IntPrompt
    from pathlib import Path
    
    console = Console()
    
    # Scan for mmproj files in model directory
    model_dir = model.file_path.parent
    mmproj_files = list(model_dir.glob("mmproj-*.gguf"))
    
    if not mmproj_files:
        return None
    
    console.print()
    if not Confirm.ask("要加载投影器吗？", default=False):
        return None
    
    # Show projector list
    console.print()
    table = Table(title="可用投影器")
    table.add_column("#", style="cyan", width=4)
    table.add_column("文件名", style="green")
    table.add_column("大小", style="magenta")
    
    for i, f in enumerate(mmproj_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        table.add_row(str(i), f.name, f"{size_mb:.0f}MB")
    
    console.print(table)
    
    # Select projector
    while True:
        try:
            choice = IntPrompt.ask(
                "选择投影器编号",
                default=1,
                choices=[str(i) for i in range(1, len(mmproj_files) + 1)],
                show_choices=False,
            )
            if 1 <= choice <= len(mmproj_files):
                return mmproj_files[choice - 1]
        except (KeyboardInterrupt, EOFError):
            return None


def configure_drafter_interactive(
    model: ModelMetadata,
) -> Optional[tuple[Path, str]]:
    """Ask whether to load an MTP / DSpark speculative-decoding module.

    Companion files in the model's directory:
      - ``mtp-*.gguf``          -> "draft-mtp"
      - ``*dspark*.gguf``       -> "draft-dspark"

    Returns:
        (module_path, spec_type) or None if skipped
    """
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Confirm, IntPrompt

    console = Console()

    # Find candidate modules in the same directory
    model_dir = model.file_path.parent
    candidates: list[tuple[Path, str]] = []
    for f in sorted(model_dir.glob("mtp-*.gguf")):
        candidates.append((f, "draft-mtp"))
    dspark_files = list(model_dir.glob("*dspark*.gguf"))
    if not dspark_files:
        dspark_files = list(model_dir.glob("*DSpark*.gguf"))
    for f in sorted(dspark_files):
        candidates.append((f, "draft-dspark"))

    if not candidates:
        return None

    console.print()
    if not Confirm.ask("要加载推测解码模块（MTP/DSpark）吗？", default=False):
        return None

    # Show module list
    console.print()
    table = Table(title="可用推测解码模块")
    table.add_column("#", style="cyan", width=4)
    table.add_column("文件名", style="green")
    table.add_column("类型", style="yellow")
    table.add_column("大小", style="magenta")

    for i, (f, spec_type) in enumerate(candidates, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        table.add_row(str(i), f.name, spec_type, f"{size_mb:.0f}MB")

    console.print(table)

    # Select a module
    while True:
        try:
            choice = IntPrompt.ask(
                "选择模块编号",
                default=1,
                choices=[str(i) for i in range(1, len(candidates) + 1)],
                show_choices=False,
            )
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1]
        except (KeyboardInterrupt, EOFError):
            return None


def select_launch_mode_interactive() -> str:
    """Interactively select launch mode.
    
    Returns:
        "server" or "cli"
    """
    from rich.console import Console
    from rich.prompt import IntPrompt
    
    console = Console()
    console.print()
    console.print("[bold]选择启动模式:[/bold]")
    console.print("  [cyan]1[/cyan] - llama-server (HTTP API)")
    console.print("  [cyan]2[/cyan] - llama-cli (交互式推理)")
    
    while True:
        try:
            choice = IntPrompt.ask(
                "模式",
                default=1,
                choices=["1", "2"],
                show_choices=False,
            )
            return "server" if choice == 1 else "cli"
        except (KeyboardInterrupt, EOFError):
            return "server"


def configure_gpu_interactive(model_size_gb: float) -> tuple[int, str]:
    """Interactively configure GPU layers.
    
    Args:
        model_size_gb: Model file size in GB
        
    Returns:
        Tuple of (gpu_layers, fit_mode)
        fit_mode: "on" = auto, "off" = manual
    """
    from rich.console import Console
    from rich.prompt import IntPrompt
    
    console = Console()
    
    gpus = detect_gpus()
    if not gpus:
        console.print("[yellow]未检测到 GPU，将使用 CPU 模式[/yellow]")
        return 0, "off"
    
    console.print()
    console.print("[bold]GPU 信息:[/bold]")
    for gpu in gpus:
        console.print(f"  - {gpu.display_name}")
    
    total_free_gb = sum(g.free_memory_gb for g in gpus)
    console.print(f"  总可用显存: {total_free_gb:.1f}GB")
    
    console.print()
    console.print("[bold]GPU 层数分配:[/bold]")
    console.print("  [cyan]1[/cyan] - 自动分配 (推荐, 使用 --fit)")
    console.print("  [cyan]2[/cyan] - 全部 GPU (-1)")
    console.print("  [cyan]3[/cyan] - 纯 CPU (0)")
    console.print("  [cyan]4[/cyan] - 自定义层数")
    
    while True:
        try:
            choice = IntPrompt.ask("选择", default=1)
            if choice == 1:
                return -1, "on"  # fit mode
            elif choice == 2:
                return -1, "off"
            elif choice == 3:
                return 0, "off"
            elif choice == 4:
                console.print("[dim]输入 GPU 层数 (0=纯CPU, -1=全部)[/dim]")
                layers = IntPrompt.ask("GPU 层数", default=-1)
                return layers, "off"
        except (KeyboardInterrupt, EOFError):
            return -1, "on"


def configure_context_interactive(default: int = 4096) -> int:
    """Interactively configure context length.
    
    Args:
        default: Default context length
        
    Returns:
        Context length (0 = load from model, omit -c)
    """
    from rich.console import Console
    from rich.prompt import IntPrompt
    
    console = Console()
    console.print()
    console.print(f"[dim]上下文长度 (默认: {default}, 0=从模型加载)[/dim]")
    
    while True:
        try:
            ctx = IntPrompt.ask(
                "上下文长度",
                default=default,
            )
            if ctx == 0:
                return 0  # Omit -c, let llama.cpp load from model
            if ctx > 0:
                return ctx
            console.print("[red]请输入正数或 0[/red]")
        except (KeyboardInterrupt, EOFError):
            return default


def configure_server_interactive(default_port: int = 8080) -> tuple[str, int]:
    """Interactively configure server settings.
    
    Args:
        default_port: Default port number
        
    Returns:
        Tuple of (host, port)
    """
    from rich.console import Console
    from rich.prompt import IntPrompt, Prompt
    
    console = Console()
    console.print()
    
    host = Prompt.ask(
        "监听地址",
        default="127.0.0.1",
    )
    
    port = IntPrompt.ask(
        "端口号",
        default=default_port,
    )
    
    return host, port


def configure_cache_interactive() -> tuple[str, str, bool]:
    """Interactively configure KV cache settings.
    
    Returns:
        Tuple of (cache_type_k, cache_type_v, flash_attn)
    """
    from rich.console import Console
    from rich.prompt import IntPrompt, Confirm
    
    console = Console()
    console.print()
    console.print("[bold]KV 缓存设置[/bold]")
    
    # Cache type K
    console.print()
    console.print("[dim]K 缓存量化类型:[/dim]")
    console.print("  [cyan]1[/cyan] - f16 (默认, 精度最高)")
    console.print("  [cyan]2[/cyan] - q8_0 (推荐, 节省50%显存)")
    console.print("  [cyan]3[/cyan] - q4_0 (激进, 节省75%显存)")
    
    cache_types = {"1": "f16", "2": "q8_0", "3": "q4_0"}
    
    while True:
        try:
            choice = IntPrompt.ask("K 缓存类型", default=1)
            if 1 <= choice <= 3:
                cache_type_k = cache_types[str(choice)]
                break
        except (KeyboardInterrupt, EOFError):
            cache_type_k = "f16"
            break
    
    # Cache type V
    console.print()
    console.print("[dim]V 缓存量化类型:[/dim]")
    console.print("  [cyan]1[/cyan] - f16 (默认)")
    console.print("  [cyan]2[/cyan] - q8_0 (推荐)")
    console.print("  [cyan]3[/cyan] - q4_0 (激进)")
    
    while True:
        try:
            choice = IntPrompt.ask("V 缓存类型", default=1)
            if 1 <= choice <= 3:
                cache_type_v = cache_types[str(choice)]
                break
        except (KeyboardInterrupt, EOFError):
            cache_type_v = "f16"
            break
    
    # Flash Attention
    console.print()
    console.print("[dim]Flash Attention:[/dim]")
    console.print("  [cyan]1[/cyan] - auto (默认, 自动决定)")
    console.print("  [cyan]2[/cyan] - on (强制启用)")
    console.print("  [cyan]3[/cyan] - off (强制禁用)")
    
    flash_attn_options = {"1": "auto", "2": "on", "3": "off"}
    
    while True:
        try:
            choice = IntPrompt.ask("Flash Attention", default=1)
            if 1 <= choice <= 3:
                flash_attn = flash_attn_options[str(choice)]
                break
        except (KeyboardInterrupt, EOFError):
            flash_attn = "auto"
            break
    
    return cache_type_k, cache_type_v, flash_attn


def configure_batch_interactive() -> tuple[int, int]:
    """Interactively configure batch settings.
    
    Returns:
        Tuple of (batch_size, ubatch_size)
    """
    from rich.console import Console
    from rich.prompt import IntPrompt
    
    console = Console()
    console.print()
    console.print("[bold]批处理设置[/bold]")
    console.print("[dim]提示处理批大小 (默认: 512)[/dim]")
    
    while True:
        try:
            batch_size = IntPrompt.ask("批处理大小", default=512)
            if batch_size > 0:
                break
        except (KeyboardInterrupt, EOFError):
            batch_size = 512
            break
    
    console.print("[dim]推理微批大小 (默认: 512)[/dim]")
    
    while True:
        try:
            ubatch_size = IntPrompt.ask("微批大小", default=512)
            if ubatch_size > 0:
                break
        except (KeyboardInterrupt, EOFError):
            ubatch_size = 512
            break
    
    return batch_size, ubatch_size


def select_config_mode_interactive() -> str:
    """Interactively select configuration mode.
    
    Returns:
        "manual" or "profile"
    """
    from rich.console import Console
    from rich.prompt import IntPrompt
    
    console = Console()
    console.print()
    console.print("[bold]配置方式[/bold]")
    console.print("  [cyan]1[/cyan] - 手动配置 - 逐步设置参数")
    console.print("  [cyan]2[/cyan] - 加载配置 - 从文件加载预设")
    
    while True:
        try:
            choice = IntPrompt.ask("配置方式", default=1)
            if choice == 1:
                return "manual"
            elif choice == 2:
                return "profile"
        except (KeyboardInterrupt, EOFError):
            return "manual"


def select_profile_interactive() -> Optional["ConfigProfile"]:
    """Interactively select a configuration profile.
    
    Returns:
        Selected ConfigProfile or None if cancelled
    """
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import IntPrompt
    from .config import list_profiles, get_profiles_dir
    
    console = Console()
    profiles = list_profiles()
    
    if not profiles:
        console.print("[yellow]未找到配置文件[/yellow]")
        console.print(f"[dim]配置目录: {get_profiles_dir()}[/dim]")
        return None
    
    # Show profiles table
    console.print()
    table = Table(title=f"可用配置 ({get_profiles_dir()})")
    table.add_column("#", style="cyan", width=4)
    table.add_column("文件名", style="green")
    
    for i, profile in enumerate(profiles, 1):
        table.add_row(str(i), f"{profile.display_name}.json")
    
    console.print(table)
    
    # Select profile
    while True:
        try:
            choice = IntPrompt.ask(
                "选择配置编号",
                default=1,
                choices=[str(i) for i in range(1, len(profiles) + 1)],
                show_choices=False,
            )
            if 1 <= choice <= len(profiles):
                return profiles[choice - 1]
        except (KeyboardInterrupt, EOFError):
            return None


# Advanced configuration items: (key, param display, hint, value type)
ADVANCED_ITEMS: list[tuple[str, str, str, str]] = [
    ("threads", "-t 线程数", "CPU 线程数，如 8", "int"),
    ("n_cpu_moe", "--n-cpu-moe N", "MoE 前 N 层专家留 CPU（仅 MoE 模型）", "int"),
    ("cpu_moe", "--cpu-moe", "所有 MoE 专家留 CPU（仅 MoE 模型）", "bool"),
    ("kv_offload", "--kv-offload", "KV 缓存 offload：on/off，默认 on", "text"),
    ("override_tensor", "--override-tensor", '张量级放置，如 "ffn_down_exps=CPU"', "text"),
    # Sampling (None = not set, llama.cpp defaults apply)
    ("temperature", "--temp", "采样温度，如 0.5", "float"),
    ("top_p", "--top-p", "核采样阈值，如 0.9", "float"),
    ("top_k", "--top-k", "候选 token 数，如 40", "int"),
    ("repeat_penalty", "--repeat-penalty", "重复惩罚，如 1.3", "float"),
]

# Keys that only apply to MoE models
MOE_KEYS = {"n_cpu_moe", "cpu_moe"}


def _is_moe_model(model: ModelMetadata) -> bool:
    """Check if the model architecture is MoE-based."""
    return bool(model.architecture and "moe" in model.architecture.lower())


def _format_advanced_value(key: str, value: Any) -> str:
    """Format an advanced value for display."""
    if value is None:
        return "(未设置)"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def show_advanced_config_interactive(
    model: ModelMetadata,
    values: dict[str, Any],
) -> str:
    """Advanced-config UI.

    Shows the advanced parameter list and lets the user configure items
    one by one (choice 1) or jump straight into command editing
    (choice 2). Values are stored in ``values`` but are not wired into
    the launch command yet (UI stage).

    Args:
        model: selected model (drives MoE-only item visibility)
        values: mutable dict holding current advanced values (in place)

    Returns:
        "continue" to proceed to the normal confirm step,
        or "edit" to jump straight into command editing
    """
    from rich.console import Console
    from rich.prompt import IntPrompt, Prompt
    from rich.table import Table

    console = Console()
    moe = _is_moe_model(model)

    def visible_items() -> list[tuple[str, str, str, str]]:
        out = []
        for item in ADVANCED_ITEMS:
            if item[0] in MOE_KEYS and not moe:
                continue
            out.append(item)
        return out

    def show_list(items: list[tuple[str, str, str, str]]) -> None:
        table = Table(title="高级配置", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("参数", style="green")
        table.add_column("当前值", style="yellow")
        table.add_column("说明", style="dim")
        for i, (key, param, hint, _t) in enumerate(items, 1):
            table.add_row(str(i), param, _format_advanced_value(key, values.get(key)), hint)
        console.print(table)

    while True:
        items = visible_items()
        console.print()
        console.print("[bold]高级配置[/bold]")
        console.print("[dim]（已设置的参数将传入启动命令，未设置则由 llama.cpp 默认）[/dim]")
        show_list(items)
        console.print()
        console.print("操作:")
        console.print("  [cyan]1[/cyan] - 开始配置（输入编号逐项设置，配置一项返回此列表）")
        console.print("  [cyan]2[/cyan] - 进入启动命令编辑")
        try:
            choice = IntPrompt.ask("操作", default=1, choices=["1", "2"], show_choices=False)
        except (KeyboardInterrupt, EOFError):
            return "continue"

        if choice == 2:
            return "edit"

        # Per-item configuration loop
        while True:
            console.print()
            show_list(items)
            try:
                num = IntPrompt.ask("输入编号配置 (0=返回)", default=0)
            except (KeyboardInterrupt, EOFError):
                break
            if num == 0:
                break
            if not 1 <= num <= len(items):
                console.print("[red]无效编号[/red]")
                continue
            key, param, hint, typ = items[num - 1]
            console.print()
            console.print(f"[bold]{param}[/bold]")
            console.print(f"[dim]{hint}[/dim]")
            console.print(f"[dim]当前值: {_format_advanced_value(key, values.get(key))}[/dim]")
            try:
                if typ == "bool":
                    raw = Prompt.ask(
                        "值 (on/off, 回车=不变)",
                        default="" if values.get(key) is None else _format_advanced_value(key, values.get(key)),
                    )
                    if raw.strip().lower() in ("on", "true", "1", "y"):
                        values[key] = True
                        console.print(f"[green]✓ 已设置 {param} = on（将传入 --cpu-moe 等）[/green]")
                    elif raw.strip().lower() in ("off", "false", "0", "n"):
                        values[key] = False
                        console.print(f"[green]✓ 已设置 {param} = off[/green]")
                    else:
                        console.print("[dim]未修改[/dim]")
                elif typ == "int":
                    raw = Prompt.ask(
                        "值 (回车=不变)",
                        default="" if values.get(key) is None else str(values[key]),
                    )
                    if raw.strip():
                        try:
                            values[key] = int(raw.strip())
                            console.print(f"[green]✓ 已设置 {param} = {values[key]}[/green]")
                        except ValueError:
                            console.print("[red]请输入数字[/red]")
                    else:
                        console.print("[dim]未修改[/dim]")
                elif typ == "float":
                    raw = Prompt.ask(
                        "值 (回车=不变)",
                        default="" if values.get(key) is None else str(values[key]),
                    )
                    if raw.strip():
                        try:
                            values[key] = float(raw.strip())
                            console.print(f"[green]✓ 已设置 {param} = {values[key]}[/green]")
                        except ValueError:
                            console.print("[red]请输入数字[/red]")
                    else:
                        console.print("[dim]未修改[/dim]")
                else:
                    raw = Prompt.ask(
                        "值 (回车=不变)",
                        default="" if values.get(key) is None else str(values[key]),
                    )
                    if raw.strip():
                        values[key] = raw.strip()
                        console.print(f"[green]✓ 已设置 {param} = {values[key]}[/green]")
                    else:
                        console.print("[dim]未修改[/dim]")
            except (KeyboardInterrupt, EOFError):
                continue


def show_profile_detail(profile: "ConfigProfile") -> bool:
    """Show profile details (base + advanced sections) and ask to continue.
    
    Args:
        profile: The selected profile
        
    Returns:
        True to continue, False to reselect
    """
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Confirm
    
    console = Console()
    
    console.print()
    console.print(f"[bold]配置详情: {profile.display_name}.json[/bold]")
    
    for section_name, lines in profile.get_summary_sections():
        console.print()
        console.print(f"[bold]{section_name}[/bold]")
        table = Table(show_header=False)
        table.add_column("参数", style="cyan")
        table.add_column("值", style="green")
        for key, value in lines:
            table.add_row(key, value)
        console.print(table)
    
    while True:
        try:
            choice = Confirm.ask("使用此配置?", default=True)
            return choice
        except (KeyboardInterrupt, EOFError):
            return True


def load_profile_config(
    profile: "ConfigProfile",
    model: "ModelMetadata",
    mmproj_path: Optional[Path] = None,
    drafter: Optional[tuple[Path, str]] = None,
    default_port: int = 8080,
) -> LaunchConfig:
    """Create LaunchConfig from a profile.
    
    Args:
        profile: Configuration profile
        model: Selected model
        mmproj_path: Optional projector path
        drafter: Optional (module_path, spec_type) pair
        default_port: Default server port
        
    Returns:
        LaunchConfig with profile settings
    """
    # Still ask for mode and server settings (manual each time)
    mode = select_launch_mode_interactive()
    
    host = "127.0.0.1"
    port = default_port
    if mode == "server":
        from rich.console import Console
        from rich.prompt import Prompt, IntPrompt
        console = Console()
        console.print()
        host = Prompt.ask("监听地址", default="127.0.0.1")
        port = IntPrompt.ask("端口号", default=default_port)
    
    drafter_path = drafter[0] if drafter else None
    drafter_type = drafter[1] if drafter else None
    
    return LaunchConfig(
        model=model,
        mmproj_path=mmproj_path,
        drafter_path=drafter_path,
        drafter_type=drafter_type,
        mode=mode,
        gpu_layers=profile.gpu_layers,
        fit=profile.fit,
        context_length=profile.context_length,
        host=host,
        port=port,
        cache_type_k=profile.cache_type_k,
        cache_type_v=profile.cache_type_v,
        flash_attn=profile.flash_attn,
        defrag_thold=profile.defrag_thold,
        threads=profile.threads,
        batch_size=profile.batch_size,
        ubatch_size=profile.ubatch_size,
        n_cpu_moe=profile.n_cpu_moe,
        cpu_moe=profile.cpu_moe,
        kv_offload=profile.kv_offload,
        override_tensor=profile.override_tensor,
    )


def show_config_summary(config: LaunchConfig) -> None:
    """Display configuration summary before launch.
    
    Args:
        config: Launch configuration
    """
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    table = Table(title="启动配置", show_header=False)
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("模型", config.model.display_name)
    table.add_row("架构", config.model.architecture_display)
    table.add_row("量化", config.model.quantization_type or "Unknown")
    table.add_row("大小", config.model.file_size_display)
    
    if config.is_multimodal:
        table.add_row("多模态", "✅ 是")
    
    if config.model.drafter_type:
        table.add_row("推测解码", f"✅ {config.model.drafter_type}")
    
    table.add_row("模式", "server" if config.mode == "server" else "cli")
    
    # GPU layers display
    if config.fit == "on":
        table.add_row("GPU 层数", "自动 (fit)")
    elif config.gpu_layers == -1:
        table.add_row("GPU 层数", "全部")
    elif config.gpu_layers == 0:
        table.add_row("GPU 层数", "纯CPU")
    else:
        table.add_row("GPU 层数", str(config.gpu_layers))
    
    table.add_row("上下文长度", str(config.context_length))
    
    # KV Cache settings
    table.add_row("K 缓存量化", config.cache_type_k)
    table.add_row("V 缓存量化", config.cache_type_v)
    
    # Flash Attention
    flash_display = {"on": "启用", "off": "禁用", "auto": "自动"}
    table.add_row("Flash Attn", flash_display.get(config.flash_attn, config.flash_attn))
    
    if config.mode == "server":
        table.add_row("地址", f"{config.host}:{config.port}")
    
    console.print(table)
    
    # Upstream caveat: --fit VRAM estimation does not account for mmproj
    if config.is_multimodal and config.fit == "on":
        console.print(
            "[yellow]提示: --fit 的显存估算不含 mmproj 投影器, "
            "多模态模型建议手动指定 -ngl 以避免 OOM[/yellow]"
        )


def build_launch_command(config: LaunchConfig) -> str:
    """Build the launch command string from config.

    Uses the same argument construction as the actual launch path
    (to_server_args / to_cli_args), so the displayed/edited command
    always matches what would really be executed.
    """
    exe = "llama-server.exe" if config.mode == "server" else "llama-cli.exe"
    args = config.to_server_args() if config.mode == "server" else config.to_cli_args()
    return subprocess.list2cmdline([exe] + args)


def show_command_and_confirm(config: LaunchConfig) -> tuple[bool, Optional[str]]:
    """Show launch command and ask user to launch or edit.
    
    Returns:
        Tuple of (should_launch, edited_command)
        If edited_command is not None, use it instead of building from config
    """
    from rich.console import Console
    from rich.prompt import IntPrompt
    
    console = Console()
    
    command = build_launch_command(config)
    
    console.print()
    console.print("[bold]启动命令:[/bold]")
    console.print(f"[dim]{command}[/dim]")
    console.print()
    console.print("  [cyan]1[/cyan] - 启动")
    console.print("  [cyan]2[/cyan] - 编辑命令")
    
    while True:
        try:
            choice = IntPrompt.ask("选择", default=1)
            if choice == 1:
                return True, None
            elif choice == 2:
                # Print the command; the user may copy it to edit elsewhere,
                # or press Enter to run the current command as-is.
                console.print()
                console.print("[bold]启动命令:[/bold]")
                console.print(f"[dim]{command}[/dim]")
                console.print()
                console.print("[dim]回车 = 执行当前命令（如需修改，请复制到终端自行编辑执行）[/dim]")
                try:
                    input("")
                except (KeyboardInterrupt, EOFError):
                    return False, None
                return True, None
        except (KeyboardInterrupt, EOFError):
            return False, None


def prompt_for_config(
    models: list[ModelMetadata],
    default_port: int = 8080,
) -> Optional[tuple[LaunchConfig, Optional[str]]]:
    """Full interactive configuration flow.
    
    Args:
        models: List of available models
        default_port: Default server port
        
    Returns:
        Tuple of (LaunchConfig, edited_command) or None if cancelled
        If edited_command is not None, use it instead of config
    """
    from rich.console import Console
    
    console = Console()
    
    # Select model
    model = select_model_interactive(models)
    if not model:
        return None
    
    # Deep-scan the selected model for full capability info
    # (quick scan only listed the file; read GGUF metadata now)
    from .scanner import ModelScanner
    console.print(f"[dim]正在读取模型信息: {model.display_name} …[/dim]")
    deep = ModelScanner(model_dirs=[model.file_path.parent]).scan_deep(model)
    if deep:
        model = deep
    
    # Show one-line detail, then proceed straight into multimodal scan
    console.print(f"[bold]📋 {model.model_summary}[/bold]")
    
    # Configure multimodal projector
    mmproj_path = configure_multimodal_interactive(model)
    
    # Configure MTP / DSpark speculative-decoding module
    drafter = configure_drafter_interactive(model)
    
    # Select configuration mode
    config_mode = select_config_mode_interactive()
    
    if config_mode == "profile":
        # Load from profile
        while True:
            profile = select_profile_interactive()
            if not profile:
                console.print("[yellow]未选择配置，返回手动配置[/yellow]")
                config_mode = "manual"
                break
            
            # Show profile detail and confirm
            if show_profile_detail(profile):
                config = load_profile_config(profile, model, mmproj_path, drafter, default_port)
                # Show command and confirm
                should_launch, edited_cmd = show_command_and_confirm(config)
                if should_launch:
                    return config, edited_cmd
                # If not confirmed, loop back
            # If not confirmed, loop back to selection
    
    if config_mode == "manual":
        # Manual configuration - new order
        
        # Ask whether to enable advanced configuration
        console.print()
        advanced_enabled = False
        try:
            from rich.prompt import Confirm
            advanced_enabled = Confirm.ask("是否启用高级配置？", default=False)
        except (KeyboardInterrupt, EOFError):
            advanced_enabled = False
        
        # 1. Context length (cap the default to avoid 1M-ctx surprises)
        model_ctx = model.context_length or 4096
        context_length = configure_context_interactive(min(model_ctx, 16384))
        
        # 2. GPU layers
        gpu_layers, fit = configure_gpu_interactive(model.file_size_gb)
        
        # 3. KV cache
        cache_type_k, cache_type_v, flash_attn = configure_cache_interactive()
        
        # 4. Batch settings
        batch_size, ubatch_size = configure_batch_interactive()
        
        # 5. Launch mode
        mode = select_launch_mode_interactive()
        
        # 6. Server settings (if server mode)
        host = "127.0.0.1"
        port = default_port
        if mode == "server":
            host, port = configure_server_interactive(default_port)
        
        # Build config
        config = LaunchConfig(
            model=model,
            mode=mode,
            gpu_layers=gpu_layers,
            fit=fit,
            context_length=context_length,
            host=host,
            port=port,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            flash_attn=flash_attn,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            mmproj_path=mmproj_path,
            drafter_path=drafter[0] if drafter else None,
            drafter_type=drafter[1] if drafter else None,
        )
        
        # Apply advanced configuration (set values are wired into launch args)
        if advanced_enabled:
            advanced_values: dict[str, Any] = {}
            action = show_advanced_config_interactive(model, advanced_values)
            config.threads = advanced_values.get("threads", config.threads)
            config.n_cpu_moe = advanced_values.get("n_cpu_moe", config.n_cpu_moe)
            config.cpu_moe = advanced_values.get("cpu_moe", config.cpu_moe)
            config.kv_offload = advanced_values.get("kv_offload", config.kv_offload)
            config.override_tensor = advanced_values.get("override_tensor", config.override_tensor)
            config.temperature = advanced_values.get("temperature", config.temperature)
            config.top_p = advanced_values.get("top_p", config.top_p)
            config.top_k = advanced_values.get("top_k", config.top_k)
            config.repeat_penalty = advanced_values.get("repeat_penalty", config.repeat_penalty)
            if action == "edit":
                # Print the command and let the user decide: run as-is or
                # copy it out to edit/run themselves. Tool keeps running.
                command = build_launch_command(config)
                console.print()
                console.print("[bold]启动命令:[/bold]")
                console.print(f"[dim]{command}[/dim]")
                console.print()
                console.print("[dim]回车 = 执行当前命令（如需修改，请复制到终端自行编辑执行）[/dim]")
                try:
                    input("")
                except (KeyboardInterrupt, EOFError):
                    return None
                return config, None
        
        # Show command and confirm
        should_launch, edited_cmd = show_command_and_confirm(config)
        if should_launch:
            return config, edited_cmd
    
    return None
