"""Main CLI entry point for llama-launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .launcher import check_prerequisites, launch_cli_with_config, launch_server_with_config
from .metadata import ModelMetadata
from .prompts import LaunchConfig, prompt_for_config, show_config_summary
from .scanner import ModelScanner, ScanResult


console = Console()


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold]llama-launcher[/bold] v{__version__}")
        ctx.exit()


def print_banner() -> None:
    """Print application banner."""
    console.print()
    console.print("[bold cyan]🦙 llama-launcher[/bold cyan]")
    console.print("[dim]llama.cpp CLI 启动器 - 支持多模态模型[/dim]")
    console.print()


def scan_models(
    models_dir: Optional[Path],
    recursive: bool,
    quick: bool = False,
) -> ScanResult:
    """Scan for available models.
    
    Args:
        models_dir: Directory to scan
        recursive: Scan subdirectories
        quick: Only list files (fast, no GGUF metadata reads)
    """
    console.print("[bold]快速扫描模型...[/bold]" if quick else "[bold]扫描模型...[/bold]")
    
    dirs = [models_dir] if models_dir else None
    scanner = ModelScanner(model_dirs=dirs, recursive=recursive)
    
    def progress(path: Path, current: int, total: int) -> None:
        console.print(f"  [{current}/{total}] {path.name}".ljust(60), end="\r")
    
    if quick:
        result = scanner.scan_quick(progress_callback=progress)
    else:
        result = scanner.scan(progress_callback=progress)
    
    # Clear progress line
    console.print(" " * 60, end="\r")
    
    if result.count > 0:
        console.print(f"[green]✓ 找到 {result.count} 个模型[/green]")
    else:
        console.print("[yellow]未找到模型[/yellow]")
    
    if result.errors:
        console.print(f"[yellow]⚠ {len(result.errors)} 个文件读取失败[/yellow]")
    
    return result


def list_models_table(models: list[ModelMetadata]) -> None:
    """Display models in a table."""
    table = Table(title="可用模型", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("名称", style="green")
    table.add_column("类型", style="magenta")
    table.add_column("架构", style="yellow")
    table.add_column("量化", style="blue")
    table.add_column("大小", style="magenta")
    table.add_column("层数", style="cyan")
    
    for i, model in enumerate(models, 1):
        table.add_row(
            str(i),
            model.display_name,
            "[cyan]🖼️ 多模态[/cyan]" if model.is_multimodal else "[dim]📝 文本[/dim]",
            model.architecture_display,
            model.quantization_type or "Unknown",
            model.file_size_display,
            str(model.block_count) if model.block_count else "-",
        )
    
    console.print(table)


@click.group()
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="显示版本信息",
)
@click.pass_context
def main(ctx: click.Context) -> None:
    """🦙 llama-launcher - llama.cpp CLI 启动器
    
    支持多模态模型的智能启动器，自动检测模型类型并配置参数。
    """
    ctx.ensure_object(dict)
    print_banner()


@main.command()
@click.option(
    "-d", "--models-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="模型目录路径",
)
@click.option(
    "-r", "--recursive/--no-recursive",
    default=True,
    help="递归扫描子目录",
)
@click.option(
    "-l", "--list-only",
    is_flag=True,
    help="仅列出模型，不启动",
)
@click.option(
    "-p", "--port",
    default=8080,
    type=int,
    help="服务器端口 (默认: 8080)",
)
@click.pass_context
def run(
    ctx: click.Context,
    models_dir: Optional[Path],
    recursive: bool,
    list_only: bool,
    port: int,
) -> None:
    """扫描模型并启动 llama.cpp"""
    # Check prerequisites
    prereqs = check_prerequisites()
    if not any(prereqs.values()):
        console.print("[red]错误: 未找到 llama-server 或 llama-cli[/red]")
        console.print("请安装 llama.cpp 或确保可执行文件在 PATH 中")
        console.print()
        console.print("[dim]安装方式:[/dim]")
        console.print("  git clone https://github.com/ggerganov/llama.cpp")
        console.print("  cd llama.cpp && mkdir build && cd build")
        console.print("  cmake .. && cmake --build . --config Release")
        ctx.exit(1)
    
    # Scan models (fast: file listing only, metadata read after selection)
    result = scan_models(models_dir, recursive, quick=True)
    
    if result.count == 0:
        console.print()
        console.print("[yellow]提示: 使用 -d 指定模型目录[/yellow]")
        ctx.exit(1)
    
    if list_only:
        console.print()
        list_models_table(result.models)
        ctx.exit(0)
    
    # Interactive configuration
    console.print()
    result = prompt_for_config(result.models, default_port=port)
    
    if not result:
        console.print("[yellow]已取消[/yellow]")
        ctx.exit(0)
    
    config, edited_cmd = result
    
    # Launch
    if edited_cmd:
        # Use edited command directly
        import subprocess
        console.print()
        console.print(f"[dim]{edited_cmd}[/dim]")
        console.print()
        try:
            subprocess.run(edited_cmd, shell=True)
        except KeyboardInterrupt:
            pass
        ctx.exit(0)
    elif config.mode == "server":
        exit_code = launch_server_with_config(config)
    else:
        exit_code = launch_cli_with_config(config)
    
    ctx.exit(exit_code)


@main.command()
@click.option(
    "-d", "--models-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="模型目录路径",
)
@click.option(
    "-r", "--recursive/--no-recursive",
    default=True,
    help="递归扫描子目录",
)
def list(models_dir: Optional[Path], recursive: bool) -> None:
    """列出可用模型"""
    result = scan_models(models_dir, recursive)
    
    if result.count == 0:
        console.print("[yellow]未找到模型[/yellow]")
        ctx = click.get_current_context()
        ctx.exit(1)
    
    console.print()
    list_models_table(result.models)
    
    # Summary
    multimodal_count = len(result.filter_multimodal())
    if multimodal_count > 0:
        console.print(f"\n[cyan]其中 {multimodal_count} 个支持多模态[/cyan]")


@main.command()
@click.argument(
    "model",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-ngl", "--gpu-layers",
    default=-1,
    type=int,
    help="GPU 层数 (默认: -1=全部)",
)
@click.option(
    "-c", "--context-size",
    default=4096,
    type=int,
    help="上下文长度 (默认: 4096)",
)
@click.option(
    "--port",
    default=8080,
    type=int,
    help="服务器端口 (默认: 8080)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="监听地址 (默认: 127.0.0.1)",
)
@click.option(
    "--cli-mode",
    is_flag=True,
    help="使用 CLI 模式而非 server",
)
def launch(
    model: Path,
    gpu_layers: int,
    context_size: int,
    port: int,
    host: str,
    cli_mode: bool,
) -> None:
    """直接启动指定模型"""
    from .metadata import read_model_metadata
    
    # Read model metadata
    console.print(f"[bold]加载模型: {model.name}[/bold]")
    metadata = read_model_metadata(model)
    
    if not metadata:
        metadata = ModelMetadata(
            file_path=model,
            file_size_bytes=model.stat().st_size,
        )
    
    # Auto-attach mmproj projector for multimodal models
    if not metadata.mmproj_path:
        from .scanner import ModelScanner
        mmproj = ModelScanner().find_mmproj_for_model(model)
        if mmproj:
            metadata.mmproj_path = mmproj
            console.print(f"[cyan]检测到投影器: {mmproj.name}[/cyan]")
    
    # Auto-attach MTP / DSpark drafter module
    if not metadata.drafter_path:
        from .scanner import ModelScanner
        drafter = ModelScanner().find_drafter_for_model(model)
        if drafter:
            metadata.drafter_path, metadata.drafter_type = drafter
            console.print(
                f"[cyan]检测到推测解码模块: {metadata.drafter_path.name} "
                f"({metadata.drafter_type})[/cyan]"
            )
    
    # Show model info
    if metadata.is_multimodal:
        console.print("[cyan]🖼️ 多模态模型[/cyan]")
    
    # Create config
    config = LaunchConfig(
        model=metadata,
        mode="cli" if cli_mode else "server",
        gpu_layers=gpu_layers,
        context_length=context_size,
        host=host,
        port=port,
    )
    
    show_config_summary(config)
    console.print()
    
    # Launch
    if config.mode == "server":
        exit_code = launch_server_with_config(config)
    else:
        exit_code = launch_cli_with_config(config)
    
    ctx = click.get_current_context()
    ctx.exit(exit_code)


@main.command()
def check() -> None:
    """检查环境依赖"""
    from .gpu import detect_gpus
    
    console.print("[bold]环境检查[/bold]")
    console.print()
    
    # Check binaries
    prereqs = check_prerequisites()
    console.print("[bold]可执行文件:[/bold]")
    for name, available in prereqs.items():
        status = "[green]✓[/green]" if available else "[red]✗[/red]"
        console.print(f"  {status} {name}")
    
    console.print()
    
    # Check GPUs
    console.print("[bold]GPU:[/bold]")
    gpus = detect_gpus()
    if gpus:
        for gpu in gpus:
            console.print(f"  [green]✓[/green] {gpu.display_name}")
    else:
        console.print("  [yellow]未检测到 GPU (将使用 CPU)[/yellow]")
    
    console.print()
    
    # Summary
    if all(prereqs.values()):
        console.print("[green]✓ 环境检查通过[/green]")
    else:
        console.print("[yellow]⚠ 部分依赖缺失[/yellow]")


if __name__ == "__main__":
    main()
