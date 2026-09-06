"""Launcher for llama.cpp processes."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text


# Common locations for llama.cpp executables
LLAMA_CPP_PATHS = [
    Path.home() / "llama.cpp" / "build" / "bin",
    Path.home() / "llama.cpp" / "bin",
    Path("/usr/local/bin"),
    Path("/usr/bin"),
    # Windows common locations
    Path("C:/llama.cpp/build/bin"),
    Path("D:/llama.cpp/build/bin"),
    Path("C:/Program Files/llama.cpp"),
    Path("D:/Program Files/llama.cpp"),
]


def _build_number(dir_name: str) -> int:
    """Extract the build number from a dir name like 'llama-b10662-bin-win-x64'."""
    m = re.search(r"llama-b(\d+)", dir_name)
    return int(m.group(1)) if m else 0


def _discover_llama_dirs() -> list[Path]:
    """Dynamically discover release build dirs, newest first.

    Matches directories like ``D:\\llama-b10662-bin-win-vulkan-x64``,
    so upgrading to a newer build (or any build placed on C:/ or D:/)
    is picked up automatically.
    """
    dirs: list[Path] = []
    for root in ("D:/", "C:/"):
        try:
            for p in Path(root).glob("llama-b*-bin-win-*"):
                if p.is_dir() and (p / "llama-server.exe").exists():
                    dirs.append(p)
        except OSError:
            pass
    # Newest build first
    dirs.sort(key=lambda p: _build_number(p.name), reverse=True)
    return dirs


def find_llama_binary(name: str) -> Optional[Path]:
    """Find a llama.cpp binary.
    
    Search order:
      1. ``LLAMA_CPP_DIR`` env var (set by start.bat to the build dir it found)
      2. system PATH
      3. common locations, then dynamically discovered build dirs
    
    Args:
        name: Binary name (e.g., "llama-server", "llama-cli")
        
    Returns:
        Path to binary if found, None otherwise
    """
    # 1. Explicit env override (start.bat already located the build dir)
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        for base_dir in (Path(env_dir), Path(env_dir) / "bin"):
            exe_path = base_dir / f"{name}.exe"
            if exe_path.exists():
                return exe_path
            plain = base_dir / name
            if plain.exists():
                return plain
    
    # 2. System PATH
    found = shutil.which(name)
    if found:
        return Path(found)
    
    # 3. Common + dynamically discovered locations
    for base_dir in LLAMA_CPP_PATHS + _discover_llama_dirs():
        binary_path = base_dir / name
        if binary_path.exists():
            return binary_path
        
        # Try with .exe suffix on Windows
        if sys.platform == "win32":
            exe_path = base_dir / f"{name}.exe"
            if exe_path.exists():
                return exe_path
    
    return None


def get_llama_server_path() -> Optional[Path]:
    """Find llama-server binary."""
    return find_llama_binary("llama-server")


def get_llama_cli_path() -> Optional[Path]:
    """Find llama-cli binary."""
    return find_llama_binary("llama-cli")


class ProcessManager:
    """Manage llama.cpp processes."""
    
    def __init__(self):
        """Initialize process manager."""
        self.console = Console()
        self._process: Optional[subprocess.Popen] = None
    
    def launch_server(
        self,
        args: list[str],
        binary_path: Optional[Path] = None,
    ) -> subprocess.Popen:
        """Launch llama-server.
        
        Args:
            args: Command line arguments
            binary_path: Optional path to binary
            
        Returns:
            Popen process object
        """
        binary = binary_path or get_llama_server_path()
        if not binary:
            self.console.print("[red]错误: 找不到 llama-server[/red]")
            self.console.print("请确保 llama-server 在 PATH 中或在常见位置")
            raise FileNotFoundError("llama-server not found")
        
        cmd = [str(binary)] + args
        self.console.print(f"[dim]命令: {' '.join(cmd)}[/dim]")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            return self._process
        except Exception as e:
            self.console.print(f"[red]启动失败: {e}[/red]")
            raise
    
    def launch_cli(
        self,
        args: list[str],
        binary_path: Optional[Path] = None,
    ) -> subprocess.Popen:
        """Launch llama-cli.
        
        Args:
            args: Command line arguments
            binary_path: Optional path to binary
            
        Returns:
            Popen process object
        """
        binary = binary_path or get_llama_cli_path()
        if not binary:
            self.console.print("[red]错误: 找不到 llama-cli[/red]")
            self.console.print("请确保 llama-cli 在 PATH 中或在常见位置")
            raise FileNotFoundError("llama-cli not found")
        
        cmd = [str(binary)] + args
        self.console.print(f"[dim]命令: {' '.join(cmd)}[/dim]")
        
        try:
            # For CLI, inherit stdio for interactive use
            self._process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                stdin=sys.stdin,
            )
            return self._process
        except Exception as e:
            self.console.print(f"[red]启动失败: {e}[/red]")
            raise
    
    def wait_for_process(self) -> int:
        """Wait for process to complete.
        
        Returns:
            Exit code
        """
        if not self._process:
            return -1
        
        try:
            self._process.wait()
            return self._process.returncode or 0
        except KeyboardInterrupt:
            self.console.print("\n[yellow]收到中断信号...[/yellow]")
            self.terminate()
            return 130
    
    def stream_output(self) -> None:
        """Stream process output to console."""
        if not self._process or not self._process.stdout:
            return
        
        try:
            for line in self._process.stdout:
                print(line, end="")
        except KeyboardInterrupt:
            self.terminate()
    
    def terminate(self) -> None:
        """Terminate the running process."""
        if self._process and self._process.poll() is None:
            self.console.print("[yellow]正在终止进程...[/yellow]")
            try:
                if sys.platform == "win32":
                    self._process.terminate()
                else:
                    self._process.send_signal(signal.SIGTERM)
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception:
                pass
    
    @property
    def is_running(self) -> bool:
        """Check if process is running."""
        return self._process is not None and self._process.poll() is None
    
    @property
    def return_code(self) -> Optional[int]:
        """Get process return code."""
        if self._process:
            return self._process.returncode
        return None


def launch_server_with_config(config) -> int:
    """Launch server with configuration.
    
    Args:
        config: LaunchConfig object
        
    Returns:
        Exit code
    """
    console = Console()
    manager = ProcessManager()
    
    args = config.to_server_args()
    
    console.print()
    console.print("[bold green]🚀 启动 llama-server...[/bold green]")
    console.print(f"[dim]模型: {config.model.display_name}[/dim]")
    console.print(f"[dim]地址: {config.server_url}[/dim]")
    console.print()
    console.print("[yellow]按 Ctrl+C 停止服务器[/yellow]")
    console.print()
    
    try:
        manager.launch_server(args)
        manager.stream_output()
        return manager.wait_for_process()
    except FileNotFoundError:
        return 1
    except KeyboardInterrupt:
        manager.terminate()
        return 130


def launch_cli_with_config(config) -> int:
    """Launch CLI with configuration.
    
    Args:
        config: LaunchConfig object
        
    Returns:
        Exit code
    """
    console = Console()
    manager = ProcessManager()
    
    args = config.to_cli_args()
    
    console.print()
    console.print("[bold green]🚀 启动 llama-cli...[/bold green]")
    console.print(f"[dim]模型: {config.model.display_name}[/dim]")
    console.print()
    
    try:
        manager.launch_cli(args)
        return manager.wait_for_process()
    except FileNotFoundError:
        return 1
    except KeyboardInterrupt:
        manager.terminate()
        return 130


def check_prerequisites() -> dict[str, bool]:
    """Check if required binaries are available.
    
    Returns:
        Dict of binary name to availability
    """
    return {
        "llama-server": get_llama_server_path() is not None,
        "llama-cli": get_llama_cli_path() is not None,
    }
