"""Tests for launch argument construction and command consistency."""

import shlex
from pathlib import Path

from llama_launcher.metadata import ModelMetadata
from llama_launcher.prompts import LaunchConfig, build_launch_command


def make_model() -> ModelMetadata:
    # Windows-style backslash path: list2cmdline keeps backslashes as-is
    m = ModelMetadata(file_path=Path(r"C:\models\test-Q4_K_M.gguf"), file_size_bytes=1024**3)
    m.architecture = "qwen35"
    m.quantization_type = "Q4_K_M"
    return m


def cmd_tokens(cmd: str) -> list[str]:
    # posix=False: keep backslashes literal (Windows quoting rules)
    return shlex.split(cmd, posix=False)


def test_ctx_zero_omits_c_server():
    c = LaunchConfig(model=make_model(), mode="server", context_length=0)
    assert "-c" not in c.to_server_args()


def test_ctx_zero_omits_c_cli():
    c = LaunchConfig(model=make_model(), mode="cli", context_length=0)
    assert "-c" not in c.to_cli_args()


def test_ctx_positive_includes_c():
    c = LaunchConfig(model=make_model(), mode="server", context_length=8192)
    args = c.to_server_args()
    assert args[args.index("-c") + 1] == "8192"


def test_mmproj_explicit_config(workdir):
    mproj = workdir / "mmproj-test-BF16.gguf"
    mproj.write_bytes(b"x")
    c = LaunchConfig(model=make_model(), mode="server", mmproj_path=mproj)
    args = c.to_server_args()
    assert "--mmproj" in args
    assert str(mproj) in args


def test_mmproj_falls_back_to_model(workdir):
    mproj = workdir / "mmproj-test-BF16.gguf"
    mproj.write_bytes(b"x")
    m = make_model()
    m.mmproj_path = mproj
    c = LaunchConfig(model=m, mode="server")
    assert "--mmproj" in c.to_server_args()


def test_build_launch_matches_server_args():
    c = LaunchConfig(
        model=make_model(),
        mode="server",
        gpu_layers=-1,
        fit="off",
        context_length=8192,
    )
    cmd = build_launch_command(c)
    parts = cmd_tokens(cmd)
    assert parts[0] == "llama-server.exe"
    assert parts[1:] == c.to_server_args()


def test_build_launch_matches_cli_args():
    c = LaunchConfig(
        model=make_model(),
        mode="cli",
        gpu_layers=0,
        fit="off",
        context_length=0,
    )
    cmd = build_launch_command(c)
    parts = cmd_tokens(cmd)
    assert parts[0] == "llama-cli.exe"
    assert parts[1:] == c.to_cli_args()


def test_build_launch_fit_on_auto():
    # fit=on: no -ngl/--fit, relies on llama-server default (--fit on)
    c = LaunchConfig(model=make_model(), mode="server", gpu_layers=-1, fit="on")
    parts = cmd_tokens(build_launch_command(c))
    assert parts[1:] == c.to_server_args()
    assert not any(p.startswith("-ngl") or p == "--fit" for p in parts)


def test_server_args_include_batch_and_cache():
    c = LaunchConfig(
        model=make_model(),
        mode="server",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        flash_attn="on",
    )
    args = c.to_server_args()
    assert "--cache-type-k" in args
    assert "--cache-type-v" in args
    assert "--flash-attn" in args
    assert "-b" in args and "-ub" in args


def test_advanced_args_present_in_server():
    c = LaunchConfig(
        model=make_model(),
        mode="server",
        n_cpu_moe=12,
        cpu_moe=True,
        kv_offload="off",
        override_tensor="ffn_down_exps=CPU",
        threads=8,
    )
    args = c.to_server_args()
    assert args[args.index("--n-cpu-moe") + 1] == "12"
    assert "--cpu-moe" in args
    assert "--no-kv-offload" in args
    assert args[args.index("--override-tensor") + 1] == "ffn_down_exps=CPU"
    assert args[args.index("-t") + 1] == "8"


def test_advanced_args_kv_on_explicit():
    c = LaunchConfig(model=make_model(), mode="server", kv_offload="on")
    assert "--kv-offload" in c.to_server_args()


def test_advanced_args_unset_not_present():
    # Nothing set -> no advanced flags, llama.cpp uses its own defaults
    c = LaunchConfig(model=make_model(), mode="server")
    args = c.to_server_args()
    for flag in ("--n-cpu-moe", "--cpu-moe", "--kv-offload", "--no-kv-offload",
                 "--override-tensor", "-t"):
        assert flag not in args


def test_advanced_cpu_moe_false_not_present():
    c = LaunchConfig(model=make_model(), mode="server", cpu_moe=False)
    assert "--cpu-moe" not in c.to_server_args()


def test_advanced_args_in_cli():
    c = LaunchConfig(
        model=make_model(),
        mode="cli",
        n_cpu_moe=5,
        cpu_moe=True,
        kv_offload="off",
        override_tensor="blk.3.attn_q=CPU",
    )
    args = c.to_cli_args()
    assert args[args.index("--n-cpu-moe") + 1] == "5"
    assert "--cpu-moe" in args
    assert "--no-kv-offload" in args
    assert args[args.index("--override-tensor") + 1] == "blk.3.attn_q=CPU"


def test_spec_decoding_args_present_server(workdir):
    m = make_model()
    draf = workdir / "mtp-foo.gguf"
    draf.write_bytes(b"x")
    m.drafter_path = draf
    m.drafter_type = "draft-mtp"
    args = LaunchConfig(model=m, mode="server").to_server_args()
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-model") + 1] == str(draf)


def test_spec_decoding_args_present_cli(workdir):
    m = make_model()
    draf = workdir / "LFM2.5-8B-A1B-DSpark-F16.gguf"
    draf.write_bytes(b"x")
    m.drafter_path = draf
    m.drafter_type = "draft-dspark"
    args = LaunchConfig(model=m, mode="cli").to_cli_args()
    assert args[args.index("--spec-type") + 1] == "draft-dspark"
    assert args[args.index("--spec-draft-model") + 1] == str(draf)


def test_spec_decoding_default_off():
    args = LaunchConfig(model=make_model(), mode="server").to_server_args()
    assert "--spec-type" not in args
    assert "--spec-draft-model" not in args


def test_spec_decoding_explicit_overrides_model(workdir):
    m = make_model()
    model_draf = workdir / "mtp-fallback.gguf"
    model_draf.write_bytes(b"x")
    m.drafter_path = model_draf
    m.drafter_type = "draft-mtp"
    explicit = workdir / "other-dspark.gguf"
    explicit.write_bytes(b"x")
    c = LaunchConfig(
        model=m, mode="server",
        drafter_path=explicit, drafter_type="draft-dspark",
    )
    args = c.to_server_args()
    assert args[args.index("--spec-type") + 1] == "draft-dspark"
    assert args[args.index("--spec-draft-model") + 1] == str(explicit)


def test_spec_decoding_falls_back_to_model(workdir):
    m = make_model()
    draf = workdir / "mtp-foo.gguf"
    draf.write_bytes(b"x")
    m.drafter_path = draf
    m.drafter_type = "draft-mtp"
    args = LaunchConfig(model=m, mode="server").to_server_args()
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-model") + 1] == str(draf)


def test_sampling_args_present():
    c = LaunchConfig(
        model=make_model(), mode="server",
        temperature=0.5, top_p=0.9, top_k=30, repeat_penalty=1.3,
    )
    args = c.to_server_args()
    assert args[args.index("--temp") + 1] == "0.5"
    assert args[args.index("--top-p") + 1] == "0.9"
    assert args[args.index("--top-k") + 1] == "30"
    assert args[args.index("--repeat-penalty") + 1] == "1.3"


def test_sampling_default_off():
    args = LaunchConfig(model=make_model(), mode="server").to_server_args()
    for flag in ("--temp", "--top-p", "--top-k", "--repeat-penalty"):
        assert flag not in args