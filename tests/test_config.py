"""Tests for configuration profile loading."""

from llama_launcher.config import load_profile


def test_load_profile_bool_flash_attn(workdir):
    p = workdir / "p.json"
    p.write_text(
        '{"gpu_layers": 20, "flash_attn": true, "context_length": 4096}',
        encoding="utf-8",
    )
    prof = load_profile(p)
    assert prof is not None
    assert prof.flash_attn == "on"
    assert prof.gpu_layers == 20
    assert prof.context_length == 4096


def test_load_profile_bool_flash_attn_false(workdir):
    p = workdir / "p.json"
    p.write_text('{"flash_attn": false}', encoding="utf-8")
    prof = load_profile(p)
    assert prof is not None
    assert prof.flash_attn == "off"


def test_load_profile_string_flash_attn(workdir):
    p = workdir / "p.json"
    p.write_text('{"flash_attn": "auto"}', encoding="utf-8")
    prof = load_profile(p)
    assert prof is not None
    assert prof.flash_attn == "auto"


def test_load_profile_comment_fields_stripped(workdir):
    p = workdir / "p.json"
    p.write_text(
        '{"_说明": "x", "_comment": "y", "gpu_layers": -1, "fit": "off"}',
        encoding="utf-8",
    )
    prof = load_profile(p)
    assert prof is not None
    assert prof.gpu_layers == -1
    assert prof.fit == "off"


def test_load_profile_defaults_for_missing_fields(workdir):
    p = workdir / "p.json"
    p.write_text("{}", encoding="utf-8")
    prof = load_profile(p)
    assert prof is not None
    assert prof.gpu_layers == -1
    assert prof.context_length == 4096
    assert prof.cache_type_k == "f16"
    assert prof.batch_size == 512


def test_load_profile_respects_defrag_thold(workdir):
    # defrag_thold is deprecated upstream but must still load for compat
    p = workdir / "p.json"
    p.write_text('{"defrag_thold": 0.5}', encoding="utf-8")
    prof = load_profile(p)
    assert prof is not None
    assert prof.defrag_thold == 0.5


def test_load_profile_advanced_fields(workdir):
    p = workdir / "p.json"
    p.write_text(
        '{"n_cpu_moe": 12, "cpu_moe": true, "kv_offload": "off",'
        ' "override_tensor": "ffn_down_exps=CPU"}',
        encoding="utf-8",
    )
    prof = load_profile(p)
    assert prof is not None
    assert prof.n_cpu_moe == 12
    assert prof.cpu_moe is True
    assert prof.kv_offload == "off"
    assert prof.override_tensor == "ffn_down_exps=CPU"


def test_load_profile_advanced_defaults(workdir):
    p = workdir / "p.json"
    p.write_text("{}", encoding="utf-8")
    prof = load_profile(p)
    assert prof is not None
    assert prof.n_cpu_moe is None
    assert prof.cpu_moe is None
    assert prof.kv_offload is None
    assert prof.override_tensor is None


def test_load_profile_cpu_moe_null_stays_unspecified(workdir):
    # Presets leave advanced values blank (null); null cpu_moe stays
    # "not set" so llama.cpp's own default applies.
    p = workdir / "p.json"
    p.write_text('{"cpu_moe": null}', encoding="utf-8")
    prof = load_profile(p)
    assert prof is not None
    assert prof.cpu_moe is None
    # And nothing is displayed as a default value
    adv = dict(prof.get_advanced_lines())
    assert adv["cpu-moe"] == "(未设置)"
    assert adv["KV offload"] == "(未设置)"
    assert adv["CPU 线程"] == "(未设置)"


def test_profile_summary_has_two_sections(workdir):
    p = workdir / "p.json"
    p.write_text('{"n_cpu_moe": 8, "cpu_moe": true, "threads": 4}', encoding="utf-8")
    prof = load_profile(p)
    assert prof is not None
    sections = prof.get_summary_sections()
    assert [name for name, _lines in sections] == ["基础配置", "高级配置"]
    adv = dict(sections[1][1])
    assert adv["n-cpu-moe"] == "8"
    assert adv["cpu-moe"] == "on"
    assert adv["CPU 线程"] == "4"