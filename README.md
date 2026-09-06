# 🦙 llama-launcher

llama.cpp CLI 启动器 - 支持多模态模型与推测解码模块的智能启动器。

## ✨ 功能特性

- 🔍 **两阶段扫描** - 快速列出模型（秒级，只看文件名/大小），选定后才深度读取完整信息
- 🎯 **智能识别** - 自动读取架构、量化类型、层数、上下文长度
- 🖼️ **多模态** - 检测多模态模型，自动关联同目录投影器（mmproj）并交互确认
- ⚡ **推测解码** - 检测 MTP / DSpark drafter 模块，自动拼装 `--spec-type`，加速推理
- ⚙️ **基础配置** - 上下文、GPU 层数、KV 缓存量化、批处理、启动模式，带智能默认
- 🔧 **高级配置** - MoE 专家放置、KV offload、张量级放置、采样参数
- 📋 **命令输出** - 生成可直接执行的完整命令，回车启动或复制自行编辑
- 🚀 **灵活启动** - 支持 llama-server（HTTP API）和 llama-cli（交互式推理）

## 📦 安装

### 从源码安装

```bash
git clone https://github.com/hazjy/llama-cli-launcher
cd llama-cli-launcher
pip install -e .
```

### 直接安装依赖

```bash
pip install click rich gguf psutil
```

## 🚀 使用方法

### 基本用法

```bash
# 扫描默认目录并启动
llama

# 指定模型目录
llama -d ~/models

# 仅列出模型详情（全量扫描）
llama list -d ~/models

# 直接启动指定模型（跳过交互）
llama launch ./models/model.gguf

# 检查环境
llama check
```

### Windows：start.bat

双击 `start.bat` 自动完成三件事：定位启动器 → **自动发现本机任意最新版 llama.cpp build**（扫描 `D:\`、`C:\` 下的 `llama-b*-bin-win-*` 目录）→ 找模型目录 → 进入交互流程。

## 🎯 使用流程

### 两阶段扫描与交互启动

```bash
$ llama

🦙 llama-launcher
llama.cpp CLI 启动器 - 支持多模态模型

快速扫描模型...
✓ 找到 3 个模型

可用模型
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ # ┃ 名称                   ┃ 类型    ┃ 大小  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ 1 │ Qwythos-9B-v2-Q6_K     │ 🖼️ 多模态│ 6.9GB │
│ 2 │ LFM2.5-8B-A1B-Q5_K_M   │ 📝 文本 │ 5.6GB │
│ 3 │ Ling-3.0-tiny-UD-Q6_K  │ 📝 文本 │ 6.8GB │
└───┴────────────────────────┴──────────┴───────┘

选择模型编号 (1): 2
正在读取模型信息: LFM2.5-8B-A1B-Q5_K_M …
📋 LFM2.5-8B-A1B-Q5_K_M | LFM2.5 | Q5_K_M | 5.6GB | 24 层 | ctx 128000 | draft-dspark

要加载推测解码模块（MTP/DSpark）吗？ [y/n] (n): y
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┓
┃ # ┃ 文件名                   ┃ 类型         ┃ 大小  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━┩
│ 1 │ LFM2.5-8B-A1B-DSpark-F16│ draft-dspark │ 633MB │
└───┴──────────────────────────┴──────────────┴───────┘
选择模块编号 (1): 1

配置方式:
  1 - 手动配置 - 逐步设置参数
  2 - 加载配置 - 从文件加载预设
配置方式 (1): 1
是否启用高级配置？ [y/n] (n):

[基础配置: 上下文 → GPU 层数 → KV 缓存 → 批处理 → 模式 → host/port]

启动命令:
llama-server.exe -m ... --spec-type draft-dspark --spec-draft-model ...
回车 = 执行当前命令（如需修改，请复制到终端自行编辑执行）
```

### 多模态支持

扫描/深度扫描时检测多模态信号（GGUF 内 `clip.*` 字段或同目录 `mmproj-*.gguf`），
并自动关联投影器文件：

- 交互流程中确认后选择投影器，启动命令自动带 `--mmproj`（支持 `mmproj-模型名*.gguf` 与通用名 `mmproj-BF16.gguf`）
- **具体输入类型（图像/音频/视频）由 llama.cpp 版本与 mmproj 内容决定**，本工具只标记"多模态"，不做能力宣称

### 推测解码（MTP / DSpark）

检测同目录的附加模块文件并交互选择，自动拼装 `--spec-type` + `--spec-draft-model`：

| 文件名模式 | spec-type |
|---|---|
| `mtp-*.gguf` | `draft-mtp`（多 token 预测头） |
| `*-dspark*.gguf` | `draft-dspark`（DeepSeek 推测解码 drafter） |

`llama launch` 直接启动时会自动附带检测到的模块。

## 🔧 高级配置（手动模式选"是"后出现）

已设置的参数才会传入命令，未设置则由 llama.cpp 使用默认值：

| 参数 | 说明 |
|---|---|
| `-t N` | CPU 线程数 |
| `--n-cpu-moe N` | MoE 前 N 层专家留 CPU（仅 MoE 模型） |
| `--cpu-moe` | 所有 MoE 专家留 CPU（仅 MoE 模型） |
| `--kv-offload` | KV 缓存 offload 开关 |
| `--override-tensor "模式=类型"` | 张量级放置（如 `ffn_down_exps=CPU`） |
| `--temp` / `--top-p` / `--top-k` / `--repeat-penalty` | 采样参数 |

## 📋 命令参考

| 命令 | 说明 |
|------|------|
| `llama` / `llama run` | 快速扫描并交互式启动 |
| `llama list` | 全量扫描并列出模型详情 |
| `llama launch <model>` | 直接启动指定模型（自动附 mmproj / drafter） |
| `llama check` | 检查环境依赖 |

## 📁 项目结构

```
llama-cli-launcher/
├── llama_launcher/
│   ├── __init__.py      # 版本信息
│   ├── cli.py           # CLI 入口
│   ├── scanner.py       # 两阶段扫描 + 附件(mmproj/drafter)匹配
│   ├── metadata.py      # GGUF 元数据读取
│   ├── prompts.py       # 交互式配置 + 高级配置
│   ├── launcher.py      # 进程启动 + build 自动发现
│   ├── config.py        # 配置模板/预设
│   └── gpu.py           # GPU 检测
├── config/profiles/     # 配置预设 JSON
├── tests/
├── start.bat            # Windows 一键启动（自动发现最新 llama.cpp build）
├── pyproject.toml
└── README.md
```

## 🔧 前置要求

- Python 3.10+（Windows 多版本共存时，`start.bat` 优先使用带依赖的 `py -3.14`）
- llama.cpp 预编译二进制（[GitHub Releases](https://github.com/ggml-org/llama.cpp/releases) 的 `llama-b*-bin-win-*-x64` 或源码编译）；start.bat 自动发现最新 build

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License（见 [LICENSE](LICENSE)）