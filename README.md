# 本地 Qwen3.5：对话 · LoRA 微调 · 导出 GGUF

本仓库三个脚本，典型顺序是：**在本机 Python 里装好依赖 → 放基座模型 →（可选）先聊基座 → 准备数据 → LoRA 微调 → 用 LoRA 再聊 →（可选）导出 GGUF**。  
不强制使用虚拟环境；用系统 Python、conda、pyenv 等均可，只要 **`pip install` 与 `python` 属于同一套环境**（避免装包和跑脚本用了两个不同解释器）。

| 脚本 | 作用 |
|------|------|
| `chat_finetune.py` | 终端多轮对话（基座 ± LoRA） |
| `finetune_lora.py` | 用 JSONL 做 LoRA 监督微调（SFT） |
| `export_gguf.py` | 合并 LoRA 并调用本机 **llama.cpp** 转成 `.gguf` |

---

## 一、环境需求（先对照再装）

| 项目 | 说明 |
|------|------|
| **操作系统** | macOS / Linux / Windows 均可；有 **NVIDIA GPU** 可用 CUDA 加速训练与推理 |
| **Python** | **建议 3.10～3.12**（过旧可能装不上新版 `torch`/`transformers`；其它版本以本机 `pip` 能否装上 `requirements.txt` 为准）。用系统自带、`conda`、`pyenv` 或下节可选的 **venv** 均可，无强制要求 |
| **磁盘** | 基座模型体积大（如 2B/9B），另预留空间给 LoRA 输出与可选 GGUF |
| **内存 / 显存** | Mac 上常用 **MPS + CPU 内存**；显式训练参数见下文，可按机器把 `--batch`、`--max-seq-len` 调小 |

---

## 二、依赖包、版本与用途

以下与仓库根目录 **`requirements.txt`** 一致（安装时用该文件即可）。

| 包 | 版本约束（摘录） | 用途 |
|----|------------------|------|
| **torch** | `>=2.5.0` | 张量计算、CUDA/MPS、模型加载 |
| **transformers** | `>=4.57.0` | 加载 HF 模型与 tokenizer、chat 模板 |
| **accelerate** | `>=1.0.0` | 分布式/设备映射（训练常用） |
| **peft** | `>=0.14.0` | LoRA 适配器加载、合并 |
| **trl** | `>=0.20.0` | `SFTTrainer` 做监督微调 |
| **datasets** | `>=3.0.0` | 将 JSONL 转为训练用 `Dataset` |
| **safetensors** | `>=0.4.0` | 安全加载权重 |

**可选（未写入 `requirements.txt`，按需自行安装）：**

| 包 | 何时需要 |
|----|-----------|
| **bitsandbytes** | 仅在 **CUDA** 上使用本仓库的 **`--4bit`**（QLoRA / 4bit 推理）时 |
| **pillow** | 若以后用多模态 `AutoProcessor` 做图像输入时（当前终端对话脚本以文本为主） |

**Qwen3.5 与 `transformers`：** `model_type=qwen3_5` 需较新的实现。若 `pip install -r requirements.txt` 后仍报「无法识别模型类型」，请：

1. 确认 Python **≥3.10**；
2. 按 `requirements.txt` 文件顶部注释，尝试从源码安装最新版，例如：  
   `pip install -U git+https://github.com/huggingface/transformers.git`

---

## 三、Python 准备与依赖安装（不强制虚拟环境）

**仓库根目录**：包含 `requirements.txt`、`chat_finetune.py` 等的目录（本仓库克隆后一般为 `finetune_lora`，路径随你本机存放位置而定）。**下文所有命令均假定你已在该目录打开终端**，无需再 `cd` 到其它占位路径。`python3` / `python` 请改成你实际用的命令（Windows 上可能是 `py -3.12` 等）。

### 1）要准备什么样的 Python

- **版本**：优先 **Python 3.10～3.12**（与「一、环境需求」一致）。
- **环境**：全局、conda、pyenv 等任意；关键是 **`pip` 与 `python` 成对**（同一前缀，避免「pip 装到 A，python 却是 B」）。

### 2）一次性安装 / 按清单「恢复」所有依赖

仓库根目录的 **`requirements.txt`** 既是**首次安装清单**，也是换机器、换目录时**按同样版本约束重装（恢复）依赖**的依据。

在仓库根目录执行（任选一种习惯写法）：

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

含义简述：

| 命令 | 作用 |
|------|------|
| `python3 -m pip install -U pip` | 升级 pip，减少解析依赖时的老问题（可选但推荐） |
| `python3 -m pip install -r requirements.txt` | **按文件里列出的包与版本下限，一次性装全**本仓库脚本所需依赖 |

若 **`torch` 安装失败**，请到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 按系统与 CUDA 选一条官方命令先装好 **torch**，再执行一次 `pip install -r requirements.txt`（其余包会继续补齐）。

### 3）（可选）想用虚拟环境隔离时

若你希望与系统包分开，可自建 venv，**仍在该 venv 里执行上一小节的 `pip install -r requirements.txt`** 即可：

```bash
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows cmd:  .venv\Scripts\activate.bat
# Windows PowerShell:  .venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install -r requirements.txt
```

之后运行脚本前，**记得先激活**同一 venv（或始终使用 `./.venv/bin/python` 全路径调用）。

### 4）准备基座模型（HF 目录）

将 **Hugging Face 格式**的 Qwen3.5（或其它兼容模型）放在本机**任意目录**，模型根目录下必须有 **`config.json`**。**`--model` 请始终改为你自己的基座模型路径**，不必放在本仓库内，也不必叫 `models`。

目录结构示例（路径与文件夹名均可自定）：

```text
/你的/大模型/目录/Qwen3.5-2B/
  config.json
  ...
```

模型可从 [Hugging Face Hub](https://huggingface.co/) 用 `huggingface-cli download` 或网页下载后解压；下文命令里的 `--model` 指向上述「含 `config.json` 的那一层」即可。

### 5）微调前：创建输出父目录（按需）

微调脚本的 **`--output`** 的**父目录必须已存在**。**`--output` 请改为你希望保存 LoRA 的目录**（任意本机路径均可），例如你希望写到 `/你的/训练输出/lora-run1`，则先：

```bash
mkdir -p /path/to/parent-of-your-lora-output
```

（把该路径换成你 `--output` 所选目录的**父目录**。）

**下文所有示例**里的 `python` 均表示：**你已选定的、且已执行过 `pip install -r requirements.txt` 的那个解释器**（若用 conda，请先 `conda activate` 到你的环境）。

**示例中的路径**：`/path/to/your/hf-model`、`/path/to/your/lora-output`、`/path/to/your/exports` 等均为占位符，请一律替换为你本机实际目录。

---

## 四、完整流程：基座聊天 → 微调 → 微调后再聊天

按顺序执行即可；若你**只做推理、不训练**，可只做 **A**，跳过 **B**。

### A. 启动聊天（仅基座，无 LoRA）

确认已：**第三节**里装好依赖，并已放好 **基座模型目录**。

```bash
python chat_finetune.py \
  --model /path/to/your/hf-model \
  --device auto \
  --thinking off \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9
```

运行后出现提示即可输入用户消息。**`/reset`** 清空历史；**`/quit`** 或 **Ctrl+D** 退出。

**一般只需改：** `--model` 指向你本机 Hugging Face 格式基座模型目录（任意路径）；必要时改 `--device`（`cuda` / `mps` / `cpu` / `auto`）。

---

### B. 启动微调（LoRA SFT）

确认已：**第三节**（含依赖与模型）、**第 5 步**若需要则已 `mkdir`、并已准备 **`data/*.jsonl`**（格式见第五节）。

```bash
python finetune_lora.py \
  --model /path/to/your/hf-model \
  --data ./data/train_cute.jsonl \
  --output /path/to/your/lora-output \
  --epochs 3 \
  --learning-rate 1e-4 \
  --batch 1 \
  --grad-accum 8 \
  --max-seq-len 2048 \
  --lora-r 16 \
  --lora-alpha 32
```

**说明：**

- 非 CUDA（如 Apple **MPS**）时基座会以 **fp32** 训练，更稳但更慢。
- 仅 **NVIDIA + CUDA** 可在命令末尾追加 **`--4bit`**（需已安装 **bitsandbytes**）。

---

### C. 启动微调后的聊天（基座 + LoRA）

微调结束后，`--output` 目录内会有 **`adapter_config.json`** 等文件。用同一基座路径，并加上 **`--lora`**：

```bash
python chat_finetune.py \
  --model /path/to/your/hf-model \
  --lora /path/to/your/lora-output \
  --device auto \
  --thinking off \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9
```

**一般只需改：** `--model`（基座目录）、`--lora`（与训练时 `--output` 一致，为你自选的 LoRA 输出目录）。

**Mac + LoRA + MPS：** 脚本会对 LoRA 推理使用 **fp32**，减轻部分 fp16 下的数值问题。

---

## 五、数据格式（`finetune_lora.py`）

每行一个 JSON 对象，**必须**含 `messages` 数组（`role` / `content`）。可含 `system`。

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}
```

示例文件：`data/train_cute.jsonl`。

---

## 六、对话常用可选项（追加在 `chat_finetune.py` 命令末尾）

| 参数 | 含义 |
|------|------|
| `--thinking on` | 模板启用思考链；`off` 为直接答 |
| `--system '你是助手…'` | 首轮前注入系统提示；`/reset` 后会再次注入 |
| `--greedy` | 贪心解码；温度仍可写 `0`，`--top-p 1` |
| `--no-stream` | 非流式，整段打完再显示 |
| `--device cuda` / `mps` / `cpu` | 固定设备 |
| `--4bit` | 仅 CUDA：4bit 加载基座 |

---

## 七、导出 GGUF（可选）

需本机已 clone **[llama.cpp](https://github.com/ggerganov/llama.cpp)**，并在其仓库根目录执行过 **`pip install -r requirements.txt`**。Qwen3.5 需较新的 llama.cpp。

**仅基座：**

```bash
python export_gguf.py \
  --model /path/to/your/hf-model \
  --llama-cpp /path/to/llama.cpp \
  --gguf-out /path/to/your/exports/qwen35.f16.gguf \
  --outtype f16
```

**基座 + LoRA**（`--merged-dir` 已存在且非空时需加 **`--overwrite`** 或换目录）：

```bash
python export_gguf.py \
  --model /path/to/your/hf-model \
  --lora /path/to/your/lora-output \
  --merged-dir /path/to/your/exports/merged-hf \
  --llama-cpp /path/to/llama.cpp \
  --gguf-out /path/to/your/exports/qwen35-lora.f16.gguf \
  --outtype f16
```

`--gguf-out`、`--merged-dir` 同样请改为你希望存放导出文件的本机目录（父目录需已存在，必要时先 `mkdir -p`）。

---

## 八、查看完整命令行参数

```bash
python chat_finetune.py -h
python finetune_lora.py -h
python export_gguf.py -h
```

---

## 九、目录约定（与 `.gitignore`）

| 路径 | 说明 |
|------|------|
| 基座模型 | **无固定仓库子目录**；通过各脚本的 **`--model`** 指向你本机任意 Hugging Face 格式模型目录 |
| LoRA / 训练输出 | **无固定仓库子目录**；通过 **`--output`** 指向你希望写入的目录（父目录须已存在） |
| `data/` | 训练用 JSONL，默认可放在仓库内，可入库 |

若你**选择**把大模型或输出放在仓库下的 `models/`、`outputs/` 等文件夹，可与 `.gitignore` 配合避免误提交权重；这不是脚本要求的路径。

---

## 十、常见问题（极简）

| 现象 | 方向 |
|------|------|
| 无法识别 `qwen3_5` | 升级 Python / `transformers`（见第二节） |
| Mac 训练 loss / grad NaN | 已倾向 fp32；可调小学习率或增大 `--grad-accum` |
| 转 GGUF 报架构不支持 | 升级 llama.cpp 后再试 |
| 提示输出目录父路径不存在 | 对你 `--output`（或导出相关路径）的**父目录**执行 `mkdir -p` |
| 参数报错「必填」 | 路径与多数数值参数须显式写出，缺什么补什么 |
