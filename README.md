# 本地大模型 LoRA 微调

这个仓库主要做三件事：

- `chat_finetune.py`：加载基座模型聊天。
- `finetune_lora.py`：加载基座模型做 LoRA 微调。
- `chat_lora.py`：加载基座模型 + LoRA 聊天。
- `export_gguf.py`：可选，把基座或合并 LoRA 后的模型导出为 GGUF。

下文命令默认从仓库根目录执行。脚本会自动读取仓库根目录的 `config.json`。

---

## 快速开始

安装依赖：

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

确认或修改 `config.json`。当前配置已经按本机现有路径写好：

- 基座模型：`../../../../Projects/LLMs/models/Qwen3.5-9B`
- 训练数据：`./data/train_cute.jsonl`
- LoRA 输出：`../../../../Projects/LLMs/outputs/Qwen3.5-9B`

然后直接运行三步：

```bash
python3 chat_finetune.py
```

```bash
python3 finetune_lora.py
```

```bash
python3 chat_lora.py
```

也可以指定另一个配置文件：

```bash
python3 chat_finetune.py --config ./config.json
python3 finetune_lora.py --config ./config.json
python3 chat_lora.py --config ./config.json
```

脚本启动时会打印实际使用的配置文件、profile、基座模型、微调文件和输出目录。

`finetune_lora.py` 会自动判断输出目录：

- 输出目录不存在或为空：新建 LoRA 并训练。
- 输出目录已有 LoRA：询问是否在已有 LoRA 上继续微调，输入 `y` 才会继续。
- 输出目录有非 LoRA 文件：停止，避免写乱目录。

---

## config.json

三个常用行为都在同一个 `config.json` 里。通常只需要先改 `paths`：

- `paths.model`：基座模型目录。
- `paths.data`：训练 JSONL。
- `paths.lora_output`：LoRA 输出目录，也是 LoRA 聊天加载的目录。

公共参数写在 `defaults`，profile 里可以用 `${paths.xxx}` 复用路径，避免同一个路径写好几遍：

- `profiles.base_chat`：基座聊天参数。
- `profiles.finetune`：微调路径与训练参数。
- `profiles.lora_chat`：基座 + LoRA 聊天参数。

常改项：

| 配置项 | 作用 | 建议 |
|------|------|------|
| `model` | Hugging Face 基座模型目录 | 必须包含 `config.json` |
| `data` | 训练 JSONL | 默认 `./data/train_cute.jsonl` |
| `output` | LoRA 输出目录 | 父目录必须存在 |
| `lora` | 聊天时加载的 LoRA 目录 | 指向微调输出目录 |
| `thinking` | Qwen3.5 思考开关 | 普通聊天常用 `off` |
| `max_new_tokens` | 每轮最多生成 token 数 | 常用 `512` |
| `temperature` / `top_p` | 采样参数 | 常用 `0.7` / `0.9` |

---

## 训练参数

| 配置项 | 作用 | 建议 |
|------|------|------|
| `epochs` | 训练轮数 | 小数据集先试 `1` 到 `3` |
| `learning_rate` | 学习率 | 常用 `0.0001` 或 `0.00005`；不稳就调小 |
| `batch` | 每步样本数 | 显存不够就用 `1` |
| `grad_accum` | 梯度累积 | 等效 batch 约为 `batch * grad_accum` |
| `max_seq_len` | 最大 token 长度 | 显存吃紧可降到 `1024` |
| `lora_r` | LoRA rank | 常用 `8`、`16`、`32` |
| `lora_alpha` | LoRA 缩放 | 常用 `2 * r`，例如 `r=16` 时 `32` |
| `use_4bit` | CUDA 上 4bit 加载 | 仅 NVIDIA CUDA，需要 `bitsandbytes` |

继续微调已有 LoRA 时，会沿用原 LoRA 结构，`lora_r` 和 `lora_alpha` 不会重新改变已有结构。

---

## 数据格式

训练数据是 JSONL，每行一个对象，必须包含 `messages`：

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}
```

示例文件在 `data/train_cute.jsonl`。

---

## 导出 GGUF（可选）

这一步只在需要给 LM Studio 等工具加载 GGUF 时使用。需要本机已有 `llama.cpp`，并在 `llama.cpp` 里装过它自己的 requirements。

导出基座：

```bash
python3 export_gguf.py \
  --model ../../../../Projects/LLMs/models/Qwen3.5-9B \
  --llama-cpp ../../../../Projects/LLMs/llama.cpp \
  --gguf-out ../../../../Projects/LLMs/merged/Qwen3.5-9B/Qwen3.5-9B-base-f16.gguf \
  --outtype f16
```

导出基座 + LoRA：

```bash
python3 export_gguf.py \
  --model ../../../../Projects/LLMs/models/Qwen3.5-9B \
  --lora ../../../../Projects/LLMs/outputs/Qwen3.5-9B \
  --merged-dir ../../../../Projects/LLMs/merged/Qwen3.5-9B/hf \
  --llama-cpp ../../../../Projects/LLMs/llama.cpp \
  --gguf-out ../../../../Projects/LLMs/merged/Qwen3.5-9B/Qwen3.5-9B-f16.gguf \
  --outtype f16 \
  --overwrite
```

### 导入 LM Studio

导出 GGUF 后，可以用 LM Studio 的 `lms` 命令导入。建议用 `--copy` 保留原始 GGUF 文件，避免默认导入时把文件移动到 LM Studio 的模型目录：

```bash
lms import --copy <path-to-gguf> --user-repo <creator/model-name>
```

参数说明：

- `--copy`：复制 GGUF 到 LM Studio 模型目录，原文件仍保留在 `merged/` 下。
- `<path-to-gguf>`：换成你实际导出的 GGUF 文件路径。
- `<creator/model-name>`：换成你想在 LM Studio UI 中显示的分类名，例如 `silmoon/xiaoming-9b`。

如果不确定分类名，也可以不加 `--user-repo`，运行 `lms import --copy /path/to/model.gguf` 后按提示选择 `Interactive import`，手动填写 creator 和 model name。

---

## 常见问题

| 问题 | 处理 |
|------|------|
| 找不到模型或缺少 `config.json` | 检查 `config.json` 里的 `model` 是否指向 Hugging Face 模型根目录 |
| 输出目录父路径不存在 | 先创建父目录，例如 `mkdir -p ../../../../Projects/LLMs/outputs` |
| 输出目录已有 LoRA | 输入 `y` 继续微调，输入 `n` 停止 |
| 无法识别 `qwen3_5` 等模型类型 | 升级 Python 以及 `transformers` / `torch` |
| Mac 训练 loss 或 grad 出现 NaN | 调小 `learning_rate`，或增大 `grad_accum` |
| GGUF 转换失败 | 升级 `llama.cpp`，并确认已安装它的依赖 |

查看完整参数：

```bash
python3 chat_finetune.py -h
python3 finetune_lora.py -h
python3 chat_lora.py -h
python3 export_gguf.py -h
```
