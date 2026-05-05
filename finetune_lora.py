#!/usr/bin/env python3
"""
本地 HF 格式模型：LoRA 文本 SFT（JSONL，每行含 messages）。
无默认模型路径、无默认数据路径、无默认输出目录；须全部显式指定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _is_lora_output_dir(path: Path) -> bool:
    # 常见 LoRA 保存产物；满足核心文件即可视为“可二次微调”的目录。
    names = {p.name for p in path.iterdir()}
    has_adapter_cfg = "adapter_config.json" in names
    has_adapter_weights = any(
        n in names for n in ("adapter_model.safetensors", "adapter_model.bin")
    )
    return has_adapter_cfg and has_adapter_weights


def _confirm_continue_from_lora_output(path: Path) -> None:
    print(
        "检测到输出目录中已存在 LoRA 微调产物：\n"
        f"  {path}\n"
        "你想怎么做？\n"
        "  [y] 继续（对已有目录执行二次微调并写入新结果）\n"
        "  [n] 终止",
        file=sys.stderr,
    )
    while True:
        try:
            ans = input("请输入 y 或 n: ").strip().lower()
        except EOFError:
            _die("错误：未读取到确认输入，已终止以避免覆盖现有产物。")
        if ans in {"y", "yes"}:
            print("已确认继续：将进行二次微调。", file=sys.stderr)
            return
        if ans in {"n", "no"}:
            _die("已按用户选择终止。")
        print("无效输入，请输入 y 或 n。", file=sys.stderr)


def _epilog() -> str:
    return """
必填参数（无内置默认路径）：
  --model    基座模型目录
  --data     训练 JSONL（每行 {"messages":[...]}）
  --output   输出目录（将写入 adapter 与 tokenizer）
  --epochs   训练轮数（浮点，例如 3 或 1.5）

数据格式示例（一行）：
  {"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}

运行示例：
  python finetune_lora.py \\
    --model /path/to/Qwen3.5-2B \\
    --data /path/to/train.jsonl \\
    --output /path/to/out-lora \\
    --epochs 3 \\
    --learning-rate 1e-4 \\
    --batch 1 \\
    --grad-accum 8 \\
    --max-seq-len 2048 \\
    --lora-r 16 \\
    --lora-alpha 32

说明：非 CUDA（如 Apple MPS）下基座与训练使用 fp32，以降低 LoRA 数值不稳定。
"""


def load_jsonl_messages(path: Path) -> Dataset:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败：{e}") from e
            if "messages" not in obj:
                raise ValueError(f"第 {line_no} 行缺少 `messages` 字段")
            rows.append({"messages": obj["messages"]})
    if not rows:
        raise ValueError("数据文件中没有有效样本（空文件或仅空行）")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="finetune_lora：LoRA 监督微调（SFT）。路径与训练规模参数均须显式给出。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    ap.add_argument("--model", type=Path, required=True, metavar="DIR", help="基座模型目录（必填）")
    ap.add_argument("--data", type=Path, required=True, metavar="FILE", help="训练 JSONL 路径（必填）")
    ap.add_argument("--output", type=Path, required=True, metavar="DIR", help="输出目录（必填）")
    ap.add_argument("--epochs", type=float, required=True, metavar="N", help="训练 epoch 数（必填）")
    ap.add_argument("--learning-rate", type=float, required=True, metavar="FLOAT", help="学习率（必填）")
    ap.add_argument("--batch", type=int, required=True, metavar="N", help="per_device_train_batch_size（必填）")
    ap.add_argument("--grad-accum", type=int, required=True, metavar="N", help="gradient_accumulation_steps（必填）")
    ap.add_argument("--max-seq-len", type=int, required=True, metavar="N", help="最大序列长度（必填）")
    ap.add_argument("--lora-r", type=int, required=True, metavar="N", help="LoRA rank r（必填）")
    ap.add_argument("--lora-alpha", type=int, required=True, metavar="N", help="LoRA alpha（必填）")
    ap.add_argument("--4bit", dest="use_4bit", action="store_true", help="QLoRA：4bit 基座（仅 CUDA）")

    args = ap.parse_args()

    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        _die(f"错误：--model 无效或缺少 config.json：\n  {model_path}")
    if not data_path.is_file():
        _die(f"错误：--data 不是文件：\n  {data_path}")
    out_parent = output_dir.parent
    if not out_parent.is_dir():
        _die(f"错误：输出目录的父路径不存在：\n  {out_parent}")
    if output_dir.exists():
        if not output_dir.is_dir():
            _die(f"错误：--output 不是目录：\n  {output_dir}")
        entries = [p for p in output_dir.iterdir()]
        if entries:
            if _is_lora_output_dir(output_dir):
                _confirm_continue_from_lora_output(output_dir)
            else:
                preview = ", ".join(sorted(p.name for p in entries)[:10])
                more = "" if len(entries) <= 10 else " ..."
                _die(
                    "错误：输出目录中存在非 LoRA 产物文件，已终止以避免写入到混杂目录。\n"
                    f"目录：{output_dir}\n"
                    f"内容预览：{preview}{more}"
                )

    for flag, val, pred, hint in (
        ("--epochs", args.epochs, lambda x: x > 0, "须为正数"),
        ("--learning-rate", args.learning_rate, lambda x: x > 0, "须为正数"),
        ("--batch", args.batch, lambda x: x >= 1, "须 >= 1"),
        ("--grad-accum", args.grad_accum, lambda x: x >= 1, "须 >= 1"),
        ("--max-seq-len", args.max_seq_len, lambda x: x >= 1, "须 >= 1"),
        ("--lora-r", args.lora_r, lambda x: x >= 1, "须 >= 1"),
        ("--lora-alpha", args.lora_alpha, lambda x: x >= 1, "须 >= 1"),
    ):
        if not pred(val):
            _die(f"错误：{flag} {hint}，当前为 {val!r}")

    try:
        train_dataset = load_jsonl_messages(data_path)
    except ValueError as e:
        _die(f"错误：读取训练数据失败：{e}")

    print(f"已加载 {len(train_dataset)} 条样本。输出目录：{output_dir}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    bnb = None
    if args.use_4bit:
        if not torch.cuda.is_available():
            _die("错误：--4bit 仅在 CUDA 可用时有效。")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    is_cuda = torch.cuda.is_available()
    if not is_cuda:
        model_dtype = torch.float32
    elif torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16
    model_kw: dict = {"dtype": model_dtype, "device_map": "auto", "trust_remote_code": True}
    if bnb is not None:
        model_kw["quantization_config"] = bnb

    print("正在加载基座模型…", file=sys.stderr)
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kw)
    except ValueError as e:
        if "qwen3_5" in str(e).lower() or "does not recognize" in str(e).lower():
            _die(f"错误：transformers 无法识别该模型类型。请升级 Python 与 transformers/torch。\n详情：{e}")
        raise
    except Exception as e:
        _die(f"错误：加载模型失败：{e}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        bf16=use_bf16,
        fp16=is_cuda and not use_bf16,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        max_length=args.max_seq_len,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    def formatting_prompts_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"完成。已保存至：{output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
