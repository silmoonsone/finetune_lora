#!/usr/bin/env python3
"""
本地 HF 格式模型：LoRA 文本 SFT（JSONL，每行含 messages）。
默认读取脚本同目录 config.json 中的 finetune 配置。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from config_utils import apply_config, load_profile, resolve_path
from datasets import Dataset
from peft import LoraConfig, PeftModel
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
        "  [y] 在已有 LoRA 之上继续微调，并写回该目录\n"
        "  [n] 终止",
        file=sys.stderr,
    )
    while True:
        print("请输入 y 或 n: ", end="", file=sys.stderr, flush=True)
        ans = sys.stdin.readline()
        if not ans:
            _die("错误：未读取到确认输入，已终止以避免覆盖现有产物。")
        ans = ans.strip().lower()
        if ans in {"y", "yes"}:
            print("已确认：将加载已有 LoRA 并继续微调。", file=sys.stderr)
            return
        if ans in {"n", "no"}:
            _die("已按用户选择终止。")
        print("无效输入，请输入 y 或 n。", file=sys.stderr)


def _epilog() -> str:
    return """
默认用法：
  python3 finetune_lora.py

脚本会自动读取脚本同目录 config.json 的 profiles.finetune。
如需改模型路径、训练数据、输出目录、训练参数，请改 config.json。

数据格式示例（一行）：
  {"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"}]}

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


def _sync_model_special_tokens(model, tokenizer: AutoTokenizer) -> None:
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is None:
            continue
        setattr(model.config, name, value)
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, name, value)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="finetune_lora：LoRA 监督微调（SFT），默认自动读取 config.json 的 finetune。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="指定 JSON 配置文件；不指定时读取脚本同目录的 config.json",
    )
    ap.add_argument("--profile", type=str, default=None, metavar="NAME", help=argparse.SUPPRESS)
    ap.add_argument("--model", type=Path, default=None, metavar="DIR", help="基座模型目录")
    ap.add_argument("--data", type=Path, default=None, metavar="FILE", help="训练 JSONL 路径")
    ap.add_argument("--output", type=Path, default=None, metavar="DIR", help="输出目录")
    ap.add_argument("--epochs", type=float, default=None, metavar="N", help="训练 epoch 数")
    ap.add_argument("--learning-rate", type=float, default=None, metavar="FLOAT", help="学习率")
    ap.add_argument("--batch", type=int, default=None, metavar="N", help="per_device_train_batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, metavar="N", help="gradient_accumulation_steps")
    ap.add_argument("--max-seq-len", type=int, default=None, metavar="N", help="最大序列长度")
    ap.add_argument("--lora-r", type=int, default=None, metavar="N", help="LoRA rank r")
    ap.add_argument("--lora-alpha", type=int, default=None, metavar="N", help="LoRA alpha")
    ap.add_argument("--4bit", dest="use_4bit", action="store_true", default=None, help="QLoRA：4bit 基座（仅 CUDA）")

    args = ap.parse_args()
    try:
        profile_config, config_base_dir, config_file = load_profile(
            args.config,
            args.profile or "finetune",
            default_config=str(Path(__file__).with_name("config.json")),
        )
    except ValueError as e:
        _die(f"错误：读取配置失败：{e}")
    args = apply_config(args, profile_config)

    for name, hint in (
        ("model", "配置中的 model"),
        ("data", "配置中的 data"),
        ("output", "配置中的 output"),
        ("epochs", "配置中的 epochs"),
        ("learning_rate", "配置中的 learning_rate"),
        ("batch", "配置中的 batch"),
        ("grad_accum", "配置中的 grad_accum"),
        ("max_seq_len", "配置中的 max_seq_len"),
        ("lora_r", "配置中的 lora_r"),
        ("lora_alpha", "配置中的 lora_alpha"),
    ):
        if getattr(args, name) is None:
            _die(f"错误：缺少 {hint}。请检查 config.json。")
    if args.use_4bit is None:
        args.use_4bit = False

    model_path = resolve_path(args.model, base_dir=config_base_dir)
    data_path = resolve_path(args.data, base_dir=config_base_dir)
    output_dir = resolve_path(args.output, base_dir=config_base_dir)
    assert model_path is not None
    assert data_path is not None
    assert output_dir is not None

    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        _die(f"错误：--model 无效或缺少 config.json：\n  {model_path}")
    if not data_path.is_file():
        _die(f"错误：--data 不是文件：\n  {data_path}")
    out_parent = output_dir.parent
    if not out_parent.is_dir():
        _die(f"错误：输出目录的父路径不存在：\n  {out_parent}")

    continue_from_lora = False
    existing_lora_path: Path | None = None
    if output_dir.exists():
        if not output_dir.is_dir():
            _die(f"错误：--output 不是目录：\n  {output_dir}")
        entries = [p for p in output_dir.iterdir()]
        if entries:
            if _is_lora_output_dir(output_dir):
                existing_lora_path = output_dir
            else:
                preview = ", ".join(sorted(p.name for p in entries)[:10])
                more = "" if len(entries) <= 10 else " ..."
                _die(
                    "错误：输出目录中存在非 LoRA 产物文件，已终止以避免写入到混杂目录。\n"
                    f"目录：{output_dir}\n"
                    f"内容预览：{preview}{more}"
                )

    print("运行配置：", file=sys.stderr)
    print(f"  配置文件: {config_file if config_file is not None else '未使用'}", file=sys.stderr)
    print(f"  配置 profile: {args.profile or 'finetune'}", file=sys.stderr)
    print(f"  基座模型: {model_path}", file=sys.stderr)
    print(f"  微调文件: {existing_lora_path if existing_lora_path is not None else '未使用（将新建 LoRA）'}", file=sys.stderr)
    print(f"  微调数据: {data_path}", file=sys.stderr)
    print(f"  输出目录: {output_dir}", file=sys.stderr)

    if existing_lora_path is not None:
        _confirm_continue_from_lora_output(existing_lora_path)
        continue_from_lora = True

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
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    _sync_model_special_tokens(model, tokenizer)

    if continue_from_lora:
        print(f"正在加载已有 LoRA 继续训练：{output_dir}", file=sys.stderr)
        print("提示：继续微调时会沿用已有 LoRA 结构，忽略本次传入的 --lora-r / --lora-alpha。", file=sys.stderr)
        try:
            model = PeftModel.from_pretrained(model, str(output_dir), is_trainable=True)
        except Exception as e:
            _die(f"错误：加载已有 LoRA 失败：{e}")
        peft_config = None
    else:
        print("未检测到已有 LoRA：将创建新的 LoRA 适配器。", file=sys.stderr)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / args.batch / args.grad_accum))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = int(total_steps * 0.1)
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
        warmup_steps=warmup_steps,
        max_length=args.max_seq_len,
        packing=False,
        dataloader_pin_memory=is_cuda,
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
