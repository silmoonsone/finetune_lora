#!/usr/bin/env python3
"""
将 Hugging Face 基座模型与（可选）LoRA 适配器导出为 GGUF，供 LM Studio 等加载。

流程：
  1. 若提供 --lora：merge LoRA → 保存到 --merged-dir
  2. 调用 llama.cpp 的 convert_hf_to_gguf.py 转为 .gguf

说明：Qwen3.5 需较新的 llama.cpp；若报「架构不支持」，请 git pull 升级。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _epilog() -> str:
    return """
依赖：torch、transformers、peft；以及已 clone 的 llama.cpp（根目录执行 pip install -r requirements.txt）。

必填：--model --llama-cpp --gguf-out --outtype
带 LoRA 时还需：--lora --merged-dir

示例见各参数 --help 或 README；典型：
  python export_gguf.py --model ./models/Qwen3.5-2B --lora ./outputs/lora-cute \\
    --merged-dir ./exports/merged-hf --llama-cpp ~/llama.cpp \\
    --gguf-out ./exports/out.f16.gguf --outtype f16
"""


def _convert_script(llama_cpp: Path) -> Path:
    p = (llama_cpp / "convert_hf_to_gguf.py").resolve()
    if p.is_file():
        return p
    _die(
        f"错误：未找到 convert_hf_to_gguf.py：\n  {llama_cpp}\n"
        "请确认 --llama-cpp 指向 llama.cpp 仓库根目录。"
    )


def _merge_lora(model_dir: Path, lora_dir: Path, merged_dir: Path, *, overwrite: bool) -> None:
    if merged_dir.exists() and any(merged_dir.iterdir()):
        if not overwrite:
            _die(f"错误：--merged-dir 已存在且非空：\n  {merged_dir}\n请换目录或加 --overwrite。")
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("正在合并 LoRA（CPU / fp32）…", file=sys.stderr)
    try:
        base = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), trust_remote_code=True, dtype=torch.float32, device_map="cpu"
        )
        merged = PeftModel.from_pretrained(base, str(lora_dir), is_trainable=False).merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True).save_pretrained(str(merged_dir))
    except Exception as e:
        _die(f"错误：合并或保存失败：{e}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="export_gguf：HF 基座 ± LoRA → GGUF（调用本机 llama.cpp 转换脚本）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    ap.add_argument("--model", type=Path, required=True, metavar="DIR", help="基座 HF 目录")
    ap.add_argument("--lora", type=Path, default=None, metavar="DIR", help="LoRA 目录（可选）")
    ap.add_argument("--merged-dir", type=Path, default=None, metavar="DIR", help="合并输出目录（与 --lora 同用时必填）")
    ap.add_argument("--llama-cpp", type=Path, required=True, metavar="DIR", help="llama.cpp 根目录")
    ap.add_argument("--gguf-out", type=Path, required=True, metavar="FILE", help="输出 .gguf")
    ap.add_argument("--outtype", type=str, required=True, metavar="TYPE", help="如 f16、bf16、f32")
    ap.add_argument("--overwrite", action="store_true", help="非空 merged-dir 时先删除")
    ap.add_argument("--verbose", action="store_true", help="转换加 --verbose")

    args = ap.parse_args()
    model_dir = args.model.expanduser().resolve()
    llama_cpp = args.llama_cpp.expanduser().resolve()
    gguf_out = args.gguf_out.expanduser().resolve()

    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        _die(f"错误：--model 无效或缺少 config.json：\n  {model_dir}")
    if not llama_cpp.is_dir():
        _die(f"错误：--llama-cpp 不是目录：\n  {llama_cpp}")

    lora = args.lora.expanduser().resolve() if args.lora else None
    if lora:
        if not lora.is_dir() or not (lora / "adapter_config.json").is_file():
            _die(f"错误：--lora 无效或缺少 adapter_config.json：\n  {lora}")
        if args.merged_dir is None:
            _die("错误：使用 --lora 时必须指定 --merged-dir。")
        merged = args.merged_dir.expanduser().resolve()
        _merge_lora(model_dir, lora, merged, overwrite=args.overwrite)
        hf_src = merged
    else:
        if args.merged_dir:
            print("提示：未使用 --lora，忽略 --merged-dir。", file=sys.stderr)
        hf_src = model_dir

    conv = _convert_script(llama_cpp)
    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(conv), str(hf_src), "--outfile", str(gguf_out), "--outtype", args.outtype]
    if args.verbose:
        cmd.append("--verbose")

    print("正在运行 convert_hf_to_gguf.py …", file=sys.stderr)
    try:
        subprocess.run(cmd, cwd=str(llama_cpp), check=True)
    except subprocess.CalledProcessError as e:
        _die(
            "错误：转换失败。常见原因：llama.cpp 过旧、未装 requirements、--outtype 无效。\n"
            f"返回码：{e.returncode}"
        )
    except FileNotFoundError:
        _die(f"错误：无法执行解释器或脚本：{sys.executable}")

    if not gguf_out.is_file():
        _die(f"错误：未生成文件：\n  {gguf_out}")
    print(f"完成：{gguf_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
