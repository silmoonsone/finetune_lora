#!/usr/bin/env python3
"""
本地 Hugging Face 格式多模态/文本模型：终端多轮对话（纯文本模板）。
默认读取脚本同目录 config.json 中的 base_chat 配置。
支持 Qwen3.5 类 chat 模板中的思考开关（thinking on|off，对应 enable_thinking）。
依赖：torch、transformers、peft（若使用 --lora）。
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import torch
from config_utils import apply_config, load_profile, resolve_path
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _epilog(default_profile: str, command_name: str) -> str:
    return f"""
默认用法：
  python3 {command_name}

脚本会自动读取脚本同目录 config.json 的 profiles.{default_profile}。
如需改模型路径、生成参数、system prompt，请改 config.json。

thinking 与 Qwen3.5 chat 模板一致：
    on  = 开启「思考」：生成前保留 <think> 块，由模型先推理再答
    off = 关闭：模板使用空思考占位，模型直接作答（常见对话默认）

对话命令：/reset 清空历史；/quit 或 Ctrl+D 退出。
"""


def _resolve_device(explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _require_hf_dir(path: Path, *, label: str) -> None:
    if not path.is_dir():
        _die(f"错误：{label} 不是有效目录：\n  {path}")
    if not (path / "config.json").is_file():
        _die(f"错误：{label} 下缺少 config.json：\n  {path}")


def _model_from_pretrained_kwargs(
    device: str, dtype: torch.dtype, bnb: BitsAndBytesConfig | None
) -> dict:
    kw: dict = {"trust_remote_code": True, "dtype": dtype if bnb is None else torch.bfloat16}
    if bnb is not None:
        return {**kw, "quantization_config": bnb, "device_map": "auto"}
    if device == "cuda":
        return {**kw, "device_map": "auto"}
    return {**kw, "device_map": None}


def _gen_reply(
    model,
    tokenizer: AutoTokenizer,
    inputs: dict,
    gen_kwargs: dict,
    *,
    stream: bool,
) -> str:
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    if not stream:
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(out[0, n_in:], skip_special_tokens=True).strip()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gkw = {**inputs, **gen_kwargs, "streamer": streamer}
    print("模型: ", end="", flush=True)

    def _run() -> None:
        with torch.inference_mode():
            model.generate(**gkw)

    th = threading.Thread(target=_run)
    th.start()
    parts: list[str] = []
    try:
        for ch in streamer:
            print(ch, end="", flush=True)
            parts.append(ch)
    finally:
        th.join()
    print()
    return "".join(parts).strip()


def main(default_profile: str = "base_chat") -> None:
    ap = argparse.ArgumentParser(
        description=f"chat_finetune：终端交互，默认自动读取 config.json 的 {default_profile}。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(default_profile, Path(sys.argv[0]).name),
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
    ap.add_argument("--lora", type=Path, default=None, metavar="DIR", help="LoRA 适配器目录；省略则只加载基座")
    ap.add_argument(
        "--system",
        type=str,
        default=None,
        metavar="TEXT",
        help="若指定：在首条用户消息前插入该 system（/reset 后会再次插入）",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default=None,
        metavar="NAME",
        help="计算设备：auto / cuda / mps / cpu（auto 按环境自动选）",
    )
    ap.add_argument(
        "--thinking",
        choices=("on", "off"),
        default=None,
        metavar="on|off",
        help="思考模式：on=模板启用 enable_thinking（先推理块再答）；off=不启用（直接答）",
    )
    ap.add_argument("--max-new-tokens", type=int, default=None, metavar="N", help="每轮最多生成多少个新 token")
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help="采样温度；>0 且未加 --greedy 时启用采样。加 --greedy 时忽略此项",
    )
    ap.add_argument("--top-p", type=float, default=None, metavar="FLOAT", help="nucleus 采样 top_p；贪心时可填 1")
    ap.add_argument("--greedy", action="store_true", default=None, help="贪心解码（关闭采样）")
    ap.add_argument("--no-stream", action="store_true", default=None, help="关闭流式输出，整段生成后再打印")
    ap.add_argument("--4bit", dest="use_4bit", action="store_true", default=None, help="仅 CUDA：4bit 加载基座（需 bitsandbytes）")

    args = ap.parse_args()
    try:
        profile_config, config_base_dir, config_file = load_profile(
            args.config,
            args.profile or default_profile,
            default_config=str(Path(__file__).with_name("config.json")),
        )
    except ValueError as e:
        _die(f"错误：读取配置失败：{e}")
    args = apply_config(args, profile_config)

    for name, hint in (
        ("model", "--model 或配置中的 model"),
        ("device", "--device 或配置中的 device"),
        ("thinking", "--thinking 或配置中的 thinking"),
        ("max_new_tokens", "--max-new-tokens 或配置中的 max_new_tokens"),
        ("temperature", "--temperature 或配置中的 temperature"),
        ("top_p", "--top-p 或配置中的 top_p"),
    ):
        if getattr(args, name) is None:
            _die(f"错误：缺少 {hint}")
    if args.greedy is None:
        args.greedy = False
    if args.no_stream is None:
        args.no_stream = False
    if args.use_4bit is None:
        args.use_4bit = False

    model_path = resolve_path(args.model, base_dir=config_base_dir)
    assert model_path is not None
    _require_hf_dir(model_path, label="--model")

    lora_path = resolve_path(args.lora, base_dir=config_base_dir)
    if lora_path is not None:
        if not lora_path.is_dir():
            _die(f"错误：--lora 不是目录：\n  {lora_path}")
        if not (lora_path / "adapter_config.json").is_file():
            _die(
                f"错误：LoRA 目录中缺少 adapter_config.json：\n  {lora_path}\n"
                "请确认该目录含 adapter_model.safetensors 等训练输出。"
            )

    print("运行配置：", file=sys.stderr)
    print(f"  配置文件: {config_file if config_file is not None else '未使用'}", file=sys.stderr)
    print(f"  配置 profile: {args.profile or default_profile}", file=sys.stderr)
    print(f"  基座模型: {model_path}", file=sys.stderr)
    print(f"  微调文件: {lora_path if lora_path is not None else '未使用'}", file=sys.stderr)
    print("  输出目录: 不适用", file=sys.stderr)

    device = _resolve_device(args.device)
    dtype = _pick_dtype(device)
    if lora_path is not None and device == "mps":
        dtype = torch.float32
        print("提示：LoRA + MPS 使用 fp32 加载/推理，减轻数值异常。", file=sys.stderr)

    tok_src = str(lora_path) if lora_path is not None else str(model_path)
    print(f"设备: {device}  计算 dtype: {dtype}  tokenizer 自: {tok_src}", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    except Exception as e:
        _die(f"错误：加载 tokenizer 失败：{e}")

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb = None
    if args.use_4bit:
        if device != "cuda":
            _die("错误：--4bit 仅在 --device cuda（或 auto 落到 cuda）时可用。")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("正在加载基座模型…", file=sys.stderr)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), **_model_from_pretrained_kwargs(device, dtype, bnb)
        )
    except ValueError as e:
        if "qwen3_5" in str(e).lower() or "does not recognize" in str(e).lower():
            _die(
                "错误：当前 transformers 无法识别该模型类型（例如 qwen3_5）。\n"
                "请使用 Python 3.10+，并升级 transformers / torch。\n"
                f"原始信息：{e}"
            )
        raise
    except Exception as e:
        _die(f"错误：加载基座模型失败：{e}")

    if bnb is None and device != "cuda":
        model = model.to(device)

    if lora_path is not None:
        print(f"正在加载 LoRA：{lora_path}", file=sys.stderr)
        try:
            model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=False)
        except Exception as e:
            _die(f"错误：加载 LoRA 失败：{e}")

    model.eval()

    stream_on = not args.no_stream
    do_sample = (not args.greedy) and (args.temperature > 0)
    enable_thinking = args.thinking == "on"
    messages: list[dict] = []

    print()
    print(f"模式: {'基座 + LoRA（' + lora_path.name + '）' if lora_path else '仅基座'}")
    print(
        f"生成: 思考={'开' if enable_thinking else '关'}, max_new_tokens={args.max_new_tokens}, "
        f"{'贪心' if args.greedy or not do_sample else f'采样 temperature={args.temperature} top_p={args.top_p}'}, "
        f"流式={'开' if stream_on else '关'}"
    )
    print("已就绪。/reset 清空历史；/quit 或 Ctrl+D 退出。\n" + "-" * 48)

    while True:
        try:
            user = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            print("再见。")
            break
        if user == "/reset":
            messages.clear()
            print("(已清空对话历史)")
            continue

        if args.system and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": args.system})
        messages.append({"role": "user", "content": user})

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError as e:
            if "enable_thinking" in str(e).lower() or "unexpected keyword" in str(e).lower():
                _die(
                    "错误：apply_chat_template 不支持 enable_thinking。\n"
                    "需模型自带对应 chat_template（如 Qwen3.5）。\n"
                    f"详情：{e}"
                )
            raise

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        gen_kw: dict = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kw["temperature"] = max(args.temperature, 1e-5)
            gen_kw["top_p"] = args.top_p

        reply = _gen_reply(model, tokenizer, inputs, gen_kw, stream=stream_on)
        messages.append({"role": "assistant", "content": reply})
        if not stream_on:
            print(f"模型: {reply}")


if __name__ == "__main__":
    main()
