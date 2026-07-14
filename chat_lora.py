#!/usr/bin/env python3
"""基座 + LoRA 终端聊天入口，默认读取 config.json 的 lora_chat。"""

from chat_finetune import main


if __name__ == "__main__":
    main(default_profile="lora_chat")
