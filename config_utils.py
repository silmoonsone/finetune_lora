from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

_REF_RE = re.compile(r"\$\{([^}]+)\}")


def _lookup_ref(config: dict[str, Any], ref: str) -> Any:
    cur: Any = config
    for part in ref.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"配置引用不存在：${{{ref}}}")
        cur = cur[part]
    return cur


def _resolve_refs(value: Any, config: dict[str, Any], seen: set[str] | None = None) -> Any:
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        return {k: _resolve_refs(v, config, seen.copy()) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, config, seen.copy()) for v in value]
    if not isinstance(value, str):
        return value

    full = _REF_RE.fullmatch(value)
    if full:
        ref = full.group(1)
        if ref in seen:
            raise ValueError(f"配置引用存在循环：${{{ref}}}")
        return _resolve_refs(_lookup_ref(config, ref), config, seen | {ref})

    def _replace(match: re.Match[str]) -> str:
        ref = match.group(1)
        if ref in seen:
            raise ValueError(f"配置引用存在循环：${{{ref}}}")
        found = _resolve_refs(_lookup_ref(config, ref), config, seen | {ref})
        if isinstance(found, (dict, list)):
            raise ValueError(f"配置引用不能嵌入对象或数组：{match.group(0)}")
        return str(found)

    return _REF_RE.sub(_replace, value)


def load_profile(
    config_path: Path | None,
    profile_name: str,
    *,
    default_config: str = "config.json",
) -> tuple[dict[str, Any], Path | None, Path | None]:
    if config_path is None:
        config_path = Path(default_config)
        if not config_path.exists():
            return {}, None, None

    resolved = config_path.expanduser().resolve()
    try:
        config = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ValueError(f"配置文件不存在：{resolved}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件 JSON 解析失败：{e}") from e

    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("配置文件缺少 profiles 对象")

    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(profiles))
        raise ValueError(f"配置中不存在 profile：{profile_name}。可用 profile：{available}")

    merged: dict[str, Any] = {}
    defaults = config.get("defaults")
    if isinstance(defaults, dict):
        merged.update(defaults)
    merged.update(profile)
    return _resolve_refs(merged, config), resolved.parent, resolved


def apply_config(args: Namespace, config: dict[str, Any]) -> Namespace:
    for key, value in config.items():
        attr = key.replace("-", "_")
        if attr in {"script", "description"}:
            continue
        if not hasattr(args, attr):
            continue
        if getattr(args, attr) is None:
            setattr(args, attr, value)
    return args


def resolve_path(value: str | Path | None, *, base_dir: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()
