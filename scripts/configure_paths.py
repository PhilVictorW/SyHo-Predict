#!/usr/bin/env python3
"""Relocate paths in this release checkout. Preview by default; no external writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def replace_text(text, replacements):
    """Replace once, longest prefix first; replacement values are never reprocessed."""
    if not replacements:
        return text
    pattern = re.compile("|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def build_replacements(config, root):
    required = (
        "asset_root", "model_env", "libero_eval_env", "libero_train_env",
        "plus_env", "robotwin_env", "plus_indexed_data", "robotwin_sim_root",
        "cache_root", "legacy_root",
    )
    for key in required:
        value = config.get(key)
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError(f"{key} must be an absolute path")
        if not re.fullmatch(r"/[A-Za-z0-9_./+-]+", value) or ".." in Path(value).parts:
            raise ValueError(f"{key}: use a path without spaces, quotes, shell metacharacters or '..'")
        config[key] = value.rstrip("/")
    if not re.fullmatch(r"/[A-Za-z0-9_./+-]+", str(root)):
        raise ValueError("Checkout path must not contain spaces or shell metacharacters")
    host = "/mnt/hdd/wangchaoyang"
    projects = host + "/LatentWAM"
    envs = "/home/wangchaoyang/envs"
    replacements = {
        projects + "/LaWAM_aba2_change": str(root),
        projects + "/LaWAM_aba2": str(root),
        projects + "/LaWAM_without_subgoal/libero-plus/data/indexed_rlds": config["plus_indexed_data"],
        projects + "/LaWAM_without_subgoal": str(root),
        host + "/CKPT/LaWAM": config["asset_root"],
        host + "/results/LaWAM/runtime/RoboTwin": config["robotwin_sim_root"],
        "/mnt/hdd/hf_a100/wangchaoyang": config["cache_root"] + "/huggingface",
        "/home/wangchaoyang/.cache": config["cache_root"],
        "/home/wangchaoyang/CKPT": config["legacy_root"] + "/optimizer",
        envs + "/LaWAM": config["model_env"],
        envs + "/LIBERO4LaWAM": config["libero_eval_env"],
        envs + "/LIBERO": config["libero_train_env"],
        envs + "/libero-plus": config["plus_env"],
        envs + "/RoboTwin4LaWAM": config["robotwin_env"],
        host: config["legacy_root"],
        "/home/wangchaoyang": config["legacy_root"] + "/home",
    }
    return replacements


def relocate_sibling_sources(text):
    # These historical sibling checkouts supply the same public runtime modules.
    pattern = r"""ROOT\.parent\s*/\s*(["'])LaWAM_(?:aba2|without_subgoal)(/[^"']*)?\1"""
    return re.sub(pattern, lambda m: "ROOT" + (
        " / " + repr(m.group(2).lstrip("/")) if m.group(2) else ""), text)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply edits only to this release checkout")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "configs/release_sources.json").read_text())
    config = json.loads(args.config.read_text())
    replacements = build_replacements(config, root)
    receipt = root / ".syho-paths-applied.json"
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["configuration_sha256"] != identity:
            parser.error("This checkout was configured differently; use a fresh clone for different paths.")
        print("This checkout is already configured with these paths.")
        return
    changes = []
    for rel in inventory["paths"]:
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing a path outside the checkout: {rel}")
        if not path.is_file():
            continue
        text = path.read_text()
        new = replace_text(relocate_sibling_sources(text), replacements)
        if new != text:
            changes.append((path, new))
    for path, _ in changes:
        print(("UPDATE " if args.apply else "WOULD UPDATE ") + str(path.relative_to(root)))
    if args.apply:
        for path, text in changes:
            path.write_text(text)
        receipt.write_text(json.dumps({
            "configuration_sha256": identity,
            "files_changed": len(changes),
            "configuration": config,
        }, indent=2) + "\n")
    print(f"{len(changes)} files; " + ("applied inside checkout." if args.apply else "preview only; add --apply to write."))
    print("Binary checkpoint metadata, external data and environments were not changed.")


if __name__ == "__main__":
    main()
