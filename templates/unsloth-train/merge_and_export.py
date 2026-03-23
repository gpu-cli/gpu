#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import json
from pathlib import Path

import unsloth  # noqa: F401 — side-effect: patches transformers before other imports
from unsloth import FastLanguageModel


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Unsloth adapters for export")
    parser.add_argument("--config", default="training.json")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--output-dir", default="exports/merged_16bit")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model_cfg = cfg["model"]
    runtime_cfg = cfg.get("runtime", {})
    adapter_dir = Path(
        args.adapter_dir or runtime_cfg.get("final_checkpoint_dir", "checkpoints/final")
    )

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dtype=None,
        load_in_4bit=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
    print(f"Merged model exported to {output_dir}")


if __name__ == "__main__":
    main()
