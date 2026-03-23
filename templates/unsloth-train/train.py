#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import json
import os
import shutil
import signal
from pathlib import Path
from typing import Optional

import unsloth
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    raise KeyboardInterrupt(f"Received signal {signum}")


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_eos_text(tokenizer) -> str:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return ""

    try:
        return tokenizer.decode([eos_token_id], skip_special_tokens=False)
    except Exception:
        token = getattr(tokenizer, "eos_token", None)
        return token if isinstance(token, str) else ""


def resolve_token_from_id(tokenizer, token_id: Optional[int]) -> Optional[str]:
    if token_id is None:
        return None

    try:
        token = tokenizer.convert_ids_to_tokens(token_id)
    except Exception:
        return None

    return token if isinstance(token, str) and token else None


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def format_dataset(dataset_cfg: dict, tokenizer):
    source = dataset_cfg.get("source", "local")
    path = dataset_cfg.get("path")
    split = dataset_cfg.get("split", "train")
    fmt = dataset_cfg.get("format", "alpaca")

    if source == "huggingface":
        name = dataset_cfg["name"]
        config_name = dataset_cfg.get("config")
        dataset = load_dataset(name, config_name, split=split)
    else:
        if not path:
            raise ValueError("dataset.path is required for local datasets")
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        suffix = data_path.suffix.lower()
        if suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(data_path), split="train")
        elif suffix == ".json":
            dataset = load_dataset("json", data_files=str(data_path), split="train")
        elif suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(data_path), split="train")
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

    eos_token = resolve_eos_text(tokenizer)

    if fmt == "text":
        text_field = dataset_cfg.get("text_field", "text")

        def map_text(examples):
            texts = [text + eos_token for text in examples[text_field]]
            return {"text": texts}

        return dataset.map(map_text, batched=True)

    if fmt == "alpaca":
        prompt_template = dataset_cfg.get(
            "prompt_template",
            """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}""",
        )
        instruction_field = dataset_cfg.get("instruction_field", "instruction")
        input_field = dataset_cfg.get("input_field", "input")
        output_field = dataset_cfg.get("output_field", "output")

        def map_alpaca(examples):
            texts = []
            instructions = examples[instruction_field]
            inputs = examples.get(input_field, [""] * len(instructions))
            outputs = examples[output_field]
            for instruction, inp, output in zip(instructions, inputs, outputs):
                texts.append(
                    prompt_template.format(instruction, inp, output) + eos_token
                )
            return {"text": texts}

        return dataset.map(map_alpaca, batched=True)

    if fmt == "sql_create_context":
        prompt_template = dataset_cfg.get(
            "prompt_template",
            """You are a careful text-to-SQL assistant. Given a natural language request and the database schema context, write a valid SQL query that answers the request. Return SQL only.\n\n### Question:\n{}\n\n### Schema Context:\n{}\n\n### SQL:\n{}""",
        )
        question_field = dataset_cfg.get("question_field", "question")
        context_field = dataset_cfg.get("context_field", "context")
        answer_field = dataset_cfg.get("answer_field", "answer")

        def map_sql(examples):
            texts = []
            questions = examples[question_field]
            contexts = examples[context_field]
            answers = examples[answer_field]
            for question, context, answer in zip(questions, contexts, answers):
                texts.append(
                    prompt_template.format(question, context, answer) + eos_token
                )
            return {"text": texts}

        return dataset.map(map_sql, batched=True)

    raise ValueError(f"Unsupported dataset.format: {fmt}")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model with Unsloth")
    parser.add_argument("--config", default="training.json")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg["training"]
    runtime_cfg = cfg.get("runtime", {})

    output_dir = Path(runtime_cfg.get("output_dir", "outputs/current"))
    latest_dir = Path(runtime_cfg.get("latest_checkpoint_dir", "checkpoints/latest"))
    final_dir = Path(runtime_cfg.get("final_checkpoint_dir", "checkpoints/final"))
    logs_dir = Path(runtime_cfg.get("logs_dir", "logs"))
    exports_dir = Path(runtime_cfg.get("exports_dir", "exports"))

    for path in [output_dir, latest_dir, final_dir, logs_dir, exports_dir]:
        ensure_dir(path)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside the template runtime. "
            "This usually means the Python environment installed a CPU-only torch build. "
            "Recreate the environment with the Unsloth install path in gpu.jsonc."
        )

    dtype_name = model_cfg.get("dtype")
    dtype = getattr(torch, dtype_name) if dtype_name else None

    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dtype=dtype,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        token=os.environ.get(model_cfg.get("hf_token_env", "HF_TOKEN")),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    if tokenizer.pad_token is None:
        fallback_pad_token = resolve_token_from_id(tokenizer, tokenizer.eos_token_id)
        if fallback_pad_token is not None:
            tokenizer.pad_token = fallback_pad_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("rank", 16),
        target_modules=lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("gradient_checkpointing", "unsloth"),
        random_state=runtime_cfg.get("seed", 3407),
        use_rslora=lora_cfg.get("use_rslora", False),
        loftq_config=lora_cfg.get("loftq_config"),
    )

    print("Loading dataset...")
    dataset = format_dataset(dataset_cfg, tokenizer)

    report_to = train_cfg.get("report_to", "none")
    if report_to == "wandb" and not os.environ.get("WANDB_API_KEY"):
        print(
            "WANDB report_to requested but WANDB_API_KEY is not set; falling back to no external reporting"
        )
        report_to = "none"

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=model_cfg.get("max_seq_length", 2048),
        packing=train_cfg.get("packing", False),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        warmup_steps=train_cfg.get("warmup_steps", 5),
        max_steps=train_cfg.get("max_steps", -1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_steps=train_cfg.get("save_steps", 25),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        seed=runtime_cfg.get("seed", 3407),
        report_to=report_to,
        logging_dir=str(logs_dir),
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    resume_from = args.resume_from
    if not resume_from:
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint is not None:
            resume_from = str(latest_checkpoint)

    def sync_checkpoint_dirs() -> None:
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint is not None:
            copy_tree(latest_checkpoint, latest_dir)

    metrics = {"stop_requested": False}

    try:
        print("Starting training...")
        trainer_stats = trainer.train(resume_from_checkpoint=resume_from)
        metrics = dict(trainer_stats.metrics)
        metrics["stop_requested"] = False
    except KeyboardInterrupt:
        print("Training interrupted. Saving latest checkpoint...")
        trainer.save_state()
        sync_checkpoint_dirs()
        trainer.save_model(str(latest_dir))
        tokenizer.save_pretrained(str(latest_dir))
        metrics = {"stop_requested": True}
    else:
        print("Training complete. Saving artifacts...")
        trainer.save_state()
        sync_checkpoint_dirs()
        trainer.save_model(str(latest_dir))
        tokenizer.save_pretrained(str(latest_dir))
        copy_tree(latest_dir, final_dir)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
    finally:
        metrics_path = logs_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
