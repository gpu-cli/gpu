#!/usr/bin/env python3
"""
vLLM Server for Multi-Agent Orchestration

Starts an OpenAI-compatible API server with Qwen 2.5 72B.
Use this as the LLM backend for CrewAI, LangGraph, or any OpenAI-compatible client.

Usage:
    # Start server (keep running in background)
    gpu run --publish 8000:8000 python serve.py

    # Then connect your agents to http://localhost:8000/v1
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct-AWQ",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum context length (default: 32768)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (default: 0.90)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="awq",
        help="Quantization method (default: awq)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Server - Multi-Agent Backend")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"Max context: {args.max_model_len} tokens")
    print(f"GPU memory utilization: {args.gpu_memory_utilization * 100:.0f}%")
    print("=" * 60)
    print()
    print("Server will be available at:")
    print(f"  - OpenAI API: http://localhost:{args.port}/v1")
    print(f"  - Health check: http://localhost:{args.port}/health")
    print()
    print("Connect your agents with:")
    print('  OPENAI_API_BASE="http://localhost:8000/v1"')
    print('  OPENAI_API_KEY="not-needed"')
    print()
    print("Starting server...")
    print()

    # Build vLLM command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--host",
        "0.0.0.0",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    # Run vLLM server
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
