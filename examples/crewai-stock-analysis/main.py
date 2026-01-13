#!/usr/bin/env python3
"""CrewAI Stock Analysis - GPU CLI Example

Multi-agent AI stock analysis powered by local LLMs on remote GPUs.

Usage:
    gpu run python main.py NVDA
    gpu run python main.py AAPL
    gpu run python main.py TSLA
"""
import os
import sys
from pathlib import Path
from ollama_utils import get_vram_gb, select_model, ensure_ollama_running


def main():
    # Get ticker from command line or use default
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print("=" * 60)
    print(f"CrewAI Stock Analysis - Analyzing {ticker}")
    print("=" * 60)

    # Setup Ollama with appropriate model based on GPU VRAM
    vram = get_vram_gb()
    model = select_model(vram)
    print(f"\nDetected {vram}GB VRAM, using model: {model}")

    ollama_model = ensure_ollama_running(model)

    # Set environment variable for crew.py to use
    os.environ["OLLAMA_MODEL"] = ollama_model

    # Import crew after Ollama is ready
    from crew import StockAnalysisCrew

    # Run analysis
    print(f"\nStarting analysis of {ticker}...")
    print("This may take 5-15 minutes depending on model size.\n")

    crew = StockAnalysisCrew()
    result = crew.crew().kickoff(inputs={"ticker": ticker})

    # Save report
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{ticker}_analysis.md"
    report_path.write_text(str(result))

    print("\n" + "=" * 60)
    print(f"Analysis complete! Report saved to {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
