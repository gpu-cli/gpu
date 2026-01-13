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

# Force unbuffered output for real-time feedback
os.environ["PYTHONUNBUFFERED"] = "1"


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def main():
    # Get ticker from command line or use default
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    log("=" * 60)
    log("🚀 CrewAI Stock Analysis")
    log("=" * 60)
    log(f"📊 Analyzing: {ticker}")

    # Import here to show progress before heavy imports
    from ollama_utils import get_vram_gb, select_model, ensure_ollama_running

    # Detect GPU and select model
    log("\n🔍 Detecting GPU...")
    vram = get_vram_gb()
    model = select_model(vram)
    log(f"🖥️  VRAM: {vram}GB detected")
    log(f"🤖 Model: {model}")

    # Setup Ollama
    log("\n⏳ Setting up Ollama...")
    ollama_model = ensure_ollama_running(model)
    log("✅ Ollama ready!")

    # Set environment variable for crew.py to use
    os.environ["OLLAMA_MODEL"] = ollama_model

    # Import crew after Ollama is ready
    log("\n📦 Loading CrewAI agents...")
    from crew import StockAnalysisCrew

    # Run analysis with phase indicators
    log("\n" + "=" * 60)
    log(f"🔬 Starting analysis of {ticker}")
    log("   This may take 5-15 minutes depending on model size.")
    log("=" * 60)

    log("\n🔍 Phase 1/3: Research Analyst gathering data...")
    log("   → Searching for news, financials, market data")

    crew = StockAnalysisCrew()
    result = crew.crew().kickoff(inputs={"ticker": ticker})

    # Save report
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{ticker}_analysis.md"
    report_path.write_text(str(result))

    log("\n" + "=" * 60)
    log("✅ Analysis complete!")
    log(f"📁 Report saved to: {report_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
