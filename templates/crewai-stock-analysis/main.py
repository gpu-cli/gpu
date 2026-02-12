#!/usr/bin/env python3
"""CrewAI Stock Analysis - CLI Mode

Run a single stock analysis from the command line.

Usage:
    gpu run python main.py NVDA      # Analyze a single stock

For interactive use, start the web server with `gpu use .` and use:
    curl http://localhost:8501/analyze/NVDA
"""
import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def run_analysis(ticker: str) -> str:
    """Run stock analysis for a single ticker."""
    log(f"\n{'=' * 60}")
    log(f"Analyzing {ticker}...")
    log("This may take 5-15 minutes depending on model size.")
    log("=" * 60)

    from crew import StockAnalysisCrew

    crew = StockAnalysisCrew()
    result = crew.crew().kickoff(inputs={"ticker": ticker})

    # Save report
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{ticker}_analysis.md"
    report_path.write_text(str(result))

    log(f"\nReport saved: {report_path}")
    return str(result)


def main() -> None:
    # Load model config from setup
    model_file = Path(".ollama_model")
    if not model_file.exists():
        log("Error: Run 'gpu use .' first to initialize Ollama and download the model")
        sys.exit(1)

    ollama_model = model_file.read_text().strip()
    os.environ["OLLAMA_MODEL"] = ollama_model
    log(f"Using model: {ollama_model}")

    # Require a ticker argument
    if len(sys.argv) < 2:
        log("Usage: gpu run python main.py <TICKER>")
        log("       gpu run python main.py NVDA")
        log("")
        log("For interactive analysis, use the web API:")
        log("  gpu use .  # starts the web server")
        log("  curl http://localhost:8501/analyze/NVDA")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    if not ticker.isalpha() or len(ticker) > 5:
        log(f"Invalid ticker: {ticker}. Use 1-5 letters (e.g., NVDA)")
        sys.exit(1)

    run_analysis(ticker)


if __name__ == "__main__":
    main()
