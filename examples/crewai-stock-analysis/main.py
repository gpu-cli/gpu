#!/usr/bin/env python3
"""CrewAI Stock Analysis - Interactive Mode

Multi-agent AI stock analysis powered by local LLMs on remote GPUs.

Usage:
    gpu run python main.py           # Interactive mode
    gpu run python main.py NVDA      # Single query mode
"""
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crew import StockAnalysisCrew

# Force unbuffered output for real-time feedback
os.environ["PYTHONUNBUFFERED"] = "1"


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def run_analysis(ticker: str, crew: "StockAnalysisCrew") -> str:
    """Run stock analysis for a single ticker."""
    log(f"\n{'=' * 60}")
    log(f"Analyzing {ticker}...")
    log("This may take 5-15 minutes depending on model size.")
    log("=" * 60)

    result = crew.crew().kickoff(inputs={"ticker": ticker})

    # Save report
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{ticker}_analysis.md"
    report_path.write_text(str(result))

    log(f"\nReport saved: {report_path}")
    return str(result)


def interactive_mode(crew: "StockAnalysisCrew") -> None:
    """Interactive query loop."""
    log("\n" + "=" * 60)
    log("CrewAI Stock Analysis - Interactive Mode")
    log("=" * 60)
    log("Enter stock tickers to analyze (e.g., NVDA, AAPL, TSLA)")
    log("Type 'quit' or 'exit' to stop")
    log("=" * 60)

    while True:
        try:
            ticker = input("\nTicker> ").strip().upper()

            if not ticker:
                continue
            if ticker in ("QUIT", "EXIT", "Q"):
                log("Goodbye!")
                break
            if not ticker.isalpha() or len(ticker) > 5:
                log("Invalid ticker. Use 1-5 letters (e.g., NVDA)")
                continue

            run_analysis(ticker, crew)

        except KeyboardInterrupt:
            log("\nGoodbye!")
            break
        except EOFError:
            break


def main() -> None:
    # Load model config from setup
    model_file = Path(".ollama_model")
    if not model_file.exists():
        log("Error: Run 'gpu run python setup.py' first to initialize")
        sys.exit(1)

    ollama_model = model_file.read_text().strip()
    os.environ["OLLAMA_MODEL"] = ollama_model

    log(f"Using model: {ollama_model}")

    # Import crew after model is configured
    from crew import StockAnalysisCrew
    crew = StockAnalysisCrew()

    # Check for command line ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        run_analysis(ticker, crew)
    else:
        interactive_mode(crew)


if __name__ == "__main__":
    main()
