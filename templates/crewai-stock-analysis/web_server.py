#!/usr/bin/env python3
"""CrewAI Stock Analysis - Web API Server

Wraps the multi-agent analysis in a REST API accessible via port forwarding.

Endpoints:
    GET  /health              - Health check (used by readiness hook)
    GET  /analyze/<ticker>    - Run analysis for a ticker (streaming response)
    GET  /reports             - List completed reports
    GET  /reports/<ticker>    - Get a specific report
"""
import os
import sys
import threading
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

from flask import Flask, Response, jsonify, stream_with_context

app = Flask(__name__)

# Track running analyses to prevent duplicates
running_analyses: set[str] = set()
analysis_lock = threading.Lock()


@app.route("/health")
def health():
    """Health check endpoint for readiness hook."""
    model_file = Path(".ollama_model")
    if not model_file.exists():
        return jsonify({"status": "initializing", "message": "Model not yet configured"}), 503
    return jsonify({"status": "ready", "model": model_file.read_text().strip()})


@app.route("/analyze/<ticker>")
def analyze(ticker: str):
    """Run stock analysis for a ticker. Returns streaming text output."""
    ticker = ticker.upper().strip()

    if not ticker.isalpha() or len(ticker) > 5:
        return jsonify({"error": "Invalid ticker. Use 1-5 letters (e.g., NVDA)"}), 400

    with analysis_lock:
        if ticker in running_analyses:
            return jsonify({"error": f"Analysis for {ticker} already in progress"}), 409
        running_analyses.add(ticker)

    def generate():
        try:
            yield f"Starting analysis for {ticker}...\n"
            yield "This may take 5-15 minutes depending on model size.\n\n"

            model_file = Path(".ollama_model")
            if not model_file.exists():
                yield "Error: Model not configured. Wait for initialization to complete.\n"
                return

            ollama_model = model_file.read_text().strip()
            yield f"Using model: {ollama_model}\n\n"

            # Set env var before importing crew (it reads OLLAMA_MODEL on init)
            os.environ["OLLAMA_MODEL"] = ollama_model
            from crew import StockAnalysisCrew

            crew = StockAnalysisCrew()
            result = crew.crew().kickoff(inputs={"ticker": ticker})

            # Save report
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            report_path = output_dir / f"{ticker}_analysis.md"
            report_path.write_text(str(result))

            yield f"\n{'=' * 60}\n"
            yield f"Analysis complete for {ticker}\n"
            yield f"Report saved to: {report_path}\n"
            yield f"{'=' * 60}\n\n"
            yield str(result)
        except Exception as e:
            yield f"\nError during analysis: {e}\n"
        finally:
            with analysis_lock:
                running_analyses.discard(ticker)

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no"},
    )


@app.route("/reports")
def list_reports():
    """List all completed analysis reports."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return jsonify({"reports": []})

    reports = []
    for f in sorted(reports_dir.glob("*_analysis.md")):
        ticker = f.stem.replace("_analysis", "")
        reports.append(
            {
                "ticker": ticker,
                "file": str(f),
                "size_bytes": f.stat().st_size,
            }
        )

    return jsonify({"reports": reports})


@app.route("/reports/<ticker>")
def get_report(ticker: str):
    """Get a specific analysis report."""
    ticker = ticker.upper().strip()
    report_path = Path("reports") / f"{ticker}_analysis.md"

    if not report_path.exists():
        return jsonify({"error": f"No report found for {ticker}"}), 404

    return Response(report_path.read_text(), mimetype="text/plain")


if __name__ == "__main__":
    print("Starting CrewAI web server on port 8501...", flush=True)
    app.run(host="0.0.0.0", port=8501, debug=False)
