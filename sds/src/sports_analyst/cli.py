"""sports-analyst — CLI entry point."""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from sports_analyst import model as llm
from sports_analyst.runner import run_script

console = Console()

DOCS_DIR = Path(__file__).parent / "api_docs"
MAX_RETRIES = 3


def load_api_docs() -> str:
    """Concatenate all markdown doc files into one context string."""
    docs = []
    for md in sorted(DOCS_DIR.glob("*.md")):
        docs.append(md.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(docs)


def _spinner(label: str):
    return Live(Spinner("dots", text=Text(label, style="cyan")), refresh_per_second=10)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sports-analyst",
        description="Ask a sports data question and get a data-driven answer.",
    )
    parser.add_argument("question", nargs="+", help="Your sports question")
    parser.add_argument(
        "--model",
        default=os.environ.get("SPORTS_ANALYST_MODEL", llm.DEFAULT_MODEL),
        help=f"Ollama model to use (default: {llm.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--show-code",
        action="store_true",
        help="Print the generated Python script before running it",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the raw data output before the final answer",
    )
    args = parser.parse_args(argv)

    question = " ".join(args.question)
    model = args.model
    api_docs = load_api_docs()

    console.rule("[bold blue]Sports Analyst[/bold blue]")
    console.print(f"[bold]Question:[/bold] {question}\n")

    # --- Step 0: ensure ollama is running and model is present ---
    try:
        llm.ensure_model(model)
    except Exception as e:
        console.print(f"[red]Could not reach ollama: {e}[/red]")
        console.print(
            "Start ollama with:\n"
            "  ollama serve\n"
            "or if installed user-space:\n"
            "  ~/.local/bin/ollama serve"
        )
        sys.exit(1)

    # --- Step 1: generate a Python script ---
    code = None
    result = None

    with _spinner("Generating data-retrieval script..."):
        code = llm.generate_code(question, api_docs, model)

    if args.show_code:
        console.print(Panel(code, title="[bold green]Generated Script[/bold green]", expand=False))

    _ERROR_PATTERNS = (
        "An error occurred", "Error:", "Traceback", "ModuleNotFoundError",
        "AttributeError", "ImportError", "Exception", "No module named",
    )

    def _looks_like_error(text: str) -> bool:
        """Return True if the stdout output is primarily an error message."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return True
        error_lines = sum(1 for l in lines if any(p in l for p in _ERROR_PATTERNS))
        return error_lines > 0 and error_lines >= len(lines) - 1

    # --- Step 2: run the script (with retries) ---
    for attempt in range(1, MAX_RETRIES + 1):
        label = f"Running script (attempt {attempt}/{MAX_RETRIES})..."
        with _spinner(label):
            result = run_script(code)

        # Normalize: treat empty output or error-only output as failure
        if result.success and not result.stdout:
            result = result.__class__(
                stdout="", stderr="Script ran successfully but printed nothing.", returncode=1
            )
        elif result.success and _looks_like_error(result.stdout):
            result = result.__class__(
                stdout="", stderr=result.stdout, returncode=1
            )

        if result.success and result.stdout:
            break

        console.print(
            f"[yellow]Script failed (attempt {attempt}):[/yellow]\n{result.stderr[:800]}"
        )
        if attempt < MAX_RETRIES:
            with _spinner("Asking model to fix the script..."):
                code = llm.generate_code_with_error(
                    question, api_docs, code, result.stderr, model
                )
            if args.show_code:
                console.print(
                    Panel(code, title=f"[yellow]Fixed Script (attempt {attempt+1})[/yellow]", expand=False)
                )

    if not result or not result.stdout:
        console.print("[red]Could not retrieve data after all retries.[/red]")
        if result and result.stderr:
            console.print(f"Last error:\n{result.stderr}")
        sys.exit(1)

    if args.show_raw:
        console.print(Panel(result.stdout, title="[bold]Raw Data Output[/bold]", expand=False))

    # --- Step 3: generate a natural-language response ---
    with _spinner("Formulating answer..."):
        answer = llm.generate_response(question, result.stdout, model)

    console.print()
    console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
