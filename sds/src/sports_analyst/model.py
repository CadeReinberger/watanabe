"""Ollama / DeepSeek-R1 interaction layer."""

import re
from pathlib import Path

import ollama


DEFAULT_MODEL = "deepseek-r1:14b"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_CODE_FENCE_UNCLOSED_RE = re.compile(r"```(?:python)?\n(.*?)$", re.DOTALL)
_PYTHON_START_RE = re.compile(r"^(import |from |#!|def |class |\w+ = )", re.MULTILINE)

# Sport keyword → (library name, doc filename)
_SPORT_HINTS: list[tuple[list[str], str, str]] = [
    (
        ["baseball", "mlb", "home run", "homerun", "batting average", "era", "rbi",
         "pitcher", "batter", "hank aaron", "babe ruth", "barry bonds", "willie mays",
         "statcast", "baseball reference", "fangraphs"],
        "pybaseball",
        "01_mlb.md",
    ),
    (
        ["basketball", "nba", "points per game", "rebounds", "assists", "three pointer",
         "lebron", "michael jordan", "steph curry", "kobe"],
        "nba_api",
        "02_nba_wnba.md",
    ),
    (
        ["wnba", "women's basketball", "women basketball"],
        "nba_api (with league_id='10')",
        "02_nba_wnba.md",
    ),
    (
        ["football", "nfl", "touchdown", "quarterback", "rushing yards", "receiving yards",
         "passing yards", "super bowl", "tom brady", "patrick mahomes"],
        "nfl_data_py",
        "03_nfl.md",
    ),
    (
        ["hockey", "nhl", "goals", "assists", "saves", "gretzky", "ovechkin", "crosby",
         "power play", "penalty kill"],
        "nhl API (requests)",
        "04_nhl.md",
    ),
    (
        ["college football", "ncaa football", "cfb", "sec", "big ten", "acc"],
        "cfbd",
        "05_ncaa.md",
    ),
    (
        ["college basketball", "ncaa basketball", "march madness", "ncaab", "ncaaw",
         "college hoops"],
        "sportsipy",
        "05_ncaa.md",
    ),
    (
        ["soccer", "mls", "premier league", "epl", "bundesliga", "la liga",
         "serie a", "champions league", "goal", "nwsl"],
        "soccerdata",
        "06_soccer.md",
    ),
    (
        ["cricket", "test match", "odi", "t20", "wickets", "batting average",
         "sachin", "kohli", "tendulkar", "ashes"],
        "cricpy / ESPN cricinfo",
        "07_cricket.md",
    ),
    (
        ["rugby", "nrl", "super rugby", "six nations", "all blacks", "springboks",
         "rugby union", "rugby league", "scrum", "try"],
        "requests (ESPN/World Rugby API)",
        "08_rugby.md",
    ),
]

DOCS_DIR = Path(__file__).parent / "api_docs"


def _load_doc(filename: str) -> str:
    return (DOCS_DIR / filename).read_text(encoding="utf-8")


def _load_all_docs() -> str:
    return "\n\n---\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.glob("*.md"))
    )


def _detect_sport(question: str) -> tuple[str | None, str | None]:
    """Return (library_hint, focused_doc) for the most likely sport in the question."""
    q = question.lower()
    best: tuple[int, str, str] | None = None
    for keywords, lib, doc_file in _SPORT_HINTS:
        score = sum(1 for kw in keywords if kw in q)
        if score > 0 and (best is None or score > best[0]):
            best = (score, lib, doc_file)
    if best:
        _, lib, doc_file = best
        return lib, _load_doc("00_overview.md") + "\n\n---\n\n" + _load_doc(doc_file)
    return None, None


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by DeepSeek-R1."""
    return _THINK_RE.sub("", text).strip()


def _extract_code(text: str) -> str:
    """Robustly extract Python source from a model response."""
    # 1. Properly closed code fence
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # 2. Unclosed code fence
    match = _CODE_FENCE_UNCLOSED_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if _PYTHON_START_RE.search(candidate):
            return candidate

    # 3. Find first Python-like line and take everything from there
    lines = text.split("\n")
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("import ", "from ", "#!", "def ", "class ")):
            candidate = "\n".join(lines[i:]).strip()
            candidate = re.sub(r"\n```.*$", "", candidate, flags=re.DOTALL).strip()
            return candidate

    # 4. Last resort — return raw (triggers retry)
    return text.strip()


def _start_ollama_if_needed() -> None:
    """Try to start the ollama daemon if it isn't reachable."""
    import subprocess, time, shutil
    try:
        ollama.list()
        return  # already running
    except Exception:
        pass
    # Find ollama binary
    ollama_bin = shutil.which("ollama") or str(Path.home() / ".local" / "bin" / "ollama")
    if not Path(ollama_bin).exists():
        return  # can't find it; let the caller handle the error
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)


def ensure_model(model: str = DEFAULT_MODEL) -> None:
    """Pull the model if it isn't present locally, starting ollama if needed."""
    _start_ollama_if_needed()
    try:
        local_models = {m["name"] for m in ollama.list().get("models", [])}
    except Exception:
        local_models = set()

    base = model.split(":")[0]
    if any(base in name for name in local_models):
        return

    print(f"[sports-analyst] Pulling {model} — this may take a while on first run...")
    for chunk in ollama.pull(model, stream=True):
        status = chunk.get("status", "") if isinstance(chunk, dict) else ""
        if "pulling" in status or "verifying" in status:
            print(".", end="", flush=True)
    print()


def _chat_simple(model: str, system: str, user: str) -> str:
    """Call ollama.chat and return the stripped response text."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = response["message"]["content"]
    return _strip_think(raw)


def generate_code(question: str, api_docs: str, model: str = DEFAULT_MODEL) -> str:
    """
    Ask the model to produce a self-contained Python script that answers *question*.
    Uses sport-detection to focus the API docs and add a library hint.
    """
    # Detect sport and focus docs
    lib_hint, focused_docs = _detect_sport(question)
    docs_to_use = focused_docs if focused_docs else api_docs
    hint = f"\nIMPORTANT: For this question, use the `{lib_hint}` library.\n" if lib_hint else ""

    system = """You are a sports data scientist who responds ONLY with Python code.
Your ENTIRE response must be a single ```python ... ``` code block.
Do NOT write any text, explanation, or markdown outside the code block.
The script must print its answer to stdout."""

    user = f"""Available Python libraries:

{docs_to_use}
{hint}
Question to answer with a Python script:
"{question}"

Rules:
- ONLY output a ```python ... ``` block — nothing else
- The script must import and use the correct library
- Print the final answer clearly to stdout
- Wrap network calls in try/except
- Do NOT assume external files exist"""

    raw = _chat_simple(model, system, user)
    return _extract_code(raw)


def generate_code_with_error(
    question: str,
    api_docs: str,
    previous_code: str,
    error_output: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the model to fix a broken script."""
    lib_hint, focused_docs = _detect_sport(question)
    docs_to_use = focused_docs if focused_docs else api_docs
    hint = f"\nIMPORTANT: Use the `{lib_hint}` library.\n" if lib_hint else ""

    system = """You are a Python debugging expert who responds ONLY with corrected Python code.
Your ENTIRE response must be a single ```python ... ``` code block.
Do NOT write any text or explanation outside the code block."""

    # Only show relevant error context, not the full prose script
    prev_snippet = previous_code[:2000] if previous_code else "(no previous code)"
    err_snippet = error_output[:800] if error_output else "(no error)"

    user = f"""Fix this Python script so it correctly answers: "{question}"

Error:
{err_snippet}

Broken script:
```python
{prev_snippet}
```

Available libraries:
{docs_to_use}
{hint}
Output ONLY the fixed ```python ... ``` block. The script must print the answer to stdout."""

    raw = _chat_simple(model, system, user)
    return _extract_code(raw)


def generate_response(question: str, data_output: str, model: str = DEFAULT_MODEL) -> str:
    """Formulate a natural-language answer given raw data output."""
    system = (
        "You are an expert sports analyst. "
        "Provide clear, direct, informative answers based on data."
    )
    user = f"""Question: "{question}"

Data from sports database:
{data_output}

Answer the question based on this data. Be direct and accurate.
If the data seems incomplete or the number seems off, note that."""

    return _chat_simple(model, system, user)
