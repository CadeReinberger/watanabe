#!/usr/bin/env bash
# setup.sh — install everything needed for sports-analyst
set -euo pipefail

# ── 0. Check / install ollama ──────────────────────────────────────────────
OLLAMA_BIN=""
if command -v ollama &>/dev/null; then
    OLLAMA_BIN=$(command -v ollama)
    echo "[setup] ollama found at $OLLAMA_BIN: $($OLLAMA_BIN --version 2>&1 | head -1)"
elif [[ -x "$HOME/.local/bin/ollama" ]]; then
    OLLAMA_BIN="$HOME/.local/bin/ollama"
    echo "[setup] ollama found at $OLLAMA_BIN"
else
    echo "[setup] Installing ollama (user-space)..."
    LATEST=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
    mkdir -p "$HOME/.local/bin"
    curl -fsSL "https://github.com/ollama/ollama/releases/download/${LATEST}/ollama-linux-amd64.tar.zst" | \
        zstd -d | tar -x -C "$HOME/.local/"
    OLLAMA_BIN="$HOME/.local/bin/ollama"
fi

# Ensure it's on PATH
export PATH="$HOME/.local/bin:$PATH"

# ── 1. Start ollama daemon if not running ─────────────────────────────────
if ! pgrep -x ollama &>/dev/null; then
    echo "[setup] Starting ollama daemon..."
    OLLAMA_HOST=127.0.0.1:11434 "$OLLAMA_BIN" serve &>/tmp/ollama.log &
    sleep 4
fi

# ── 2. Choose the best DeepSeek-R1 model that fits available VRAM ─────────
VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)
RAM_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)

echo "[setup] Free VRAM: ${VRAM_MB} MB  |  Free RAM: ${RAM_MB} MB"

# Pick largest model that comfortably fits (VRAM + RAM combined for offloading)
TOTAL_MB=$(( VRAM_MB + RAM_MB ))
if   (( TOTAL_MB >= 50000 )); then MODEL="deepseek-r1:32b"
elif (( TOTAL_MB >= 20000 )); then MODEL="deepseek-r1:14b"
elif (( TOTAL_MB >= 10000 )); then MODEL="deepseek-r1:8b"
elif (( TOTAL_MB >=  5000 )); then MODEL="deepseek-r1:7b"
else                               MODEL="deepseek-r1:1.5b"
fi

echo "[setup] Selected model: ${MODEL}"
echo "export SPORTS_ANALYST_MODEL=${MODEL}" > .env

# Pull the model (skip if already present)
if "$OLLAMA_BIN" list | grep -q "${MODEL%:*}"; then
    echo "[setup] Model ${MODEL} already present."
else
    echo "[setup] Pulling ${MODEL} — this will download several GB, please wait..."
    "$OLLAMA_BIN" pull "${MODEL}"
fi

# ── 3. Install Python dependencies ────────────────────────────────────────
echo "[setup] Installing Python packages..."
pip install -e ".[dev]" 2>/dev/null || pip install -e .

# ── 4. Done ───────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Setup complete!  Model: ${MODEL}"
echo ""
echo " Usage:"
echo '   sports-analyst "How many home runs did Hank Aaron hit on Sundays?"'
echo ""
echo " Optional: set CFBD_API_KEY for NCAA football data"
echo "   export CFBD_API_KEY=your_key_from_collegefootballdata.com"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
