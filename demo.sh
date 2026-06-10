#!/usr/bin/env bash
# demo.sh — build, start, and exercise the inference server end-to-end.
set -euo pipefail

BOLD="\033[1m"
GRN="\033[1;32m"
CYN="\033[1;36m"
RED="\033[1;31m"
RST="\033[0m"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
MODELS_DIR="$PROJECT_DIR/models"
MODEL="$MODELS_DIR/tiny_classifier.onnx"
PORT="${PORT:-8080}"
BASE="http://localhost:$PORT"

step() { echo -e "\n${CYN}${BOLD}▶  $*${RST}"; }
ok()   { echo -e "${GRN}✓  $*${RST}"; }
fail() { echo -e "${RED}✗  $*${RST}" >&2; exit 1; }

# ── 1. Build ──────────────────────────────────────────────────────────────────
if [[ ! -f "$BUILD_DIR/inference_server" ]]; then
    step "Building inference_server (first run)..."
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
          -Wno-dev --log-level=WARNING 2>&1 | grep -v "^--"
    cmake --build "$BUILD_DIR" --parallel 2>&1 | tail -5
    ok "Build complete → $BUILD_DIR/inference_server"
else
    ok "Binary already built — skipping cmake (delete build/ to rebuild)"
fi

# ── 2. Test model ─────────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
    step "Generating test ONNX model..."
    (cd "$MODELS_DIR" && python3 make_test_model.py)
    ok "Model saved → $MODEL"
else
    ok "Model already exists → $MODEL"
fi

# ── 3. Start server ───────────────────────────────────────────────────────────
step "Starting server on port $PORT..."
"$BUILD_DIR/inference_server" "$MODEL" "$PORT" &
SERVER_PID=$!
trap 'echo -e "\n${CYN}Shutting down server (PID $SERVER_PID)...${RST}"; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; ok "Done."' EXIT INT TERM

# Poll /health until the server is accepting connections (max 6 s).
READY=0
for i in $(seq 1 30); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then
        READY=1; break
    fi
    sleep 0.2
done
[[ $READY -eq 1 ]] || fail "Server did not start within 6 s"

# ── 4. Health check ───────────────────────────────────────────────────────────
step "GET /health"
curl -s "$BASE/health" | python3 -m json.tool

# ── 5. Single prediction ──────────────────────────────────────────────────────
step "POST /predict  →  input [1.0, 2.0, 3.0, 4.0]"
curl -s -X POST "$BASE/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [1.0, 2.0, 3.0, 4.0], "shape": [1, 4]}' \
  | python3 -m json.tool

# ── 6. Batch prediction ───────────────────────────────────────────────────────
step "POST /predict  →  batch of 3 rows  (shape [3, 4])"
curl -s -X POST "$BASE/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [1,2,3,4, 5,6,7,8, 9,10,11,12], "shape": [3, 4]}' \
  | python3 -m json.tool

# ── 7. Benchmark ──────────────────────────────────────────────────────────────
step "GET /benchmark?n=500"
curl -s "$BASE/benchmark?n=500" | python3 -m json.tool

# ── 8. Error handling demo ────────────────────────────────────────────────────
step "POST /predict  →  wrong shape (expect 400)"
curl -s -X POST "$BASE/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [1.0, 2.0], "shape": [1, 2]}' \
  | python3 -m json.tool

echo
ok "Demo complete."
