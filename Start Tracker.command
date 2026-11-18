#!/bin/zsh
# Start the cached VLM and dashboard; stop only our own children on exit.
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Set up first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [[ "${VISUAL_TRACKER_NATIVE:-0}" != 1 ]]; then
  /bin/zsh scripts/launch/build_native_app.sh
  native_pid_file="$PWD/.venv/native-launch-$$.pid"
  cleanup_native() {
    if [[ -f "$native_pid_file" ]]; then
      native_pid=$(cat "$native_pid_file")
      if [[ "$native_pid" == <-> ]]; then
        kill -TERM "$native_pid" 2>/dev/null || true
      fi
      rm -f "$native_pid_file"
    fi
  }
  trap cleanup_native EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  echo "Opening Visual Tracker to request native camera access…"
  open -n -W "$PWD/.venv/Visual Tracker.app" --args "$PWD" "$native_pid_file" &
  native_open_pid=$!
  wait "$native_open_pid"
  exit 0
fi
export HF_HOME="$PWD/.venv/huggingface"
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export YOLO_CONFIG_DIR="$PWD/.venv/ultralytics"
export VLM_URL="http://127.0.0.1:8002/verify"
export PORT="${PORT:-8001}"
vlm_pid=""
dashboard_pid=""
cleanup() {
  for run_pid in "$dashboard_pid" "$vlm_pid"; do
    if [[ -n "$run_pid" ]]; then
      kill -TERM "$run_pid" 2>/dev/null || true
      wait "$run_pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_ready() {
  local run_url="$1" run_pid="$2"
  for run_attempt in {1..120}; do
    if curl --max-time 2 -fsS "$run_url" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$run_pid" ]] && ! kill -0 "$run_pid" 2>/dev/null; then
      echo "Service stopped before it was ready: $run_url"
      return 1
    fi
    sleep 1
  done
  echo "Service is taking too long: $run_url. Check the messages above."
  return 1
}
if ! curl --max-time 2 -fsS http://127.0.0.1:8002/health >/dev/null 2>&1; then
  echo "Loading the local VLM from .venv/huggingface…"
  .venv/bin/python -u vlm_server.py &
  vlm_pid=$!
fi
wait_ready http://127.0.0.1:8002/health "$vlm_pid"
if ! curl --max-time 2 -fsS "http://127.0.0.1:$PORT/api/clips" >/dev/null 2>&1; then
  .venv/bin/python -u dashboard/app.py &
  dashboard_pid=$!
fi
wait_ready "http://127.0.0.1:$PORT/api/clips" "$dashboard_pid"
curl --max-time 15 -fsS "http://127.0.0.1:$PORT/live/set?verify=true&paused=false" >/dev/null
echo "Ready: http://localhost:$PORT/#live"
echo "Choose your camera or clip in Live. Camera access was granted to Visual Tracker."
echo "Keep this Terminal open. Ctrl-C stops the services started here."
open "http://localhost:$PORT/#live"
while true; do
  for run_pid in "$dashboard_pid" "$vlm_pid"; do
    if [[ -n "$run_pid" ]] && ! kill -0 "$run_pid" 2>/dev/null; then
      echo "A service stopped. Check the messages above."
      exit 1
    fi
  done
  sleep 1
done
