#!/bin/bash
# Launch one weight-editing proposer as a headless Codex CLI process inside a macOS Seatbelt sandbox (macOS only).
#
#   launch_codex.sh RUN_DIR PACKET_DIR MODEL EFFORT [PROMPT_FILE]
# Same packet and run folder as launch_claude.sh. Codex runs with its own sandbox and approvals off because macOS
# cannot nest Seatbelt profiles; proposer_codex.sb is the boundary. CODEX_HOME is an isolated directory in the run's
# temp dir with a minimal config.toml (model and effort only: no plugins, MCP servers, hooks or user instructions)
# and a symlink to ~/.codex/auth.json.
# Env (required): WP_MODEL_DIR, WP_VENV. Optional: WP_PYHOME, WP_CODEX (default /opt/homebrew/bin/codex), WP_CONV,
#   WP_TMP, WP_DRY=1, WP_RESET_FILE (a file holding an ISO UTC time: after a usage-limit ending, wait until then),
#   WP_RETRIES (default 12): a conversation that ends on a usage limit or a provider capacity/server error is resumed
#   with a continuation prompt (pointing to notebook.md, state.json and the done files) without counting as a new
#   conversation. Output: RUN_DIR/transcript_<WP_CONV>[_retry<i>].jsonl (codex exec --json events), launcher.log.
set -euo pipefail
RUN=$(cd "$1" && pwd); PACKET=$(cd "$2" && pwd); MODEL=$3; EFFORT=$4; PROMPT_FILE=${5:-}
: "${WP_MODEL_DIR:?set WP_MODEL_DIR}"; : "${WP_VENV:?set WP_VENV}"
MODELDIR=$(cd "$WP_MODEL_DIR" && pwd); VENV=$(cd "$WP_VENV" && pwd)
PYHOME=${WP_PYHOME:-$HOME/.local/share/uv/python}
CODEX=${WP_CODEX:-/opt/homebrew/bin/codex}
RUN_ID=$(basename "$RUN")
TMPD=${WP_TMP:-/private/tmp/wp_sbx_$RUN_ID}
SB=$(cd "$(dirname "$0")" && pwd)/proposer_codex.sb
[ -d "$RUN/requests" ] || { echo "run folder $RUN has no requests/" >&2; exit 2; }
CH=$TMPD/codex_home
mkdir -p "$CH"
printf 'model = "%s"\nmodel_reasoning_effort = "%s"\n' "$MODEL" "$EFFORT" > "$CH/config.toml"
ln -sfn "$HOME/.codex/auth.json" "$CH/auth.json"
PROMPT="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN."
[ -n "$PROMPT_FILE" ] && PROMPT=$(cat "$PROMPT_FILE")
CONTINUE="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN. Your previous conversation was cut off before it finished. Read the run folder's notebook.md, state.json and requests/*.done.json first, then continue from where it stopped."
run_codex() {  # $1 prompt, $2 transcript
  local cmd=(env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$VENV/bin:$(dirname "$CODEX"):/usr/bin:/bin:/usr/sbin:/sbin"
             TERM=dumb LANG=en_US.UTF-8 TMPDIR="$TMPD/" CODEX_HOME="$CH" SHELL=/bin/bash
             /usr/bin/sandbox-exec -f "$SB" -D HOME="$HOME" -D PACKET="$PACKET" -D RUN="$RUN" -D MODEL="$MODELDIR"
             -D VENV="$VENV" -D PYHOME="$PYHOME" -D TMPD="$TMPD"
             "$CODEX" exec --skip-git-repo-check -s danger-full-access -c approval_policy='"never"' -C "$PACKET"
             --add-dir "$RUN" --json "$1")
  if [ -n "${WP_DRY:-}" ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
  (cd "$PACKET" && "${cmd[@]}" < /dev/null > "$2" 2>> "$RUN/launcher.log")
}
ends_on() {  # $1 transcript, $2 regex: 0 if the last failure event matches and no turn completed after it
  [ -f "$1" ] && /usr/bin/python3 -I - "$1" "$2" <<'PY'
import json, re, sys
bad = re.compile(sys.argv[2], re.I)
hit = False
for line in open(sys.argv[1]):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if isinstance(e, dict) and e.get("type") in ("error", "turn.failed") and bad.search(json.dumps(e)):
        hit = True
    elif isinstance(e, dict) and e.get("type") == "turn.completed":
        hit = False
sys.exit(0 if hit else 1)
PY
}
USAGE_RE="usage limit|rate.?limit|quota|429|too many requests|insufficient|credits"
TRANSIENT_RE="at capacity|overloaded|server error|internal error|50[234]|stream disconnected"
wait_reset() {
  local now t
  now=$(date -u +%s)
  t=$( [ -n "${WP_RESET_FILE:-}" ] && [ -f "$WP_RESET_FILE" ] && date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$(cat "$WP_RESET_FILE")" +%s 2>/dev/null || echo 0)
  if [ "$t" -gt "$now" ]; then sleep $((t - now + 180)); else sleep 1800; fi
}
OUT=$RUN/transcript_${WP_CONV:-1}.jsonl
echo "[$(date -u +%FT%TZ)] launch $RUN_ID codex model=$MODEL effort=$EFFORT packet=$PACKET" >> "$RUN/launcher.log"
run_codex "$PROMPT" "$OUT" && rc=0 || rc=$?
[ -n "${WP_DRY:-}" ] && exit 0
echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
for retry in $(seq 1 "${WP_RETRIES:-12}"); do
  [ ! -f "$RUN/final_answer.md" ] || break
  if ends_on "$OUT" "$USAGE_RE"; then
    echo "[$(date -u +%FT%TZ)] usage limit: waiting" >> "$RUN/launcher.log"; wait_reset
  elif ends_on "$OUT" "$TRANSIENT_RE"; then
    echo "[$(date -u +%FT%TZ)] provider error: waiting 10 min" >> "$RUN/launcher.log"; sleep 600
  else
    break
  fi
  OUT=$RUN/transcript_${WP_CONV:-1}_retry$retry.jsonl
  echo "[$(date -u +%FT%TZ)] relaunch (retry $retry)" >> "$RUN/launcher.log"
  run_codex "$CONTINUE" "$OUT" && rc=0 || rc=$?
  echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
done
exit $rc
