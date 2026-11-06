#!/bin/bash
# Launch one weight-editing proposer as a headless Claude Code process inside a macOS Seatbelt sandbox (macOS only).
#
#   launch_claude.sh RUN_DIR PACKET_DIR MODEL EFFORT [PROMPT_FILE]
#     RUN_DIR     the server's run folder (weightprog init-run), read/write for the proposer
#     PACKET_DIR  the packet (weightprog packet), read-only except scratch/; also the working directory
#     MODEL       claude model alias or id;  EFFORT  low | medium | high | xhigh | max
#     PROMPT_FILE optional; default: "Read brief.md in the current directory and follow it exactly. ..."
# Env (required): WP_MODEL_DIR (model directory, read-only), WP_VENV (Python environment the proposer runs)
# Env (optional): WP_PYHOME (base interpreter dir; default ~/.local/share/uv/python), WP_CLAUDE (claude binary;
#   default ~/.local/bin/claude), WP_TOOLS (default "Bash Read Write Edit Glob Grep"), WP_MAX_USD (--max-budget-usd),
#   WP_CONV (conversation number, default 1), WP_TMP (default /private/tmp/wp_sbx_<run name>), WP_DRY=1 (print only).
# Output: RUN_DIR/transcript_<WP_CONV>.jsonl (stream-json, the audit's input) and RUN_DIR/launcher.log.
# Isolation: env cleared (env -i); no user/project settings (--setting-sources ""), so no hooks, permissions or
# plugins; no MCP servers; tools restricted; permission prompts bypassed because the OS sandbox is the boundary.
set -euo pipefail
RUN=$(cd "$1" && pwd); PACKET=$(cd "$2" && pwd); MODEL=$3; EFFORT=$4; PROMPT_FILE=${5:-}
: "${WP_MODEL_DIR:?set WP_MODEL_DIR}"; : "${WP_VENV:?set WP_VENV}"
MODELDIR=$(cd "$WP_MODEL_DIR" && pwd); VENV=$(cd "$WP_VENV" && pwd)
PYHOME=${WP_PYHOME:-$HOME/.local/share/uv/python}
CLAUDE=${WP_CLAUDE:-$HOME/.local/bin/claude}
RUN_ID=$(basename "$RUN")
TMPD=${WP_TMP:-/private/tmp/wp_sbx_$RUN_ID}
SB=$(cd "$(dirname "$0")" && pwd)/proposer_claude.sb
[ -d "$RUN/requests" ] || { echo "run folder $RUN has no requests/: run weightprog init-run first" >&2; exit 2; }
[ -f "$PACKET/brief.md" ] || { echo "packet $PACKET has no brief.md: run weightprog packet first" >&2; exit 2; }
mkdir -p "$TMPD"
SLUG=$(printf '%s' "$PACKET" | sed 's/[^A-Za-z0-9]/-/g')  # Claude Code's per-cwd state dir name
PROMPT="Read brief.md in the current directory and follow it exactly. Your run folder is $RUN."
[ -n "$PROMPT_FILE" ] && PROMPT=$(cat "$PROMPT_FILE")
TOOLS=${WP_TOOLS:-"Bash Read Write Edit Glob Grep"}
EXTRA=()
[ -n "${WP_MAX_USD:-}" ] && EXTRA+=(--max-budget-usd "$WP_MAX_USD")
CMD=(env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$VENV/bin:/usr/bin:/bin:/usr/sbin:/sbin" TERM=dumb
     LANG=en_US.UTF-8 TMPDIR="$TMPD/" CLAUDE_CODE_TMPDIR="$TMPD" SHELL=/bin/bash
     /usr/bin/sandbox-exec -f "$SB" -D HOME="$HOME" -D PACKET="$PACKET" -D RUN="$RUN" -D MODEL="$MODELDIR"
     -D VENV="$VENV" -D PYHOME="$PYHOME" -D TMPD="$TMPD" -D SLUG="$SLUG"
     "$CLAUDE" -p "$PROMPT" --model "$MODEL" --effort "$EFFORT" --permission-mode bypassPermissions
     --setting-sources "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --tools $TOOLS --add-dir "$RUN"
     --output-format stream-json --verbose ${EXTRA[@]+"${EXTRA[@]}"})
if [ -n "${WP_DRY:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
echo "[$(date -u +%FT%TZ)] launch $RUN_ID model=$MODEL effort=$EFFORT tools=[$TOOLS] packet=$PACKET" >> "$RUN/launcher.log"
cd "$PACKET"
"${CMD[@]}" < /dev/null > "$RUN/transcript_${WP_CONV:-1}.jsonl" 2>> "$RUN/launcher.log" && rc=0 || rc=$?
echo "[$(date -u +%FT%TZ)] exit rc=$rc" >> "$RUN/launcher.log"
exit $rc
