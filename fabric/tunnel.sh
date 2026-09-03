#!/usr/bin/env bash
# Keep the dashboard tunnel up: laptop:LOCAL_PORT -> CTRL:REMOTE_PORT.
#
# The orchestrator runs on the FABRIC control node, so the dashboard is only
# reachable through this forward (FABRIC management addresses are IPv6-only,
# which most venue networks do not route). A bare `ssh -L` dies quietly on a
# network blip and takes the display with it, so this supervises it: if the
# local port stops listening, the tunnel is re-established.
#
#   ./fabric/tunnel.sh            # supervise in the foreground (Ctrl-C to stop)
#   ./fabric/tunnel.sh status     # is it up?
#   ./fabric/tunnel.sh stop       # tear it down
#
# The job keeps running on the testbed regardless of this tunnel's state.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PORT="${LOCAL_PORT:-8091}"
REMOTE_PORT="${REMOTE_PORT:-8080}"
SSH_CONFIG="${SSH_CONFIG:-$HOME/work/fabric_config/ssh_config}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
HOST_FILE="${HOST_FILE:-$HERE/ctrl_host.txt}"
CHECK_EVERY="${CHECK_EVERY:-15}"

if [[ -n "${CTRL_HOST:-}" ]]; then
  HOST="$CTRL_HOST"
elif [[ -r "$HOST_FILE" ]]; then
  HOST="$(tr -d '[:space:]' < "$HOST_FILE")"
else
  echo "No control-node host. Set CTRL_HOST=<user@addr> or write it to $HOST_FILE" >&2
  exit 1
fi

listening() { lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; }
tunnel_pids() { pgrep -f "$LOCAL_PORT:localhost:$REMOTE_PORT" 2>/dev/null; }

start_tunnel() {
  ssh -N -f \
      -L "$LOCAL_PORT:localhost:$REMOTE_PORT" \
      -F "$SSH_CONFIG" -i "$SSH_KEY" \
      -o StrictHostKeyChecking=no \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o ConnectTimeout=20 \
      "$HOST" 2>/dev/null
}

case "${1:-supervise}" in
  status)
    if listening; then
      echo "up: http://localhost:$LOCAL_PORT -> $HOST:$REMOTE_PORT (pid $(tunnel_pids | tr '\n' ' '))"
    else
      echo "down"
      exit 1
    fi
    ;;
  stop)
    # shellcheck disable=SC2046
    [[ -n "$(tunnel_pids)" ]] && kill $(tunnel_pids) && echo "stopped" || echo "not running"
    ;;
  supervise)
    echo "supervising $LOCAL_PORT -> $HOST:$REMOTE_PORT (check every ${CHECK_EVERY}s)"
    trap 'echo; echo "leaving tunnel running; use \"$0 stop\" to remove it"; exit 0' INT TERM
    while true; do
      if ! listening; then
        # Clear any half-dead forwarder before rebinding the port.
        # shellcheck disable=SC2046
        [[ -n "$(tunnel_pids)" ]] && kill $(tunnel_pids) 2>/dev/null
        printf '%s reconnecting… ' "$(date '+%H:%M:%S')"
        if start_tunnel && sleep 2 && listening; then
          echo "up"
        else
          echo "failed (retrying in ${CHECK_EVERY}s)"
        fi
      fi
      sleep "$CHECK_EVERY"
    done
    ;;
  *)
    echo "usage: $0 [supervise|status|stop]" >&2
    exit 2
    ;;
esac
