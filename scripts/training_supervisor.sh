#!/usr/bin/env bash
# 30-minute training supervisor: health checks, service remediation, git autosync,
# structured state snapshots.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIB="${ROOT}/scripts/lib"
STATE_DIR="${ROOT}/outputs/runs/_supervisor"
LOG_FILE="${STATE_DIR}/supervisor.log"
STATE_JSON="${STATE_DIR}/state.json"
INTERVAL_S="${SUPERVISOR_INTERVAL_S:-1800}"

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
WANDB_GROUP="${WANDB_GROUP:-overnight_$(date +%Y-%m-%d)}"
MAX_RUNS="${MAX_RUNS:-6}"

mkdir -p "$STATE_DIR"

# shellcheck disable=SC1091
source "${LIB}/git_autosync.sh"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] supervisor $*" | tee -a "$LOG_FILE"
}

ensure_overnight() {
  if pgrep -af "[a]utonomous_train.py" >/dev/null 2>&1; then
    return 0
  fi
  log "REMEDIATE orchestrator not running; restarting overnight session"
  tmux kill-session -t overnight 2>/dev/null || true
  tmux new-session -d -s overnight -c "$ROOT" "bash -lc '
export LD_LIBRARY_PATH=\"/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}\"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate genesis
set -a && source .env && set +a
exec python scripts/autonomous_train.py \
  --wait-for-trainer \
  --max-runs '"${MAX_RUNS}"' \
  --wandb-group '"${WANDB_GROUP}"' \
  >> outputs/runs/_orchestrator/orchestrator.log 2>&1
'"
}

ensure_git_sync() {
  if pgrep -af "[g]it_sync_watcher.sh" >/dev/null 2>&1; then
    return 0
  fi
  log "REMEDIATE git_sync watcher not running; restarting"
  tmux kill-session -t git_sync 2>/dev/null || true
  bash "${ROOT}/scripts/start_git_sync.sh" >>"$LOG_FILE" 2>&1 || true
}

collect_state() {
  SUPERVISOR_ROOT="$ROOT" \
  SUPERVISOR_REMOTE="$REMOTE" \
  SUPERVISOR_BRANCH="$BRANCH" \
  SUPERVISOR_STATE_JSON="$STATE_JSON" \
  SUPERVISOR_ISSUES="${issues[*]:-}" \
  SUPERVISOR_ACTIONS="${actions[*]:-}" \
    python3 <<'PY'
import json, os, re, subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["SUPERVISOR_ROOT"])
remote = os.environ["SUPERVISOR_REMOTE"]
branch = os.environ["SUPERVISOR_BRANCH"]
issues = [x for x in os.environ.get("SUPERVISOR_ISSUES", "").split() if x]
actions = [x for x in os.environ.get("SUPERVISOR_ACTIONS", "").split() if x]

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

head = sh("git rev-parse HEAD").stdout.strip()
remote_sha = sh(f"git rev-parse {remote}/{branch} 2>/dev/null").stdout.strip()
ahead = sh(f"git rev-list --count {remote_sha}..HEAD 2>/dev/null").stdout.strip() or "0"
behind = sh(f"git rev-list --count HEAD..{remote_sha} 2>/dev/null").stdout.strip() or "0"
git_clean = not sh("git status --porcelain").stdout.strip()

tmux = sh("tmux ls 2>/dev/null").stdout.strip().splitlines()
tmux_sessions = [l.split(":")[0] for l in tmux if ":" in l]

trainers = []
for line in sh("pgrep -af standalone_trainer.py").stdout.strip().splitlines():
    if "standalone_trainer" not in line:
        continue
    pid = int(line.split()[0])
    m = re.search(r"--run-id\s+(\S+)", line)
    run_id = m.group(1) if m else "unknown"
    step = nan = stale = None
    log_path = root / "outputs" / "runs" / run_id / "run.log"
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        ms = re.findall(r"standalone_trainer INFO: step=(\d+) buffer=", text)
        step = int(ms[-1]) if ms else None
        nan = text.count("NaN constraint forces")
        stale = round((datetime.now().timestamp() - log_path.stat().st_mtime) / 60, 1)
    trainers.append({
        "pid": pid, "run_id": run_id, "step": step,
        "nan_warnings": nan, "log_stale_minutes": stale,
    })

orchestrator = bool(re.search(r"autonomous_train", sh("pgrep -af autonomous_train.py").stdout))
git_sync = bool(re.search(r"git_sync_watcher", sh("pgrep -af git_sync_watcher.sh").stdout))
gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null").stdout.strip()

state = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "git": {
        "head": head[:12] if head else None,
        "remote_head": remote_sha[:12] if remote_sha else None,
        "branch": branch,
        "ahead": int(ahead),
        "behind": int(behind),
        "clean": git_clean,
    },
    "services": {
        "orchestrator": orchestrator,
        "git_sync": git_sync,
        "tmux_sessions": tmux_sessions,
    },
    "trainers": trainers,
    "gpu": gpu or None,
    "issues": issues,
    "actions": actions,
}
Path(os.environ["SUPERVISOR_STATE_JSON"]).write_text(json.dumps(state, indent=2))
print(json.dumps(state, indent=2))
PY
}

run_health_checks() {
  local issues=()

  # NaN storms
  while IFS= read -r rid; do
    [ -n "$rid" ] || continue
    issues+=("nan_storm:${rid}")
    log "ISSUE NaN storm on ${rid}"
  done < <(SUPERVISOR_ROOT="$ROOT" python3 -c "
import json,re,subprocess,os
from pathlib import Path
root=os.environ['SUPERVISOR_ROOT']
for line in subprocess.run(['pgrep','-af','standalone_trainer.py'],capture_output=True,text=True).stdout.splitlines():
    m=re.search(r'--run-id\s+(\S+)', line)
    if not m: continue
    p=Path(root)/'outputs'/'runs'/m.group(1)/'run.log'
    if p.exists() and p.read_text(errors='replace').count('NaN constraint forces')>100:
        print(m.group(1))
")

  # Stalled trainers (log not updated in 15+ min while process alive)
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    local rid stale
    rid=$(echo "$line" | awk '{print $1}')
    stale=$(echo "$line" | awk '{print $2}')
    issues+=("trainer_stalled:${rid}")
    log "ISSUE trainer stalled: ${rid} (${stale} min since log update)"
  done < <(SUPERVISOR_ROOT="$ROOT" python3 -c "
import re,subprocess,os
from datetime import datetime
from pathlib import Path
root=os.environ['SUPERVISOR_ROOT']
for line in subprocess.run(['pgrep','-af','standalone_trainer.py'],capture_output=True,text=True).stdout.splitlines():
    m=re.search(r'--run-id\s+(\S+)', line)
    if not m: continue
    p=Path(root)/'outputs'/'runs'/m.group(1)/'run.log'
    if p.exists():
        stale=(datetime.now().timestamp()-p.stat().st_mtime)/60
        if stale>15: print(m.group(1), round(stale,1))
")

  # OOM in logs
  while IFS= read -r rid; do
    [ -n "$rid" ] || continue
    issues+=("oom:${rid}")
    log "ISSUE OOM detected in ${rid} log"
  done < <(SUPERVISOR_ROOT="$ROOT" python3 -c "
import re,subprocess,os
from pathlib import Path
root=os.environ['SUPERVISOR_ROOT']
for line in subprocess.run(['pgrep','-af','standalone_trainer.py'],capture_output=True,text=True).stdout.splitlines():
    m=re.search(r'--run-id\s+(\S+)', line)
    if not m: continue
    p=Path(root)/'outputs'/'runs'/m.group(1)/'run.log'
    if p.exists():
        t=p.read_text(errors='replace')
        if 'CUDA out of memory' in t or 'OutOfMemoryError' in t:
            print(m.group(1))
")

  printf '%s\n' "${issues[@]}"
}

run_cycle() {
  local issues=() actions=()
  log "=== cycle start ==="

  # Run checkin (human-readable snapshot)
  bash "${ROOT}/scripts/training_checkin.sh" >>"${STATE_DIR}/checkins.log" 2>&1 || true

  fix_script_crlf "$LOG_FILE"
  actions+=("crlf_check")

  if safe_git_commit "$LOG_FILE"; then
    actions+=("committed_local")
  fi

  if safe_git_sync "$LOG_FILE"; then
    actions+=("git_synced")
  else
    issues+=("git_sync_failed")
    log "ISSUE git sync failed (see ${LOG_FILE})"
  fi

  # Service remediation
  if ! pgrep -af "[a]utonomous_train.py" >/dev/null 2>&1; then
    issues+=("orchestrator_down")
    ensure_overnight
    actions+=("restarted_overnight")
  fi

  if ! pgrep -af "[g]it_sync_watcher.sh" >/dev/null 2>&1; then
    issues+=("git_sync_down")
    ensure_git_sync
    actions+=("restarted_git_sync")
  fi

  # Kill stale tmux session from Jun 19 if no useful process
  if tmux has-session -t 0 2>/dev/null; then
    tmux kill-session -t 0 2>/dev/null && actions+=("killed_stale_tmux_0") && log "REMEDIATE killed stale tmux session 0"
  fi

  # Health checks
  while IFS= read -r issue; do
    [ -n "$issue" ] || continue
    issues+=("$issue")
  done < <(run_health_checks)

  collect_state
  log "=== cycle end issues=[${issues[*]:-none}] actions=[${actions[*]:-none}] ==="
}

log "supervisor start interval=${INTERVAL_S}s branch=${BRANCH}"
run_cycle || log "initial cycle error (continuing)"
while true; do
  sleep "$INTERVAL_S"
  run_cycle || log "cycle error (continuing)"
done
